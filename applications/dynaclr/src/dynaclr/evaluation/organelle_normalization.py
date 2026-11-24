"""Persist control median/MAD representations without replacing encoder X."""

import hashlib
import json

import numpy as np
import pandas as pd

NORMALIZED_KEY = "X_control_mad"
CONTRACT_KEY = "organelle_normalization"
RECIPE = "time_matched_control_median_mad_v1"
RECIPES = {
    "time_matched": RECIPE,
    "per_timepoint": "per_timepoint_control_median_mad_v1",
    "pooled": "pooled_control_median_mad_v1",
}


def normalization_mode(m):
    settings = m.get("normalization", {})
    if not isinstance(settings, dict) or set(settings) - {"mode"}:
        raise ValueError("normalization must be a mapping with only the mode option")
    mode = settings.get("mode", "time_matched")
    if not isinstance(mode, str) or mode not in RECIPES:
        raise ValueError(f"normalization.mode must be one of {', '.join(RECIPES)}")
    return mode


def array_checksum(x):
    x = np.ascontiguousarray(x)
    digest = hashlib.sha256(str((x.shape, x.dtype.str)).encode())
    digest.update(x.view(np.uint8))
    return digest.hexdigest()


def source_checksum(data, m):
    columns = ["cell_id", "marker", "experiment", "control_group", "perturbation", "hours_post_perturbation"]
    obs = data.obs[columns].copy()
    for column in columns[:-1]:
        obs[column] = obs[column].astype(str)
    obs[columns[-1]] = obs[columns[-1]].astype(float)
    digest = hashlib.sha256(array_checksum(data.X).encode())
    digest.update(pd.util.hash_pandas_object(obs, index=False).to_numpy().tobytes())
    digest.update(json.dumps([RECIPES[normalization_mode(m)], m["feature_space"], m["control_label"]]).encode())
    return digest.hexdigest()


def control_normalize(x, obs, control, *, contracts=None, mode="time_matched"):
    """Fit controls per acquisition, optionally splitting or pooling timepoints."""
    from scipy.ndimage import gaussian_filter1d

    normalization_mode({"normalization": {"mode": mode}})
    x = np.asarray(x, dtype=np.float32).copy()
    for key, group in obs.groupby(["experiment", "control_group"]):
        rows = group.index.to_numpy()
        times = group.hours_post_perturbation.to_numpy(float)
        is_control = group.perturbation.eq(control).to_numpy()
        grid = np.unique(times)
        if mode != "time_matched":
            subsets = [times == time for time in grid] if mode == "per_timepoint" else [np.ones(len(rows), bool)]
            for subset in subsets:
                controls = x[rows[is_control & subset]]
                if not len(controls):
                    raise ValueError(f"Missing controls for group {key} at HPI {np.unique(times[subset]).tolist()}")
                center = np.median(controls, axis=0)
                scale = 1.4826 * np.median(np.abs(controls - center), axis=0)
                positive = scale[scale > 0]
                if not len(positive):
                    raise ValueError(f"Every control dimension has zero MAD for group {key}")
                scale = np.maximum(scale, np.quantile(positive, 0.01))
                x[rows[subset]] = (x[rows[subset]] - center) / scale
                if contracts is not None:
                    contracts.append(
                        dict(
                            experiment=str(key[0]),
                            control_group=str(key[1]),
                            hpi_grid=np.unique(times[subset]),
                            control_centers=center[None, :],
                            control_mad=scale,
                            n_control=int(len(controls)),
                        )
                    )
            continue
        centers = np.full((len(grid), x.shape[1]), np.nan, dtype=np.float32)
        for i, time in enumerate(grid):
            take = is_control & (times == time)
            if take.any():
                centers[i] = np.median(x[rows[take]], axis=0)
        occupied = np.flatnonzero(np.isfinite(centers[:, 0]))
        if len(occupied) < 2:
            raise ValueError("Control normalization requires controls at two HPI values per group")
        for d in range(x.shape[1]):
            centers[:, d] = np.interp(grid, grid[occupied], centers[occupied, d])
        centers = gaussian_filter1d(centers, sigma=1.0, axis=0, mode="nearest")
        x[rows] -= centers[np.searchsorted(grid, times)]
        controls = x[rows[is_control]]
        scale = 1.4826 * np.median(np.abs(controls - np.median(controls, axis=0)), axis=0)
        positive = scale[scale > 0]
        if not len(positive):
            raise ValueError("Every control dimension has zero MAD")
        scale = np.maximum(scale, np.quantile(positive, 0.01))
        x[rows] /= scale
        if contracts is not None:
            contracts.append(
                dict(
                    experiment=str(key[0]),
                    control_group=str(key[1]),
                    hpi_grid=grid,
                    control_centers=centers,
                    control_mad=scale,
                    n_control=int(is_control.sum()),
                )
            )
    return x


def normalize_embeddings(data, m):
    """Fit each marker/acquisition's own controls, always starting from raw X."""
    mode = normalization_mode(m)
    normalized = np.empty(data.shape, dtype=np.float32)
    groups = {}
    for marker in sorted(data.obs.marker.unique()):
        rows = np.flatnonzero(data.obs.marker.eq(marker))
        contracts = []
        normalized[rows] = control_normalize(
            data.X[rows], data.obs.iloc[rows].reset_index(drop=True), m["control_label"], contracts=contracts, mode=mode
        )
        for contract in contracts:
            contract["marker"] = str(marker)
            groups[str(len(groups))] = contract
    data.obsm[NORMALIZED_KEY] = normalized
    data.uns[CONTRACT_KEY] = dict(
        recipe=RECIPES[mode],
        mode=mode,
        source_checksum=source_checksum(data, m),
        normalized_checksum=array_checksum(normalized),
        control_label=m["control_label"],
        feature_space=m["feature_space"],
        source="X",
        output=f"obsm/{NORMALIZED_KEY}",
        smooth_sigma_timepoints=1.0 if mode == "time_matched" else 0.0,
        mad_scale_factor=1.4826,
        mad_floor_quantile=0.01,
        group_by=["marker", "experiment", "control_group"]
        + (["hours_post_perturbation"] if mode == "per_timepoint" else []),
        groups=groups,
    )


def require_normalized(data, m):
    """Reject absent/stale representations rather than silently normalizing twice."""
    contract = data.uns.get(CONTRACT_KEY, {})
    if (
        NORMALIZED_KEY not in data.obsm
        or contract.get("recipe") != RECIPES[normalization_mode(m)]
        or contract.get("mode", "time_matched") != normalization_mode(m)
        or contract.get("source_checksum") != source_checksum(data, m)
        or contract.get("normalized_checksum") != array_checksum(data.obsm[NORMALIZED_KEY])
    ):
        raise ValueError("Missing or stale control normalization; run the normalize stage on this cohort first")
    return data.obsm[NORMALIZED_KEY]


def export_embeddings(data, output, *, normalized=False):
    """One row per observation; explicit raw and normalized Arrow vector columns."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    frame = data.obs.copy()
    frame["observation_id"] = data.obs_names.to_numpy()
    reserved = {"embeddings", "ctrl-norm-embedding"} & set(frame.columns)
    if reserved:
        raise ValueError(f"Reserved embedding column names in observation metadata: {sorted(reserved)}")
    table = pa.Table.from_pandas(frame, preserve_index=False)
    vectors = {"embeddings": data.X}
    if normalized:
        vectors["ctrl-norm-embedding"] = data.obsm[NORMALIZED_KEY]
    for name, values in vectors.items():
        # ListArray avoids materializing millions of Python float objects.
        values = np.asarray(values)
        offsets = np.arange(len(values) + 1, dtype=np.int64) * values.shape[1]
        array = pa.ListArray.from_arrays(offsets, pa.array(values.reshape(-1)))
        table = table.append_column(name, array)
    pq.write_table(table, output / "embeddings.parquet")
    if normalized:
        (output / "normalization.json").write_text(
            json.dumps(
                data.uns[CONTRACT_KEY],
                indent=2,
                default=lambda value: value.tolist() if isinstance(value, np.ndarray) else value.item(),
            )
            + "\n"
        )
