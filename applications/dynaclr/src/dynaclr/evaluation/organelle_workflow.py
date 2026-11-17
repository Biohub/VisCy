"""Portable organelle analysis bundles. Run with python -m ...organelle_workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml
from sklearn.decomposition import PCA


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load_manifest(path):
    path = Path(path).resolve()
    m = yaml.safe_load(path.read_text())
    if m.get("schema_version") != 1:
        raise ValueError("schema_version must be 1")
    root = (path.parent / m.get("root", ".")).resolve()
    for key in ("embeddings", "cell_index", "training_config", "checkpoint"):
        if m.get(key):
            m[key] = str((root / m[key]).resolve())
    for model in m.get("classifiers", []):
        model["path"] = str((root / model["path"]).resolve())
    return m


def canonical_obs(frame, m):
    """Apply explicit column aliases; never infer treatment or timing from paths."""
    frame = frame.rename(columns=m.get("columns", {})).copy()
    for key, value in m.get("metadata", {}).items():
        if key not in frame:
            frame[key] = value
    if "hours_post_perturbation" not in frame and "timing" in m:
        timing = m["timing"]
        frame["hours_post_perturbation"] = frame["t"] * timing["interval_minutes"] / 60 + timing["first_frame_hpi"]
    required = [
        "cell_id",
        "experiment",
        "marker",
        "perturbation",
        "control_group",
        "replicate",
        "hours_post_perturbation",
    ]
    missing = set(required) - set(frame)
    if missing:
        raise ValueError(f"Missing metadata: {sorted(missing)}")
    if frame[required].isna().any().any():
        raise ValueError("Required metadata contains null values")
    if frame.duplicated(["experiment", "cell_id", "marker"]).any():
        raise ValueError("Observation keys must be unique per experiment/cell_id/marker")
    if not np.isfinite(frame.hours_post_perturbation.to_numpy(float)).all():
        raise ValueError("HPI must be finite")
    return frame


def validate_cells(frame, normalization_level="timepoint_statistics"):
    from viscy_data.cell_index import validate_cell_index

    validate_cell_index(frame)
    from iohub.ngff import open_ome_zarr

    for (store, well, fov), group in frame.groupby(["store_path", "well", "fov"]):
        with open_ome_zarr(f"{store}/{well}/{fov}", mode="r") as pos:
            normalization = pos.zattrs.get("normalization", {})
            for index, row in group.iterrows():
                stats = normalization.get(row.channel_name, {}).get(normalization_level, {})
                if normalization_level == "timepoint_statistics":
                    stats = stats.get(str(int(row.t)))
                if not stats or not np.isfinite(stats.get("std", np.nan)) or stats["std"] <= 0:
                    raise ValueError("Missing or invalid per-frame image normalization metadata")
                for name in ["mean", "std", "median", "iqr", "min", "max"]:
                    frame.loc[index, "norm_" + name] = stats[name]
            unknown = set(group.channel_name) - set(pos.channel_names)
            if unknown:
                raise ValueError(f"Unknown image channels: {sorted(unknown)}")
            for column, size in zip(
                ["T_shape", "C_shape", "Z_shape", "Y_shape", "X_shape"], pos["0"].shape, strict=True
            ):
                if column not in group or not group[column].eq(size).all():
                    raise ValueError(f"{column} disagrees with source image")
    for coordinate, bound in [("t", "T_shape"), ("y", "Y_shape"), ("x", "X_shape")]:
        values = frame[coordinate].to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"Invalid {coordinate} coordinates")
        if bound not in frame or (values >= frame[bound]).any():
            raise ValueError(f"Missing or exceeded {bound}")
    if (frame.t % 1 != 0).any():
        raise ValueError("Frame indices must be integers")


def infer(m, output):
    """Reuse the production prediction configuration generator, with explicit outputs."""
    from dynaclr.evaluation.evaluate import _generate_predict_yaml
    from dynaclr.evaluation.evaluate_config import EvaluationConfig

    cells = canonical_obs(pd.read_parquet(m["cell_index"]), m)
    recipe = m.get("inference_recipe", "paper")
    if recipe not in ("paper", "training"):
        raise ValueError("inference_recipe must be paper or training")
    validate_cells(cells, "fov_statistics" if recipe == "paper" else "timepoint_statistics")
    cells.to_parquet(output / "cells.parquet", index=False)
    checkpoint = Path(m["checkpoint"])
    if sha256(checkpoint) != m["checkpoint_sha256"]:
        raise ValueError("Encoder checksum mismatch")
    if recipe == "paper":
        from dynaclr.evaluation.organelle_inference import predict_paper_cells

        destination = predict_paper_cells(cells, checkpoint, output, m.get("predict", {}).get("batch_size", 32))
        return dict(m, embeddings=str(destination), inference_recipe="paper")
    config = EvaluationConfig.model_validate(
        dict(
            training_config=m["training_config"],
            ckpt_path=str(checkpoint),
            cell_index_path=str(output / "cells.parquet"),
            output_dir=str(output),
            steps=["predict"],
            predict=m.get("predict", {"batch_size": 32, "num_workers": 0}),
        )
    )
    (output / "configs").mkdir()
    cfg = _generate_predict_yaml(config, yaml.safe_load(Path(m["training_config"]).read_text()), output)
    resolved = yaml.safe_load(cfg.read_text())
    if config.predict.num_workers == 0:
        resolved["data"]["init_args"]["prefetch_factor"] = None
    init = resolved["data"]["init_args"]
    for axis, patch in zip(["y", "x"], init["yx_patch_size"], strict=True):
        reference = init.get("reference_pixel_size_xy_um")
        scale = reference / cells.pixel_size_xy_um if reference else 1.0
        half = np.round((patch // 2) * scale)
        if ((cells[axis] < half) | (cells[axis] + half > cells[axis.upper() + "_shape"])).any():
            raise ValueError("Cell crop crosses image boundary; filter boundary cells explicitly before inference")
    cfg.write_text(yaml.safe_dump(resolved, sort_keys=False))
    subprocess.run(["viscy", "predict", "-c", str(cfg)], check=True)
    m = dict(m, embeddings=str(output / "embeddings/embeddings.zarr"))
    return m


def classify(data, m):
    import joblib

    for spec in m.get("classifiers", []):
        if spec["feature_space"] != m["feature_space"]:
            raise ValueError("Classifier feature space mismatch")
        if sha256(spec["path"]) != spec["sha256"]:
            raise ValueError("Classifier checksum mismatch")
        model = joblib.load(spec["path"])  # Only trusted, explicitly pinned artifacts.
        mask = data.obs.marker.eq(spec["marker"]).to_numpy()
        if not mask.any():
            raise ValueError(f"Classifier marker absent: {spec['marker']}")
        column = "predicted_" + spec["task"]
        if column not in data.obs:
            data.obs[column] = "unknown"
        data.obs[column] = data.obs[column].astype(object)
        data.obs.loc[mask, column] = model.predict(data.X[mask]).astype(str)
        classes = getattr(model, "classes_", None)
        if classes is None:
            classes = model.classifier.classes_
        probabilities = model.predict_proba(data.X[mask])
        for i, label in enumerate(classes):
            name = f"{column}_probability_{label}"
            if name not in data.obs:
                data.obs[name] = np.nan
            data.obs.loc[mask, name] = probabilities[:, i]


def control_normalize(x, obs, control):
    """Paper control centering/MAD recipe, independently per acquisition group."""
    from scipy.ndimage import gaussian_filter1d

    x = np.asarray(x, dtype=np.float32).copy()
    for _, group in obs.groupby(["experiment", "control_group"]):
        rows = group.index.to_numpy()
        times = group.hours_post_perturbation.to_numpy(float)
        is_control = group.perturbation.eq(control).to_numpy()
        grid = np.unique(times)
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
        x[rows] /= np.maximum(scale, np.quantile(positive, 0.01))
    return x


def analyze(data, m, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from umap import UMAP

    from dynaclr.evaluation.organelle_mmd import summarize_mmd_pairs
    from viscy_utils.evaluation.mmd import median_heuristic

    settings = dict(
        seed=42,
        min_cells=20,
        max_cells=180,
        permutations=200,
        repeats=50,
        representation="control_pca80",
        sampling_unit="observation",
    )
    settings.update(m.get("analysis", {}))
    if settings["sampling_unit"] not in ("observation", "track_id", "lineage_id"):
        raise ValueError("Unsupported sampling_unit")
    if (
        settings["min_cells"] < 2
        or settings["max_cells"] < settings["min_cells"]
        or settings["permutations"] < 2
        or settings["repeats"] < 2
    ):
        raise ValueError("Invalid MMD sampling settings")
    rng = np.random.default_rng(settings["seed"])
    results, samples = [], []
    control = m["control_label"]
    for marker in sorted(data.obs.marker.unique()):
        idx = np.flatnonzero(data.obs.marker.eq(marker))
        obs = data.obs.iloc[idx].reset_index(drop=True)
        x = np.asarray(data.X[idx], dtype=float)
        if len(x) < 4 or not np.isfinite(x).all():
            raise ValueError("Need at least four finite embeddings per marker")
        if settings["representation"] == "control_pca80":
            x = control_normalize(x, obs, control)
            fit_rows = []
            for _, group in obs.groupby(["experiment", "control_group", "perturbation"]):
                fit_rows.extend(rng.choice(group.index, min(5000, len(group)), replace=False))
            pca = PCA(n_components=0.8, svd_solver="full").fit(x[fit_rows])
            x = pca.transform(x)
        elif settings["representation"] != "raw":
            raise ValueError("representation must be control_pca80 or raw")
        scores = PCA(n_components=min(2, x.shape[1], len(x)), svd_solver="full").fit_transform(x)
        if scores.shape[1] < 2:
            scores = np.column_stack([scores, np.zeros(len(x))])
        umap = UMAP(
            random_state=settings["seed"], n_neighbors=min(15, len(x) - 1), init="random", n_jobs=1
        ).fit_transform(x)
        coords = obs.copy()
        coords[["pca_1", "pca_2"]] = scores
        coords[["umap_1", "umap_2"]] = umap
        # Marker names are metadata, not filesystem paths.
        stem = hashlib.sha256(str(marker).encode()).hexdigest()[:12]
        coords.to_parquet(output / f"{stem}_coordinates.parquet", index=False)
        for color in ["perturbation", "infection_state", "predicted_infection_state"]:
            if color not in obs:
                continue
            for name, points in [("pca", scores), ("umap", umap)]:
                fig, ax = plt.subplots(figsize=(5, 4))
                labels = obs[color].fillna("unknown").astype(str)
                for label in sorted(labels.unique()):
                    take = labels.eq(label)
                    ax.scatter(*points[take].T, s=3, alpha=0.5, label=label)
                ax.set(title=f"{marker}: {color}", xlabel=f"{name} 1", ylabel=f"{name} 2")
                ax.legend(markerscale=3)
                for ext in ["png", "pdf"]:
                    fig.savefig(output / f"{stem}_{color}_{name}.{ext}", bbox_inches="tight")
                plt.close(fig)
        bandwidth = median_heuristic(x, x)
        for key, group in obs.groupby(["experiment", "control_group", "hours_post_perturbation"], sort=True):
            for condition in sorted(set(group.perturbation) - {control}):
                c = group.index[group.perturbation.eq(control)].to_numpy()
                t = group.index[group.perturbation.eq(condition)].to_numpy()
                row = dict(
                    marker=marker,
                    experiment=key[0],
                    control_group=key[1],
                    hpi=key[2],
                    treatment=condition,
                    n_control=len(c),
                    n_treated=len(t),
                    bandwidth=bandwidth,
                )
                unit = settings["sampling_unit"]
                if unit != "observation":
                    unit_keys = ["fov_name", unit]
                    if obs[unit_keys].isna().any().any():
                        raise ValueError("Missing temporal identities for requested sampling unit")
                    c = obs.loc[rng.permutation(c)].drop_duplicates(unit_keys).index.to_numpy()
                    t = obs.loc[rng.permutation(t)].drop_duplicates(unit_keys).index.to_numpy()
                n = min(len(c), len(t), settings["max_cells"])
                if n < settings["min_cells"]:
                    results.append(dict(row, status="insufficient_matched_cells"))
                    continue
                c, t = rng.choice(c, n, replace=False), rng.choice(t, n, replace=False)
                summary = summarize_mmd_pairs(
                    x[t],
                    x[c],
                    bandwidth=bandwidth,
                    permutations=settings["permutations"],
                    repeats=settings["repeats"],
                    permutation_seed=settings["seed"] + 300_000,
                    subsample_seed=settings["seed"] + 600_000,
                )
                results.append(dict(row, **summary))
                for label, indices in [("control", c), ("treated", t)]:
                    for i in indices:
                        samples.append(dict(row, group=label, cell_id=obs.iloc[i].cell_id))
    table = pd.DataFrame(results)
    table.to_csv(output / "mmd_curves.csv", index=False)
    pd.DataFrame(samples).to_parquet(output / "mmd_samples.parquet", index=False)
    if len(table):
        for key, group in table[table.status.eq("ok")].groupby(["marker", "experiment", "control_group", "treatment"]):
            fig, ax = plt.subplots()
            group = group.sort_values("hpi")
            ax.plot(group.hpi, group.subsample_median)
            ax.fill_between(group.hpi, group.subsample_q025, group.subsample_q975, alpha=0.2)
            ax.set(xlabel="HPI", ylabel="Null-standardized MMD²", title=" / ".join(map(str, key)))
            stem = hashlib.sha256(repr(key).encode()).hexdigest()[:12]
            for ext in ["png", "pdf"]:
                fig.savefig(output / f"mmd_{stem}.{ext}", bbox_inches="tight")
            plt.close(fig)
    return settings


def run(manifest, output, stage):
    # AnnData's installed writer predates pandas Arrow-string inference.
    with pd.option_context("future.infer_string", False):
        return _run(manifest, output, stage)


def _run(manifest, output, stage):
    m = load_manifest(manifest)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "resolved_manifest.yaml").write_text(yaml.safe_dump(m))
    if stage == "infer":
        m = infer(m, output)
    data = ad.read_zarr(m["embeddings"])
    if stage == "infer":
        cells = pd.read_parquet(output / "cells.parquet")
        keys = ["experiment", "cell_id", "marker"]
        if len(data) != len(cells) or not set(keys).issubset(data.obs):
            raise ValueError("Prediction output does not cover the requested observation cohort")
        joined = data.obs[keys].merge(cells, on=keys, how="left", validate="one_to_one", indicator=True)
        if not joined["_merge"].eq("both").all():
            raise ValueError("Prediction/source identity mismatch")
        joined = joined.drop(columns="_merge")
        joined.index = data.obs.index
        data.obs = joined
    data.obs = canonical_obs(data.obs, m)
    data.obs.index = pd.Index(
        [
            json.dumps(list(key))
            for key in data.obs[["experiment", "cell_id", "marker"]].astype(str).itertuples(index=False, name=None)
        ],
        dtype=object,
        name="observation_id",
    )
    if not isinstance(data.X, np.ndarray):
        data.X = data.X.toarray()
    if not np.isfinite(data.X).all():
        raise ValueError("Nonfinite embeddings")
    classify(data, m)
    data.obs.to_parquet(output / "predictions.parquet")
    data.uns["organelle_feature_space"] = m["feature_space"]
    for column in data.obs:
        if isinstance(data.obs[column].dtype, (pd.StringDtype, pd.CategoricalDtype)):
            data.obs[column] = data.obs[column].astype(object)
    data.obs.index = pd.Index(data.obs.index.to_numpy(dtype=str), dtype=object)
    data.write_zarr(output / "predictions.zarr", convert_strings_to_categoricals=False)
    settings = analyze(data, m, output) if stage == "analyze" else {}
    provenance = dict(
        schema_version=1,
        manifest=m,
        analysis=settings,
        stage=stage,
        representation=settings.get("representation", "encoder X"),
        cohort="all treated observations",
        sampling_unit=settings.get("sampling_unit", "observation"),
        interval="subsampling stability; not biological replicate confidence",
        git_revision=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    )
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    (output / "artifacts.json").write_text(json.dumps(sorted(p.name for p in output.iterdir()), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["infer", "analyze"])
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    run(args.manifest, args.output, args.stage)


if __name__ == "__main__":
    main()
