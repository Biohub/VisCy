"""Build an untracked cell index or pseudo-tracks from nuclear-label centroids.

Each nucleus instance at each frame is one observation. Label values are
per-frame instance IDs, not track identities: they appear only in cell_id and
nucleus_label, and the track/lineage fields stay null. Pseudo-tracks follow the
classical-sampling CSV convention: track_id is the per-frame label value.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from iohub.ngff import open_ome_zarr
from skimage.measure import regionprops_table

from dynaclr.evaluation.organelle_inference import PAPER_RECIPE

TIMING_HELP = (
    "Supply timing: {first_frame_hpi: <hours post perturbation at t=0>, interval_minutes: <frame interval>}. "
    "Ask the data owner; the start HPI differs between datasets and is never assumed."
)
_TIME_UNIT_MINUTES = {"second": 1 / 60, "minute": 1.0, "hour": 60.0}


def require_timing(m):
    timing = m.get("timing")
    if not isinstance(timing, dict) or timing.get("first_frame_hpi") is None or not timing.get("interval_minutes"):
        raise ValueError(TIMING_HELP)
    return float(timing["first_frame_hpi"]), float(timing["interval_minutes"])


def paper_crop_half_px(pixel_size_xy_um):
    """Half of the native-pixel crop the paper recipe reads (TripletDataModule rounding)."""
    patch = round(PAPER_RECIPE["final_yx_patch_size"][0] * PAPER_RECIPE["reference_pixel_size"] / pixel_size_xy_um)
    return patch // 2


def _check_interval(pos, interval_minutes):
    axis = pos.axes[0]
    unit_minutes = _TIME_UNIT_MINUTES.get(getattr(axis, "unit", None))
    if axis.type != "time" or unit_minutes is None:
        return
    recorded = pos.scale[0] * unit_minutes
    if not np.isclose(recorded, interval_minutes, rtol=1e-3):
        raise ValueError(
            f"timing.interval_minutes={interval_minutes} disagrees with the image time scale ({recorded} minutes)"
        )


def _label_frames(pos, where, label_name):
    """Yield (t, props) with nucleus_label, y and x centroids for each frame of 2D labels/<label_name>."""
    if label_name not in pos.zgroup.get("labels", {}):
        raise ValueError(f"{where} has no labels/{label_name}")
    shape = pos["0"].shape
    labels = pos.zgroup["labels"][label_name]["0"]
    if labels.shape[0] != shape[0] or labels.shape[-2:] != shape[-2:]:
        raise ValueError(f"{where} labels/{label_name} shape {labels.shape} != image {shape}")
    if labels.shape[2] != 1:
        raise ValueError("Label centroids need 2D (Z=1) nuclear labels")
    for t in range(shape[0]):
        props = pd.DataFrame(regionprops_table(np.asarray(labels[t, 0, 0]), properties=("label", "centroid")))
        yield t, props.rename(columns={"centroid-0": "y", "centroid-1": "x", "label": "nucleus_label"})


TRACK_COLUMNS = ["track_id", "t", "y", "x", "id", "parent_track_id", "parent_id"]


def tracks_from_labels(store, label_name, output):
    """Write one pseudo-tracking Parquet per FOV from nuclear-label centroids.

    ``track_id`` is the nucleus label value, which restarts each frame and so does
    not link nuclei over time. ``id`` is ``100000`` plus the row number from 1 in
    (t, label) order, unique within the FOV, and parents are -1. Files are ``<output>/<row>/<col>/<fov>/tracks_<row>_<col>_<fov>.parquet``.

    Returns
    -------
    dict[str, int]
        Observation count per FOV.
    """
    counts = {}
    with open_ome_zarr(store, mode="r") as plate:
        for name, pos in plate.positions():
            frames = [props.assign(t=t) for t, props in _label_frames(pos, name, label_name)]
            tracks = pd.concat(frames, ignore_index=True).sort_values(["t", "nucleus_label"], ignore_index=True)
            tracks["track_id"] = tracks.nucleus_label
            tracks["id"] = 100000 + np.arange(1, len(tracks) + 1)
            tracks["parent_track_id"] = -1
            tracks["parent_id"] = -1
            fov_dir = Path(output) / name
            fov_dir.mkdir(parents=True)
            tracks[TRACK_COLUMNS].to_parquet(fov_dir / f"tracks_{name.replace('/', '_')}.parquet", index=False)
            counts[name] = len(tracks)
    if not any(counts.values()):
        raise ValueError(f"No labelled nuclei found in labels/{label_name}")
    return counts


def cells_from_labels(m):
    """Return (cells, excluded) DataFrames in the canonical cell-index schema."""
    first_frame_hpi, interval_minutes = require_timing(m)
    spec = m["labels"]
    if m.get("inference_recipe", "paper") != "paper":
        raise ValueError("cells-from-labels sizes the border filter for inference_recipe: paper")
    channels = {channel["name"]: channel["marker"] for channel in spec["channels"]}
    if len(set(channels.values())) != len(spec["channels"]):
        raise ValueError("Each channel needs its own name and marker; markers are separate feature spaces")
    well_perturbation = {}
    for perturbation, wells in spec["perturbation_wells"].items():
        for well in wells:
            if well in well_perturbation:
                raise ValueError(f"Well {well} is assigned to more than one perturbation")
            well_perturbation[well] = perturbation
    if m["control_label"] not in spec["perturbation_wells"]:
        raise ValueError(f"control_label {m['control_label']!r} has no wells in labels.perturbation_wells")
    label_name = spec.get("label", "nuclei")
    metadata = m["metadata"]
    if "microscope" not in metadata:
        raise ValueError("metadata.microscope is required by the cell-index schema")

    rows = []
    with open_ome_zarr(spec["store"], mode="r") as plate:
        wells = dict(plate.wells())
        unknown = set(well_perturbation) - set(wells)
        if unknown:
            raise ValueError(f"Wells not in store: {sorted(unknown)}")
        for well_name, perturbation in sorted(well_perturbation.items()):
            for fov, pos in wells[well_name].positions():
                missing = set(channels) - set(pos.channel_names)
                if missing:
                    raise ValueError(f"{well_name}/{fov} lacks channels {sorted(missing)}")
                _check_interval(pos, interval_minutes)
                shape = pos["0"].shape
                pixel_xy, pixel_z = pos.scale[-1], pos.scale[-3]
                half = paper_crop_half_px(pixel_xy)
                for t, props in _label_frames(pos, f"{well_name}/{fov}", label_name):
                    if props.empty:
                        continue
                    props["t"] = t
                    props["well"] = well_name
                    props["fov"] = fov
                    props["perturbation"] = perturbation
                    props["exclusion_reason"] = None
                    inside = props.y.between(half, shape[-2] - half, inclusive="neither") & props.x.between(
                        half, shape[-1] - half, inclusive="neither"
                    )
                    props.loc[~inside, "exclusion_reason"] = "crop_crosses_image_border"
                    props["pixel_size_xy_um"] = pixel_xy
                    props["pixel_size_z_um"] = pixel_z
                    for name, size in zip(["T_shape", "C_shape", "Z_shape", "Y_shape", "X_shape"], shape, strict=True):
                        props[name] = size
                    rows.append(props)
    if not rows:
        raise ValueError(f"No labelled nuclei found in labels/{label_name}")
    nuclei = pd.concat(rows, ignore_index=True)
    nuclei["cell_id"] = (
        nuclei.well + "/" + nuclei.fov + ":" + nuclei.nucleus_label.astype(str) + ":" + nuclei.t.astype(str)
    )
    nuclei["store_path"] = str(spec["store"])
    nuclei["tracks_path"] = None
    nuclei["z"] = 0
    nuclei["interval_minutes"] = interval_minutes
    nuclei["hours_post_perturbation"] = first_frame_hpi + nuclei.t * interval_minutes / 60
    for key, value in metadata.items():
        if key not in nuclei:
            nuclei[key] = value
    cells = pd.concat(
        [nuclei.assign(channel_name=channel, marker=marker) for channel, marker in sorted(channels.items())],
        ignore_index=True,
    )
    excluded = cells[cells.exclusion_reason.notna()].reset_index(drop=True)
    cells = cells[cells.exclusion_reason.isna()].drop(columns="exclusion_reason").reset_index(drop=True)
    if cells.empty:
        raise ValueError("Every nucleus was excluded; inspect excluded_cells.parquet")
    return cells, excluded
