"""Prepare independent cell instances from 2D nuclear labels for embedding inference."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from iohub.ngff import open_ome_zarr
from skimage.measure import regionprops_table

from dynaclr.evaluation.organelle_inference import PAPER_RECIPE

_TIME_UNIT_MINUTES = {"second": 1 / 60, "minute": 1.0, "hour": 60.0}
TRACK_COLUMNS = ["track_id", "t", "y", "x", "id", "parent_track_id", "parent_id"]


def _timing(m, pos):
    timing = m.get("timing") or {}
    first = timing.get("first_frame_hpi")
    interval = timing.get("interval_minutes")
    if first is None or not np.isfinite(float(first)):
        raise ValueError("Supply a finite timing.first_frame_hpi; never infer experimental time from frame zero")
    if interval is None and pos["0"].shape[0] == 1:
        return float(first), 0.0
    if interval is None or not np.isfinite(float(interval)) or float(interval) <= 0:
        raise ValueError("Supply a positive timing.interval_minutes for multiple frames")
    interval = float(interval)
    axis = pos.axes[0]
    unit = _TIME_UNIT_MINUTES.get(getattr(axis, "unit", None))
    if pos["0"].shape[0] > 1 and axis.type == "time" and unit is not None:
        if not np.isclose(pos.scale[0] * unit, interval, rtol=1e-3):
            raise ValueError("timing.interval_minutes disagrees with the image time scale")
    return float(first), interval


def _label_frames(pos, where, label_name):
    if label_name not in pos.zgroup.get("labels", {}):
        raise ValueError(f"{where} has no labels/{label_name}")
    shape = pos["0"].shape
    labels = pos.zgroup["labels"][label_name]["0"]
    if labels.ndim != 5 or labels.shape[1:3] != (1, 1):
        raise ValueError(f"{where} labels must have shape (T, 1, 1, Y, X)")
    if labels.shape[0] != shape[0] or labels.shape[-2:] != shape[-2:]:
        raise ValueError(f"{where} label dimensions disagree with the source image")
    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"{where} labels must contain nonnegative integers")
    for t in range(shape[0]):
        image = np.asarray(labels[t, 0, 0])
        if (image < 0).any():
            raise ValueError(f"{where} labels must contain nonnegative integers")
        props = pd.DataFrame(regionprops_table(image, properties=("label", "centroid")))
        yield t, props.rename(columns={"centroid-0": "y", "centroid-1": "x", "label": "nucleus_label"})


def _inside_crop(cells, shape, pixel_size, data_init):
    inside = np.ones(len(cells), dtype=bool)
    if data_init is None:
        patch = PAPER_RECIPE["final_yx_patch_size"]
        scale = PAPER_RECIPE["reference_pixel_size"] / pixel_size
    else:
        patch = data_init["yx_patch_size"]
        reference = data_init.get("reference_pixel_size_xy_um")
        scale = reference / pixel_size if reference else 1.0
    for axis, size, width in zip(["y", "x"], shape[-2:], patch, strict=True):
        if data_init is None:
            half = round(width * scale) // 2
            centers = np.rint(cells[axis])
            inside &= (centers > half) & (centers < size - half)
        else:
            half = round((width // 2) * scale)
            inside &= (cells[axis] >= half) & (cells[axis] <= size - half)
    return inside


def cells_from_labels(m, output, data_init=None):
    """Write independent-instance tables and return the retained canonical cell index.

    ``data_init=None`` selects the paper crop geometry. Otherwise, use the
    selected training recipe's crop geometry for Z=1 images. Instance IDs are
    unique across frames within each FOV and remain separate from biological tracks.
    """
    spec = m["labels"]
    channels = {item["name"]: item["marker"] for item in spec["channels"]}
    if not channels or len(channels) != len(spec["channels"]) or len(set(channels.values())) != len(channels):
        raise ValueError("Each channel needs a unique name and marker")
    metadata = m.get("metadata", {})
    required = {"experiment", "control_group", "replicate", "microscope"}
    if required - metadata.keys():
        raise ValueError(f"Supply label-input metadata: {sorted(required - metadata.keys())}")
    assignments = {}
    for perturbation, wells in spec["perturbation_wells"].items():
        for well in wells:
            if well in assignments:
                raise ValueError(f"Well {well} has more than one perturbation")
            assignments[well] = perturbation
    if not assignments:
        raise ValueError("labels.perturbation_wells must select at least one well")
    label_name = spec.get("label", "nuclei")
    rows = []
    output = Path(output)
    with open_ome_zarr(spec["store"], mode="r") as plate:
        wells = dict(plate.wells())
        if assignments.keys() - wells.keys():
            raise ValueError(f"Unknown wells: {sorted(assignments.keys() - wells.keys())}")
        for well, perturbation in sorted(assignments.items()):
            for fov, pos in wells[well].positions():
                if channels.keys() - set(pos.channel_names):
                    raise ValueError(f"{well}/{fov} lacks channels {sorted(channels.keys() - set(pos.channel_names))}")
                first, interval = _timing(m, pos)
                shape = pos["0"].shape
                if data_init is not None and shape[2] != 1:
                    raise ValueError(
                        "Label-based training inference requires Z=1 images; supply coordinates for Z-stacks"
                    )
                pixel_xy, pixel_z = pos.scale[-1], pos.scale[-3]
                if not np.isfinite([pixel_xy, pixel_z]).all() or min(pixel_xy, pixel_z) <= 0:
                    raise ValueError(f"{well}/{fov} requires positive finite pixel sizes")
                if not np.isclose(pos.scale[-2], pixel_xy):
                    raise ValueError(f"{well}/{fov} requires equal X and Y pixel sizes")
                frames = [props.assign(t=t) for t, props in _label_frames(pos, f"{well}/{fov}", label_name)]
                instances = pd.concat(frames, ignore_index=True).sort_values(["t", "nucleus_label"], ignore_index=True)
                instances["instance_id"] = np.arange(1, len(instances) + 1)
                tracks = instances.assign(
                    id=instances.instance_id, track_id=instances.instance_id, parent_id=-1, parent_track_id=-1
                )
                directory = output / "tracks" / well / fov
                directory.mkdir(parents=True)
                tracks[TRACK_COLUMNS].to_parquet(
                    directory / f"tracks_{well.replace('/', '_')}_{fov}.parquet", index=False
                )
                instances["cell_id"] = f"{well}/{fov}:" + instances.instance_id.astype(str)
                instances["exclusion_reason"] = np.where(
                    _inside_crop(instances, shape, pixel_xy, data_init), None, "crop_crosses_image_border"
                )
                instances["well"], instances["fov"] = well, fov
                instances["perturbation"] = perturbation
                instances["interval_minutes"] = interval
                instances["hours_post_perturbation"] = first + instances.t * interval / 60
                instances["pixel_size_xy_um"], instances["pixel_size_z_um"] = pixel_xy, pixel_z
                for column, size in zip(["T_shape", "C_shape", "Z_shape", "Y_shape", "X_shape"], shape, strict=True):
                    instances[column] = size
                rows.append(instances)
    if not rows or not any(len(frame) for frame in rows):
        raise ValueError(f"No labelled nuclei found in labels/{label_name}")
    nuclei = pd.concat(rows, ignore_index=True)
    nuclei["store_path"] = str(spec["store"])
    nuclei["z"] = 0
    for column, value in metadata.items():
        if column not in nuclei:
            nuclei[column] = value
    for column in ["tracks_path", "track_id", "global_track_id", "lineage_id", "parent_track_id"]:
        # Numeric missing values survive both Parquet and AnnData Zarr round trips.
        nuclei[column] = np.nan
    cells = pd.concat(
        [nuclei.assign(channel_name=channel, marker=marker) for channel, marker in channels.items()], ignore_index=True
    )
    excluded = cells[cells.exclusion_reason.notna()].reset_index(drop=True)
    cells = cells[cells.exclusion_reason.isna()].drop(columns="exclusion_reason").reset_index(drop=True)
    excluded.to_parquet(output / "excluded_cells.parquet", index=False)
    summary = dict(
        track_identity="independent_instances",
        tracks="tracks",
        observations=len(nuclei),
        retained=len(cells),
        excluded=len(excluded),
    )
    (output / "cells_summary.json").write_text(json.dumps(summary, indent=2))
    if cells.empty:
        raise ValueError("Every nucleus was excluded; inspect excluded_cells.parquet")
    return cells
