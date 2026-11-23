---
name: organelle-pseudotracks
description: Write per-FOV pseudo-tracking Parquet files (the tracking-CSV columns) from 2D nuclear labels for an OME-Zarr plate that has no tracking.
---

Read the shared [contract](../../examples/organelle_dynamics/CONTRACT.md). Run from the VisCy repository with its locked uv environment:

```sh
uv run --no-sync python -m dynaclr.evaluation.organelle_workflow tracks-from-labels --manifest tracks.yaml --output new-tracks-directory
```

The manifest needs only the label store:

```yaml
schema_version: 1
root: .
labels:
  store: plate.ome.zarr
  label: nuclei
```

The stage reads `labels/<label>` at every position and computes each nucleus centroid per frame from the labels. It never reads or converts existing tracking CSVs. It writes `tracks/<row>/<col>/<fov>/tracks_<row>_<col>_<fov>.parquet` with the tracking-CSV columns `track_id, t, y, x, id, parent_track_id, parent_id`, plus `tracks_summary.json` and provenance.

As in the classical-sampling CSVs, `track_id` is the nucleus label value. Label values are per-frame instance IDs that restart each frame, so the same `track_id` at different frames is not the same cell. `id` is `100000` plus the row number from 1 in (t, label) order, unique within the FOV, and both parent fields are -1. Do not use these files for track or lineage sampling or time alignment; they carry no temporal identity. Labels must be 2D (Z=1) and match the image T, Y and X shape. Report the FOV and observation counts from `tracks_summary.json`. The output directory must not already exist.
