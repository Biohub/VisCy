import json

import numpy as np
import pandas as pd
import pytest
import yaml
from iohub import open_ome_zarr
from iohub.ngff.models import TransformationMeta

from dynaclr.evaluation.organelle_cells import paper_crop_half_px
from dynaclr.evaluation.organelle_workflow import run

# Native pixel size giving a 20-pixel paper crop (half = 10).
PIXEL = 0.1494 * 160 / 20
STATS = dict(mean=1.0, std=2.0, median=1.0, iqr=1.0, min=0.0, max=9.0)


def _label_store(path, interval_s=1800.0):
    with open_ome_zarr(path, layout="hcs", mode="w", channel_names=["Phase3D", "GFP"]) as plate:
        for row, col in [("A", "1"), ("A", "2")]:
            pos = plate.create_position(row, col, "0")
            scale = TransformationMeta(type="scale", scale=[interval_s, 1.0, 1.0, PIXEL, PIXEL])
            pos.create_image("0", np.ones((2, 2, 1, 64, 64), np.float32), transform=[scale])
            labels = np.zeros((2, 1, 1, 64, 64), np.uint32)
            labels[:, 0, 0, 30:35, 30:35] = 7  # centred: kept
            labels[0, 0, 0, 2:5, 2:5] = 8  # crop crosses the border at t=0 only
            pos.zgroup.create_group("labels").create_group("nuclei").create_array("0", data=labels, chunks=labels.shape)
            pos.zattrs["normalization"] = {"GFP": {"fov_statistics": STATS}}


def _manifest(tmp_path, **overrides):
    m = dict(
        schema_version=1,
        root=".",
        feature_space="encoder",
        control_label="mock",
        labels=dict(
            store="plate.zarr",
            channels=[dict(name="GFP", marker="M")],
            perturbation_wells={"mock": ["A/1"], "virus": ["A/2"]},
        ),
        metadata=dict(experiment="E", control_group="g", replicate="r", microscope="scope"),
        timing=dict(first_frame_hpi=6, interval_minutes=30),
    )
    m.update(overrides)
    path = tmp_path / "cells.yaml"
    path.write_text(yaml.safe_dump(m))
    return path


def test_crop_half_matches_paper_recipe():
    assert paper_crop_half_px(PIXEL) == 10
    assert paper_crop_half_px(0.10708735394584887) == 111


def test_cells_from_labels_untracked_index(tmp_path):
    _label_store(tmp_path / "plate.zarr")
    run(_manifest(tmp_path), tmp_path / "out", "cells-from-labels")
    cells = pd.read_parquet(tmp_path / "out/cells.parquet")
    excluded = pd.read_parquet(tmp_path / "out/excluded_cells.parquet")
    assert len(cells) == 4  # one centred nucleus x 2 wells x 2 frames
    np.testing.assert_allclose(cells[["y", "x"]].to_numpy(), 32.0)
    assert sorted(cells.hours_post_perturbation.unique()) == [6.0, 6.5]
    assert set(cells.perturbation) == {"mock", "virus"}
    assert cells.cell_id.is_unique and cells.cell_id.str.contains(":7:").all()
    assert set(cells.channel_name) == {"GFP"} and set(cells.marker) == {"M"}
    # Nucleus labels are per-frame instances, never track identities.
    assert "track_id" not in cells and cells.tracks_path.isna().all()
    assert (cells[["T_shape", "C_shape", "Z_shape", "Y_shape", "X_shape"]].to_numpy() == [2, 2, 1, 64, 64]).all()
    assert len(excluded) == 2 and set(excluded.exclusion_reason) == {"crop_crosses_image_border"}
    summary = json.loads((tmp_path / "out/cells_summary.json").read_text())
    assert summary["cells"] == 4


@pytest.mark.parametrize(
    "overrides, message",
    [
        (dict(timing=None), "start HPI"),
        (dict(timing=dict(interval_minutes=30)), "start HPI"),
        (dict(timing=dict(first_frame_hpi=6, interval_minutes=60)), "disagrees with the image time scale"),
        (dict(control_label="uninfected"), "control_label"),
        (dict(metadata=dict(experiment="E", control_group="g", replicate="r")), "microscope"),
    ],
)
def test_cells_from_labels_rejects_unsupplied_metadata(tmp_path, overrides, message):
    _label_store(tmp_path / "plate.zarr")
    with pytest.raises(ValueError, match=message):
        run(_manifest(tmp_path, **overrides), tmp_path / "out", "cells-from-labels")


def test_missing_channel_normalization_names_channel(tmp_path):
    _label_store(tmp_path / "plate.zarr")
    with open_ome_zarr(tmp_path / "plate.zarr/A/1/0", mode="r+") as pos:
        pos.zattrs["normalization"] = {"4-GFP-raw": {"fov_statistics": STATS}}
    with pytest.raises(ValueError, match="'GFP'.*4-GFP-raw"):
        run(_manifest(tmp_path), tmp_path / "out", "cells-from-labels")


def test_tracks_from_labels_per_fov_parquet(tmp_path):
    _label_store(tmp_path / "plate.zarr")
    manifest = tmp_path / "tracks.yaml"
    manifest.write_text(yaml.safe_dump(dict(schema_version=1, root=".", labels=dict(store="plate.zarr"))))
    run(manifest, tmp_path / "out", "tracks-from-labels")
    for fov in ["A/1/0", "A/2/0"]:
        tracks = pd.read_parquet(tmp_path / "out/tracks" / fov / f"tracks_{fov.replace('/', '_')}.parquet")
        assert list(tracks.columns) == ["track_id", "t", "y", "x", "id", "parent_track_id", "parent_id"]
        # t=0: labels 7 and 8; t=1: label 7. track_id is the label value; id is unique.
        assert tracks.t.tolist() == [0, 0, 1]
        assert tracks.track_id.tolist() == [7, 8, 7]
        assert tracks.id.tolist() == [100001, 100002, 100003]
        assert (tracks[["parent_track_id", "parent_id"]] == -1).all().all()
        np.testing.assert_allclose(tracks[["y", "x"]].to_numpy(), [[32, 32], [3, 3], [32, 32]])
    summary = json.loads((tmp_path / "out/tracks_summary.json").read_text())
    assert summary["fovs"] == 2 and summary["observations"] == 6
