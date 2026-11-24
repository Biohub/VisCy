import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import yaml
from iohub import open_ome_zarr
from iohub.ngff.models import TransformationMeta

from dynaclr.evaluation.organelle_cells import _inside_crop, cells_from_labels
from dynaclr.evaluation.organelle_workflow import load_manifest, main, run, sha256

PIXEL = 0.1494 * 160 / 20
STATS = dict(mean=1.0, std=2.0, median=1.0, iqr=1.0, min=0.0, max=9.0)


def _label_store(path, frames=2):
    with open_ome_zarr(path, layout="hcs", mode="w", channel_names=["Phase3D", "GFP"]) as plate:
        for row, col in [("A", "1"), ("A", "2")]:
            pos = plate.create_position(row, col, "0")
            scale = TransformationMeta(type="scale", scale=[1800.0, 1.0, 1.0, PIXEL, PIXEL])
            pos.create_image("0", np.ones((frames, 2, 1, 64, 64), np.float32), transform=[scale])
            labels = np.zeros((frames, 1, 1, 64, 64), np.uint32)
            labels[:, 0, 0, 30:35, 30:35] = 7
            labels[0, 0, 0, 2:5, 2:5] = 8
            pos.zgroup.create_group("labels").create_group("nuclei").create_array("0", data=labels, chunks=labels.shape)
            pos.zattrs["normalization"] = {
                "GFP": {"fov_statistics": STATS, "timepoint_statistics": {str(t): STATS for t in range(frames)}}
            }


def _manifest(tmp_path, **overrides):
    checkpoint = tmp_path / "encoder.ckpt"
    checkpoint.write_text("encoder replaced by a deterministic test double")
    m = dict(
        schema_version=1,
        root=".",
        feature_space="encoder",
        checkpoint="encoder.ckpt",
        checkpoint_sha256=sha256(checkpoint),
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
    path = tmp_path / "my experiment.yaml"
    path.write_text(yaml.safe_dump(m))
    return path


def _fake_embeddings(cells, destination):
    keys = cells[["experiment", "cell_id", "marker"]].copy()
    with pd.option_context("future.infer_string", False):
        keys = keys.astype(object)
        data = ad.AnnData(np.arange(len(cells) * 4, dtype="float32").reshape(len(cells), 4), obs=keys)
        data.write_zarr(destination)
    return destination


@pytest.mark.parametrize("frames", [1, 2])
@pytest.mark.parametrize("recipe", ["paper", "cell_dino"])
def test_embed_labels_preserves_independent_instances(tmp_path, monkeypatch, frames, recipe):
    _label_store(tmp_path / "plate.zarr", frames)
    output = tmp_path / "output"
    overrides = dict(timing={"first_frame_hpi": 6}) if frames == 1 else {}
    if recipe == "paper":

        def predict(cells, checkpoint, destination, batch_size):
            return _fake_embeddings(cells, destination / "raw.zarr")

        monkeypatch.setattr("dynaclr.evaluation.organelle_inference.predict_paper_cells", predict)
    else:
        config = {
            "model": {
                "class_path": "dynaclr.engine.ContrastiveModule",
                "init_args": {
                    "encoder": {
                        "class_path": "viscy_models.foundation.cell_dino.CellDinoModel",
                        "init_args": {"weights_path": "unused.pth"},
                    }
                },
            },
            "data": {
                "class_path": "dynaclr.data.datamodule.MultiExperimentDataModule",
                "init_args": {"yx_patch_size": [20, 20], "z_window": 1},
            },
        }
        (tmp_path / "recipe.yml").write_text(yaml.safe_dump(config))
        overrides.update(model_family="cell_dino", training_config="recipe.yml")

        def predict(command, check):
            if command[1] != "predict":
                pytest.fail(f"Unexpected subprocess: {command}")
            resolved = yaml.safe_load(Path(command[-1]).read_text())
            assert resolved["trainer"]["default_root_dir"] == str(output)
            from dynaclr.data.datamodule import MultiExperimentDataModule

            loader = MultiExperimentDataModule(**resolved["data"]["init_args"])
            loader.setup("predict")
            cells = pd.read_parquet(resolved["data"]["init_args"]["cell_index_path"])
            assert len(loader.predict_dataset) == len(cells)
            destination = output / "embeddings/embeddings.zarr"
            destination.parent.mkdir()
            _fake_embeddings(cells, destination)

        monkeypatch.setattr(
            "dynaclr.evaluation.organelle_workflow.subprocess",
            SimpleNamespace(run=predict, check_output=subprocess.check_output),
        )
    config = _manifest(tmp_path, **overrides)
    monkeypatch.setattr("sys.argv", ["dynacell", "embed", "--config", str(config), "--output", str(output)])
    main()
    cells = pd.read_parquet(output / "cells.parquet")
    saved = ad.read_zarr(output / "predictions.zarr")
    assert len(cells) == len(saved) == 2 * frames
    assert cells.cell_id.is_unique
    assert set(cells.nucleus_label) == {7}
    assert cells[["track_id", "lineage_id", "global_track_id", "tracks_path"]].isna().all().all()
    assert saved.obs[["track_id", "lineage_id", "global_track_id", "tracks_path"]].isna().all().all()
    assert saved.obs.cell_id.tolist() == cells.cell_id.tolist()
    np.testing.assert_allclose(cells.interval_minutes, 0 if frames == 1 else 30)
    np.testing.assert_allclose(cells[["y", "x"]], 32.0)
    np.testing.assert_allclose(sorted(cells.hours_post_perturbation.unique()), [6.0] if frames == 1 else [6.0, 6.5])
    for fov in ["A/1/0", "A/2/0"]:
        table = pd.read_parquet(output / "tracks" / fov / f"tracks_{fov.replace('/', '_')}.parquet")
        assert table.track_id.is_unique and table.id.equals(table.track_id)
        assert table.id.tolist() == list(range(1, frames + 2))
        assert table.t.tolist() == ([0, 0] if frames == 1 else [0, 0, 1])
        assert table[["parent_id", "parent_track_id"]].eq(-1).all().all()
    assert len(pd.read_parquet(output / "excluded_cells.parquet")) == 2
    provenance = json.loads((output / "provenance.json").read_text())
    assert provenance["stage"] == "embed"
    assert provenance["manifest"]["input_preparation"]["track_identity"] == "independent_instances"
    # A resolved label-input configuration must remain valid for another run.
    assert "cell_index" not in load_manifest(output / "resolved_manifest.yaml")
    assert "X_control_mad" not in saved.obsm


@pytest.mark.parametrize(
    "timing, message",
    [
        ({}, "first_frame_hpi"),
        ({"first_frame_hpi": float("nan"), "interval_minutes": 30}, "first_frame_hpi"),
        ({"first_frame_hpi": 6}, "interval_minutes"),
        ({"first_frame_hpi": 6, "interval_minutes": -30}, "positive"),
        ({"first_frame_hpi": 6, "interval_minutes": 60}, "image time scale"),
    ],
)
def test_labels_reject_invalid_timing(tmp_path, timing, message):
    _label_store(tmp_path / "plate.zarr")
    with pytest.raises(ValueError, match=message):
        run(_manifest(tmp_path, timing=timing), tmp_path / "out", "embed")
    assert not (tmp_path / "out/provenance.json").exists()


def test_missing_channel_statistics_names_source(tmp_path):
    _label_store(tmp_path / "plate.zarr")
    with open_ome_zarr(tmp_path / "plate.zarr/A/1/0", mode="r+") as pos:
        pos.zattrs["normalization"] = {"4-GFP-raw": {"fov_statistics": STATS}}
    with pytest.raises(ValueError, match="'GFP'.*4-GFP-raw"):
        run(_manifest(tmp_path), tmp_path / "out", "embed")


def test_paper_border_filter_uses_rounded_centers():
    cells = pd.DataFrame({"y": [10.49, 10.51, 53.49, 53.51], "x": [32.0] * 4})
    np.testing.assert_array_equal(_inside_crop(cells, (1, 1, 1, 64, 64), PIXEL, None), [False, True, True, False])


def test_all_excluded_still_writes_exclusion_table(tmp_path):
    _label_store(tmp_path / "plate.zarr")
    output = tmp_path / "out"
    output.mkdir()
    m = load_manifest(_manifest(tmp_path))
    with pytest.raises(ValueError, match="Every nucleus"):
        cells_from_labels(m, output, data_init={"yx_patch_size": [128, 128]})
    assert len(pd.read_parquet(output / "excluded_cells.parquet")) == 6


def test_ambiguous_input_rejected_before_output(tmp_path):
    path = _manifest(tmp_path, cell_index="also-cells.parquet")
    with pytest.raises(ValueError, match="exactly one"):
        run(path, tmp_path / "out", "embed")
    assert not (tmp_path / "out").exists()


def test_inference_does_not_require_control_cells(tmp_path):
    _label_store(tmp_path / "plate.zarr", frames=1)
    m = load_manifest(_manifest(tmp_path, timing={"first_frame_hpi": 24}))
    m["labels"]["perturbation_wells"] = {"virus": ["A/2"]}
    output = tmp_path / "out"
    output.mkdir()
    cells = cells_from_labels(m, output)
    assert cells.perturbation.tolist() == ["virus"]


def test_malformed_labels_fail_explicitly(tmp_path):
    _label_store(tmp_path / "plate.zarr")
    with open_ome_zarr(tmp_path / "plate.zarr/A/1/0", mode="r+") as pos:
        labels = pos.zgroup["labels"]["nuclei"]
        del labels["0"]
        labels.create_array("0", data=np.zeros((2, 2, 1, 64, 64), dtype="uint32"))
    with pytest.raises(ValueError, match="shape"):
        run(_manifest(tmp_path), tmp_path / "out", "embed")
