import numpy as np
import pandas as pd
import pytest

from dynaclr.evaluation.organelle_workflow import canonical_obs, load_manifest


def test_external_metadata_and_timing():
    frame = pd.DataFrame(
        dict(id=["a", "b"], t=[0, 2], marker=["M"] * 2, experiment=["E"] * 2, perturbation=["mock", "virus"])
    )
    m = dict(
        columns={"id": "cell_id"},
        metadata={"replicate": "r", "control_group": "g"},
        timing={"interval_minutes": 30, "first_frame_hpi": 3},
    )
    result = canonical_obs(frame, m)
    np.testing.assert_equal(result.hours_post_perturbation.to_numpy(), [3, 4])
    assert "track_id" not in result
    assert "infection_state" not in result
    with pytest.raises(ValueError, match="unique"):
        canonical_obs(pd.concat([frame, frame]), m)


def test_missing_metadata_rejected():
    with pytest.raises(ValueError, match="Missing metadata"):
        canonical_obs(pd.DataFrame({"cell_id": ["a"]}), {})


def test_manifest_relocatable(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("schema_version: 1\nroot: data\nembeddings: cells.zarr\n")
    assert load_manifest(manifest)["embeddings"] == str(tmp_path / "data/cells.zarr")


def test_control_normalization_translation_invariance():
    from dynaclr.evaluation.organelle_workflow import control_normalize

    rng = np.random.default_rng(3)
    x = rng.normal(size=(80, 4)).astype("float32")
    obs = pd.DataFrame(
        {
            "experiment": ["a"] * 80,
            "control_group": ["g"] * 80,
            "hours_post_perturbation": np.repeat([3.0, 4.0], 40),
            "perturbation": np.tile(np.repeat(["mock", "virus"], 20), 2),
        }
    )
    result = control_normalize(x, obs, "mock")
    np.testing.assert_allclose(result, control_normalize(x + 10, obs, "mock"), atol=1e-5)
    np.testing.assert_array_equal(x, x.copy())
    with pytest.raises(ValueError, match="two HPI"):
        control_normalize(x, obs.assign(perturbation="virus"), "mock")


def test_classifier_mismatch_rejected_before_loading():
    import anndata as ad

    from dynaclr.evaluation.organelle_workflow import classify

    with pytest.raises(ValueError, match="feature space"):
        classify(
            ad.AnnData(np.zeros((2, 2))),
            {"feature_space": "a", "classifiers": [{"feature_space": "b", "path": "missing.joblib"}]},
        )
