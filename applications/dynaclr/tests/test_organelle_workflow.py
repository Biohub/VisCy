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


def test_paper_mmd_uses_sample_null_scale_and_paired_subsampling(monkeypatch):
    from dynaclr.evaluation.organelle_mmd import summarize_mmd_pairs
    from viscy_utils.evaluation import mmd

    perturbed = np.arange(16, dtype=float).reshape(8, 2)
    control = perturbed + 100
    monkeypatch.setattr(mmd, "mmd_permutation_test", lambda *a, **k: (2.0, 0.1, np.array([-1.0, 0.0, 1.0])))
    calls = []

    def paired_statistic(t, c, **kwargs):
        np.testing.assert_array_equal(c - t, np.full_like(t, 100))
        calls.append(len(t))
        return 2.0

    monkeypatch.setattr(mmd, "compute_mmd_unbiased", paired_statistic)
    result = summarize_mmd_pairs(perturbed, control, bandwidth=1.0, repeats=5)
    assert result["null_std"] == 1.0
    assert result["standardized_mmd2"] == 2.0
    assert result["subsample_median"] == 2.0
    assert calls == [6] * 5


def test_paper_mmd_rejects_unmatched_samples():
    from dynaclr.evaluation.organelle_mmd import summarize_mmd_pairs

    with pytest.raises(ValueError, match="matched pairs"):
        summarize_mmd_pairs(np.zeros((4, 2)), np.zeros((3, 2)), bandwidth=1.0)
