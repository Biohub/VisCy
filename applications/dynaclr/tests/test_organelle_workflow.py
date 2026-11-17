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


@pytest.fixture
def normalization_data():
    import anndata as ad

    # Identical biological variation, with distinct acquisition/marker offsets/scales.
    base = np.array([[-2, -4], [2, 4], [8, 12], [-2, -4], [2, 4], [8, 12]], dtype=np.float32)
    arrays, frames = [], []
    for marker, experiment, group, offset, scale in [
        ("M1", "E1", "g1", 0, 1),
        ("M1", "E2", "g1", 100, 4),
        ("M2", "E1", "g1", -20, 2),
        ("M1", "E1", "g2", 50, 8),
    ]:
        arrays.append(base * scale + offset)
        frames.append(
            pd.DataFrame(
                dict(
                    cell_id=[f"{marker}-{experiment}-{group}-{i}" for i in range(6)],
                    marker=marker,
                    experiment=experiment,
                    control_group=group,
                    replicate="r",
                    perturbation=["mock", "mock", "virus"] * 2,
                    hours_post_perturbation=[3.0] * 3 + [4.0] * 3,
                )
            )
        )
    with pd.option_context("future.infer_string", False):
        return ad.AnnData(np.concatenate(arrays), obs=pd.concat(frames, ignore_index=True))


def test_normalization_uses_own_controls_and_preserves_raw(normalization_data):
    from dynaclr.evaluation.organelle_normalization import normalize_embeddings, require_normalized

    data = normalization_data
    m = dict(control_label="mock", feature_space="encoder")
    raw = data.X.copy()
    normalize_embeddings(data, m)
    actual = require_normalized(data, m).copy()
    # The positive-MAD 1% floor raises the smaller scale from 2 to 2.02.
    expected = np.tile(raw[:6] / np.array([2.02, 4]) / 1.4826, (4, 1))
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    np.testing.assert_array_equal(data.X, raw)
    normalize_embeddings(data, m)
    np.testing.assert_array_equal(require_normalized(data, m), actual)
    # Treated cells cannot alter control parameters.
    data.X[data.obs.perturbation.eq("virus")] += 10000
    normalize_embeddings(data, m)
    controls = data.obs.perturbation.eq("mock").to_numpy()
    np.testing.assert_array_equal(require_normalized(data, m)[controls], actual[controls])


@pytest.mark.parametrize("change", ["raw", "metadata", "normalized", "label", "cohort"])
def test_stale_normalization_rejected(normalization_data, change):
    from dynaclr.evaluation.organelle_normalization import normalize_embeddings, require_normalized

    data = normalization_data
    m = dict(control_label="mock", feature_space="encoder")
    normalize_embeddings(data, m)
    if change == "raw":
        data.X[0, 0] += 1
    elif change == "metadata":
        data.obs.loc[data.obs.index[0], "control_group"] = "other"
    elif change == "normalized":
        data.obsm["X_control_mad"][0, 0] += 1
    elif change == "label":
        m["control_label"] = "virus"
    else:
        data = data[:6].copy()
    with pytest.raises(ValueError, match="stale"):
        require_normalized(data, m)


def test_normalize_cli_roundtrip(normalization_data, tmp_path):
    import json

    import anndata as ad
    import yaml

    from dynaclr.evaluation.organelle_normalization import require_normalized
    from dynaclr.evaluation.organelle_workflow import run

    data = normalization_data
    data.write_zarr(tmp_path / "raw.zarr")
    m = dict(schema_version=1, embeddings="raw.zarr", feature_space="encoder", control_label="mock")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump(m))
    run(manifest, tmp_path / "normalized", "normalize")
    saved = ad.read_zarr(tmp_path / "normalized/normalized.zarr")
    normalized = require_normalized(saved, m)
    np.testing.assert_array_equal(saved.X, data.X)
    table = pd.read_parquet(tmp_path / "normalized/embeddings.parquet")
    np.testing.assert_array_equal(np.stack(table.embeddings), data.X)
    np.testing.assert_array_equal(np.stack(table["ctrl-norm-embedding"]), normalized)
    assert table.observation_id.tolist() == saved.obs_names.tolist()
    contract = json.loads((tmp_path / "normalized/normalization.json").read_text())
    assert len(contract["groups"]) == 4
    assert all(group["n_control"] == 4 for group in contract["groups"].values())
    # Running the explicit step again recomputes from raw X, never from normalized vectors.
    m["embeddings"] = "normalized/normalized.zarr"
    manifest.write_text(yaml.safe_dump(m))
    run(manifest, tmp_path / "again", "normalize")
    again = ad.read_zarr(tmp_path / "again/normalized.zarr")
    np.testing.assert_array_equal(require_normalized(again, m), normalized)


def test_plot_requires_normalization(normalization_data, tmp_path):
    import yaml

    from dynaclr.evaluation.organelle_workflow import run

    normalization_data.write_zarr(tmp_path / "raw.zarr")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            dict(
                schema_version=1,
                embeddings="raw.zarr",
                feature_space="encoder",
                control_label="mock",
            )
        )
    )
    with pytest.raises(ValueError, match="normalize stage"):
        run(manifest, tmp_path / "plot", "plot")
    assert not (tmp_path / "plot/provenance.json").exists()


def test_classifier_keeps_raw_feature_space_after_normalization(normalization_data, tmp_path, monkeypatch):
    import joblib

    from dynaclr.evaluation.organelle_normalization import normalize_embeddings
    from dynaclr.evaluation.organelle_workflow import classify, sha256

    data = normalization_data
    raw = data.X[data.obs.marker.eq("M1")].copy()
    m = dict(control_label="mock", feature_space="encoder")
    normalize_embeddings(data, m)

    class Classifier:
        classes_ = np.array(["uninfected", "infected"])

        def predict(self, x):
            np.testing.assert_array_equal(x, raw)
            return np.repeat("uninfected", len(x))

        def predict_proba(self, x):
            np.testing.assert_array_equal(x, raw)
            return np.tile([0.75, 0.25], (len(x), 1))

    artifact = tmp_path / "model"
    artifact.write_text("pinned test artifact")
    monkeypatch.setattr(joblib, "load", lambda path: Classifier())
    m["classifiers"] = [
        dict(
            path=str(artifact),
            sha256=sha256(artifact),
            feature_space="encoder",
            marker="M1",
            task="infection_state",
        )
    ]
    classify(data, m)
    assert data.obs.loc[data.obs.marker.eq("M1"), "predicted_infection_state"].eq("uninfected").all()
