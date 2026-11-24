"""Check explicit normalization/plotting against an existing paper-data analysis."""

import argparse
import json
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import yaml

from dynaclr.evaluation.organelle_normalization import require_normalized
from dynaclr.evaluation.organelle_workflow import load_manifest, run


def verify(reference, output):
    output.mkdir(parents=True, exist_ok=False)
    m = load_manifest(reference / "resolved_manifest.yaml")
    manifest = output / "normalize.yaml"
    manifest.write_text(yaml.safe_dump(m))
    run(manifest, output / "normalized", "normalize")
    normalized = ad.read_zarr(output / "normalized/normalized.zarr")
    raw = ad.read_zarr(m["embeddings"])
    np.testing.assert_array_equal(normalized.X, raw.X)
    expected = require_normalized(normalized, m)
    table = pd.read_parquet(output / "normalized/embeddings.parquet")
    np.testing.assert_array_equal(np.stack(table.embeddings), raw.X)
    np.testing.assert_array_equal(np.stack(table["ctrl-norm-embedding"]), expected)
    assert table.observation_id.tolist() == normalized.obs_names.tolist()
    m["embeddings"] = str(output / "normalized/normalized.zarr")
    m.pop("classifiers", None)
    manifest = output / "analysis.yaml"
    manifest.write_text(yaml.safe_dump(m))
    run(manifest, output / "plots", "plot")
    run(manifest, output / "analysis", "analyze")
    assert not (output / "plots/mmd_curves.csv").exists()
    for path in (output / "plots").glob("*_coordinates.parquet"):
        pd.testing.assert_frame_equal(pd.read_parquet(path), pd.read_parquet(output / "analysis" / path.name))
    saved = ad.read_zarr(output / "analysis/predictions.zarr")
    np.testing.assert_array_equal(require_normalized(saved, m), expected)
    pd.testing.assert_frame_equal(
        pd.read_csv(output / "analysis/mmd_curves.csv"),
        pd.read_csv(reference / "mmd_curves.csv"),
        check_exact=True,
    )
    original = pd.read_parquet(reference / "predictions.parquet")
    prediction_columns = [column for column in original if column.startswith("predicted_")]
    pd.testing.assert_frame_equal(
        original[prediction_columns],
        pd.read_parquet(output / "normalized/predictions.parquet")[prediction_columns],
        check_dtype=False,
        check_exact=True,
    )
    report = dict(
        observations=len(raw),
        dimensions=raw.n_vars,
        raw_vectors_exact=True,
        normalized_parquet_zarr_exact=True,
        saved_normalization_reused_exactly=True,
        plot_analysis_coordinates_exact=True,
        classifier_predictions_exact=True,
        previous_mmd_curves_exact=True,
        plot_png_count=len(list((output / "plots").glob("*.png"))),
        normalization_groups=len(normalized.uns["organelle_normalization"]["groups"]),
        reference=str(reference),
        output=str(output),
    )
    (output / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.reference.resolve(), args.output.resolve())
