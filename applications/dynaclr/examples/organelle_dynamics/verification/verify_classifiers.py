"""Compare infection predictions on fresh and archived paper embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import anndata as ad
import joblib
import numpy as np

from dynaclr.evaluation.organelle_workflow import sha256


def verify(inference_results: Path, prediction_root: Path, output: Path):
    report = json.loads((inference_results / "report.json").read_text())
    suffixes = {"SEC61B": "", "TOMM20": "__tomm20plate", "G3BP1": "__g3bp1plate"}
    comparisons = []
    for row in report["datasets"]:
        marker = row["marker"]
        path = (
            prediction_root
            / "classifier/DynaCLR"
            / f"infection_state__pAL17_to_{marker}{suffixes[marker]}"
            / "pipelines"
            / f"infection_state_{marker}.joblib"
        )
        model = joblib.load(path)
        folder = inference_results / f"{marker.lower()}__{row['dataset']}"
        fresh = ad.read_zarr(folder / "embeddings/embeddings.zarr").X
        archived = np.load(folder / "archived_embeddings.npy")
        labels_match = bool(np.array_equal(model.predict(fresh), model.predict(archived)))
        p1, p2 = model.predict_proba(fresh), model.predict_proba(archived)
        comparisons.append(
            dict(
                marker=marker,
                dataset=row["dataset"],
                cells=len(fresh),
                labels_match=labels_match,
                maximum_probability_difference=float(np.abs(p1 - p2).max()),
                probabilities_match=bool(np.allclose(p1, p2, atol=5e-4, rtol=1e-4)),
                classifier=str(path),
                classifier_sha256=sha256(path),
            )
        )
    result = dict(
        comparison="Same saved infection classifiers on fresh versus archived embeddings",
        tolerance_basis="Same absolute float32 replay budget as the embedding check; labels must agree exactly",
        rtol=1e-4,
        atol=5e-4,
        datasets=comparisons,
        all_match=all(x["labels_match"] and x["probabilities_match"] for x in comparisons),
    )
    with output.open("x") as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))
    if not result["all_match"]:
        raise ValueError("Classifier comparison failed")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--inference-results", type=Path, required=True)
    p.add_argument("--prediction-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    verify(a.inference_results, a.prediction_root, a.output)
