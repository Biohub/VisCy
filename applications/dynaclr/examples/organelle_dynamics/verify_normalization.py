import argparse
import gc
import json
import sys
from pathlib import Path

import anndata as ad
import joblib
import numpy as np

from dynaclr.evaluation.organelle_workflow import control_normalize


def verify(root: Path, out: Path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/evaluation"))
    import analyze_direct_pal_aligned_organelle_mmd as paper

    out.mkdir(parents=True, exist_ok=False)
    args = argparse.Namespace(**json.loads((root / "run_info.json").read_text())["arguments"])
    teacher = paper._load_teacher(Path(args.teacher))
    records = []
    pca_records = []
    for marker in paper.MARKER_ORDER:
        arrays = []
        for spec in paper._marker_specs(marker):
            print(marker, spec.name, flush=True)
            expected, obs, contract, lineage = paper._load_marker_dataset(spec, teacher, args)
            data = ad.read_zarr(spec.embedding)
            lookup = data.obs.reset_index(drop=True).reset_index(names="source_row")
            rows = (
                obs[paper.KEYS]
                .merge(lookup[paper.KEYS + ["source_row"]], on=paper.KEYS, validate="one_to_one")
                .source_row.to_numpy()
            )
            canonical = obs.assign(experiment=spec.name, control_group=spec.name).reset_index(drop=True)
            actual = control_normalize(np.asarray(data.X)[rows], canonical, "uninfected")
            delta = np.abs(expected - actual)
            record = dict(
                marker=marker,
                dataset=spec.name,
                cells=len(actual),
                max_absolute_error=float(delta.max()),
                matches=bool(np.allclose(actual, expected, rtol=1e-5, atol=1e-6)),
            )
            records.append(record)
            arrays.append(actual)
            print(record, flush=True)
            del data, expected, delta
            gc.collect()
        combined = np.concatenate(arrays)
        model = joblib.load(root / "models" / f"{marker.lower()}_pca80.joblib")
        actual = model.transform(combined).astype(np.float32)
        expected = np.load(root / f"{marker.lower()}_pca80_scores.npy")
        delta = np.abs(expected - actual)
        pca_records.append(
            dict(
                marker=marker,
                cells=len(actual),
                components=actual.shape[1],
                max_absolute_error=float(delta.max()),
                matches=bool(np.allclose(actual, expected, rtol=1e-5, atol=1e-5)),
            )
        )
        del arrays, combined, actual, expected, delta
        gc.collect()
    report = {
        "normalization": records,
        "pca_scores": pca_records,
        "all_match": all(x["matches"] for x in records + pca_records),
    }
    (out / "normalization_comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)

    if not report["all_match"]:
        raise ValueError("Normalization/PCA comparison failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare skill normalization and frozen PCA against the paper across all matched observations."
    )
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    verify(args.reference, args.output)
