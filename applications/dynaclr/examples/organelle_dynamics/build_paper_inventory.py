"""Inventory local paper assets without copying or modifying them."""

import argparse
import json
from pathlib import Path

from dynaclr.evaluation.organelle_workflow import sha256


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prediction-root", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    root = args.prediction_root
    artifacts = []
    for pattern, role in [
        ("embeddings/*.zarr", "cached embeddings"),
        ("classifier/DynaCLR/**/pipelines/*.joblib", "classifier"),
        ("classifier/DynaCLR/**/pipelines/manifest.json", "classifier metadata"),
        ("classifier_exploration/outputs/**/run_info.json", "analysis provenance"),
        ("classifier_exploration/outputs/**/predictions_by_threshold.parquet", "teacher labels"),
    ]:
        for path in sorted(root.glob(pattern)):
            artifacts.append(
                dict(
                    role=role,
                    path=str(path),
                    available=path.exists(),
                    sha256=sha256(path) if path.is_file() else None,
                    public_url=None,
                )
            )
    model_root = Path("/hpc/projects/organelle_phenotyping/models/DynaCLR-2D-MIP-BagOfChannels") / root.parent.name
    for path in [
        model_root / "config.yaml",
        model_root / "DynaCLR-2D-MIP-BagOfChannels/jbrwhzr3/checkpoints/epoch=105-step=84800.ckpt",
    ]:
        artifacts.append(
            dict(
                role="encoder configuration/weights",
                path=str(path),
                available=path.exists(),
                sha256=sha256(path) if path.is_file() else None,
                public_url=None,
            )
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as f:
        json.dump(dict(schema_version=1, repo="https://github.com/Biohub/VisCy", artifacts=artifacts), f, indent=2)


if __name__ == "__main__":
    main()
