"""Compare the paper coordinate-inference recipe with archived marker embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from dynaclr.evaluation.organelle_inference import predict_paper_cells
from dynaclr.evaluation.organelle_workflow import validate_cells


def verify(output: Path, per_condition: int = 8):
    from iohub.ngff import open_ome_zarr

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/evaluation"))
    import analyze_direct_pal_aligned_organelle_mmd as paper

    output.mkdir(parents=True, exist_ok=False)
    reports = []
    with pd.option_context("future.infer_string", False):
        for spec in paper.DATASETS:
            source = ad.read_zarr(spec.embedding)
            rng = np.random.default_rng(42)
            obs = source.obs.reset_index(drop=True)
            selected = []
            for condition in ["uninfected", "ZIKV"]:
                available = np.flatnonzero(obs.perturbation.eq(condition) & ~obs.is_blank.astype(bool))
                selected.extend(rng.choice(available, per_condition, replace=False))
            cells = obs.iloc[selected].copy().reset_index(drop=True)
            expected = np.asarray(source.X[selected]).copy()
            cells["cell_id"] = cells.fov_name.astype(str) + ":" + cells.track_id.astype(str) + ":" + cells.t.astype(str)
            cells["store_path"] = source.uns["data_path"]
            cells["tracks_path"] = None
            cells["fov"] = cells.fov_name.str.rsplit("/", n=1).str[-1]
            cells["well"] = cells.fov_name.str.rsplit("/", n=1).str[0]
            cells["channel_name"] = source.uns["channel"]
            cells["z"] = 0
            cells["control_group"] = spec.name
            cells["replicate"] = spec.name
            cells = cells.drop(
                columns=[c for c in ["track_id", "parent_track_id", "parent_id", "lineage_id"] if c in cells]
            )
            for (well, fov), group in cells.groupby(["well", "fov"]):
                with open_ome_zarr(f"{source.uns['data_path']}/{well}/{fov}", mode="r") as pos:
                    for name, size in zip(
                        ["T_shape", "C_shape", "Z_shape", "Y_shape", "X_shape"], pos["0"].shape, strict=True
                    ):
                        cells.loc[group.index, name] = size
                    cells.loc[group.index, "pixel_size_xy_um"] = pos.scale[-1]
                    cells.loc[group.index, "pixel_size_z_um"] = pos.scale[-3]
            destination = output / f"{spec.marker.lower()}__{spec.name}"
            destination.mkdir()
            validate_cells(cells, "fov_statistics")
            cells.to_parquet(destination / "cells.parquet", index=False)
            path = predict_paper_cells(cells, source.uns["checkpoint"], destination, batch_size=8)
            actual = ad.read_zarr(path).X
            delta = actual - expected
            cosine = np.sum(actual * expected, axis=1) / (
                np.linalg.norm(actual, axis=1) * np.linalg.norm(expected, axis=1)
            )
            # Float32 cross-batch/GPU agreement; archive replay is not bitwise.
            match = np.allclose(actual, expected, rtol=1e-4, atol=5e-4) and np.min(cosine) >= 0.999999
            record = dict(
                marker=spec.marker,
                dataset=spec.name,
                cells=len(cells),
                max_absolute_error=float(np.abs(delta).max()),
                rmse=float(np.sqrt(np.mean(delta * delta))),
                minimum_cosine=float(cosine.min()),
                matches=bool(match),
                source=str(spec.embedding),
                checkpoint=str(source.uns["checkpoint"]),
            )
            reports.append(record)
            print(record, flush=True)
            np.save(destination / "archived_embeddings.npy", expected)
            del source
    report = dict(
        comparison="Fresh coordinate-only paper recipe versus archived embeddings",
        rtol=1e-4,
        atol=5e-4,
        minimum_cosine_required=0.999999,
        datasets=reports,
        all_match=all(x["matches"] for x in reports),
    )
    (output / "report.json").write_text(json.dumps(report, indent=2))
    if not report["all_match"]:
        raise ValueError("Archived embedding comparison failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-condition", type=int, default=8)
    args = parser.parse_args()
    verify(args.output, args.per_condition)
