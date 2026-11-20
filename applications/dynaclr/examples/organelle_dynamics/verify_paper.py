"""Recompute paper HPI MMD from frozen matched cohorts using the skill estimator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dynaclr.evaluation import organelle_mmd
from dynaclr.evaluation.organelle_mmd import summarize_mmd_pairs
from dynaclr.evaluation.organelle_workflow import sha256


def plot_comparison(table, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    medians = table.loc[table.metric.eq("subsample_median")]
    groups = list(medians.groupby(["marker", "scope"], sort=True))
    fig, axes = plt.subplots(3, 2, figsize=(10, 9), constrained_layout=True)
    for ax, (key, rows) in zip(axes.flat, groups, strict=True):
        rows = rows.sort_values("hpi")
        ax.plot(rows.hpi, rows.expected, color="0.65", linewidth=3, label="Archived paper")
        ax.plot(rows.hpi, rows.actual, color="#0072B2", linestyle="--", linewidth=1.3, label="Skill recomputation")
        ax.set(title=" / ".join(key), xlabel="HPI", ylabel="Null-standardized MMD²")
    axes.flat[0].legend()
    for extension in ["png", "pdf"]:
        fig.savefig(output / f"mmd_hpi_comparison.{extension}", dpi=180)
    plt.close(fig)


def verify(reference: Path, output: Path):
    output.mkdir(parents=True, exist_ok=False)
    settings = json.loads((reference / "run_info.json").read_text())["arguments"]
    all_curves = pd.read_csv(reference / "mmd_curves.csv")
    samples = pd.read_parquet(reference / "mmd_point_sample_manifest.parquet")
    markers = all_curves.marker.drop_duplicates().tolist()
    scopes = all_curves.scope.drop_duplicates().tolist()
    reference_curves = all_curves.loc[all_curves.axis.eq("hpi")].copy()
    mapping = {"mmd2": "mmd2_unbiased", "pvalue": "permutation_p_value", "standardized_mmd2": "null_standardized_mmd"}
    comparisons = []
    recomputed = []
    for marker in markers:
        print(f"Verifying {marker}", flush=True)
        scores = np.load(reference / f"{marker.lower()}_pca80_scores.npy", mmap_mode="r")
        for row in reference_curves.loc[reference_curves.marker.eq(marker)].itertuples(index=False):
            selected = samples.loc[
                samples.marker.eq(marker)
                & samples.axis.eq("hpi")
                & samples.scope.eq(row.scope)
                & samples.axis_center.eq(row.axis_center)
            ]
            c = selected.loc[selected.sample_group.eq("control")]
            t = selected.loc[~selected.sample_group.eq("control")]
            if len(c) != row.n_control or len(t) != len(c):
                raise ValueError("Reference sample counts do not match the curve")
            if not np.array_equal(c.dataset.to_numpy(), t.dataset.to_numpy()):
                raise ValueError("Reference pairs disagree on dataset")
            if not np.array_equal(c.hours_post_perturbation.to_numpy(), t.hours_post_perturbation.to_numpy()):
                raise ValueError("Reference pairs disagree on HPI")
            for frame in [c, t]:
                if frame.duplicated(["dataset", "fov_name", "lineage_id"]).any():
                    raise ValueError("A lineage appears twice in a paper HPI bin")
            seed = (
                settings["random_seed"]
                + 10_000 * (markers.index(marker) + 1)
                + 100 * scopes.index(row.scope)
                + round((row.axis_center - settings["hpi_min"]) / settings["hpi_bin_hours"])
            )
            result = summarize_mmd_pairs(
                scores[t.pc_row.to_numpy()],
                scores[c.pc_row.to_numpy()],
                bandwidth=row.bandwidth,
                permutations=settings["null_permutations"],
                repeats=settings["subsample_repeats"],
                fraction=settings["subsample_fraction"],
                permutation_seed=seed + 300_000,
                subsample_seed=seed + 600_000,
            )
            if result.pop("status") != "ok":
                raise ValueError("Reference bin has a degenerate recomputed null")
            result = {mapping.get(key, key): value for key, value in result.items()}
            recomputed.append(dict(marker=marker, scope=row.scope, axis_center=row.axis_center, **result))
            for key, value in result.items():
                expected = getattr(row, key)
                comparisons.append(
                    dict(
                        marker=marker,
                        scope=row.scope,
                        hpi=row.axis_center,
                        metric=key,
                        expected=expected,
                        actual=value,
                        absolute_error=abs(value - expected),
                        matches=bool(np.isclose(value, expected, rtol=1e-5, atol=1e-6)),
                    )
                )
    table = pd.DataFrame(comparisons)
    table.to_csv(output / "comparisons.csv", index=False)
    pd.DataFrame(recomputed).to_csv(output / "mmd_curves.csv", index=False)
    plot_comparison(table, output)
    report = dict(
        comparison="Skill estimator versus archived HPI curves on identical frozen matched pairs",
        rows=len(recomputed),
        metric_checks=len(table),
        all_match=bool(table.matches.all()),
        maximum_absolute_error=float(table.absolute_error.max()),
        rtol=1e-5,
        atol=1e-6,
        cohort_checks=["counts", "paired dataset", "paired exact HPI", "unique lineage per bin"],
        inputs={
            name: sha256(reference / name)
            for name in [
                "mmd_curves.csv",
                "mmd_point_sample_manifest.parquet",
                "run_info.json",
                *[f"{marker.lower()}_pca80_scores.npy" for marker in markers],
            ]
        },
        estimator_sha256=sha256(Path(organelle_mmd.__file__)),
    )
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if not report["all_match"]:
        raise ValueError("Paper comparison failed; inspect comparisons.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    verify(args.reference, args.output)
