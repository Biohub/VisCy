"""MMD estimation shared by portable analysis and paper-reference verification."""

from __future__ import annotations

import numpy as np


def summarize_mmd_pairs(
    perturbed,
    control,
    *,
    bandwidth,
    permutations=200,
    repeats=50,
    fraction=0.8,
    permutation_seed=42,
    subsample_seed=600042,
):
    """Use the paper's ddof=1 null scale and matched-pair subsampling.

    Input row i in each array is a matched pair. Resampling the same indices
    preserves the dataset/HPI mixture. Bands describe subsampling stability.
    """
    from viscy_utils.evaluation.mmd import compute_mmd_unbiased, mmd_permutation_test

    if len(perturbed) != len(control) or len(control) < 2:
        raise ValueError("MMD requires at least two matched pairs")
    if permutations < 2 or repeats < 2 or not 0 < fraction <= 1:
        raise ValueError("Invalid permutation/subsample settings")
    observed, pvalue, null = mmd_permutation_test(
        perturbed, control, n_permutations=permutations, bandwidth=bandwidth, seed=permutation_seed
    )
    mean, scale = float(np.mean(null)), float(np.std(null, ddof=1))
    if not np.isfinite(scale) or scale <= 1e-12:
        return {"status": "degenerate_null"}
    rng = np.random.default_rng(subsample_seed)
    size = min(len(control), max(4, int(np.floor(fraction * len(control)))))
    samples = []
    for _ in range(repeats):
        selected = rng.choice(len(control), size, replace=False)
        value = compute_mmd_unbiased(perturbed[selected], control[selected], bandwidth=bandwidth)
        samples.append((value - mean) / scale)
    lo, median, hi = np.quantile(samples, [0.025, 0.5, 0.975])
    return dict(
        status="ok",
        mmd2=float(observed),
        pvalue=float(pvalue),
        null_mean=mean,
        null_std=scale,
        null_q95_z=(float(np.quantile(null, 0.95)) - mean) / scale,
        standardized_mmd2=(observed - mean) / scale,
        subsample_q025=float(lo),
        subsample_median=float(median),
        subsample_q975=float(hi),
    )
