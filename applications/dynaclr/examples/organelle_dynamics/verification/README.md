# Verification

Return to the [workflow entry point](../README.md) to run the skills.
These scripts compare workflow components with archived paper results.
They do not reproduce complete Figure 6 panels.

## Files and recorded results

| File | Purpose and recorded result |
| --- | --- |
| [verify_inference.py](verify_inference.py) | Compare fresh DynaCLR embeddings with archived values. All 96 observations pass the float32 tolerance. |
| [verify_classifiers.py](verify_classifiers.py) | Compare saved classifiers on fresh and archived embeddings. All 96 labels match. |
| [verify_normalization.py](verify_normalization.py) | Recompute control normalization and saved PCA transforms. The archived run matched 257,856 observations using the former smoothed recipe. |
| [verify_paper.py](verify_paper.py) | Recompute archived ZIKV HPI MMD. All 3,339 comparisons pass across 371 bins. |
| [verify_steps.py](verify_steps.py) | Check exports and separate plotting on 192 observations. Previous MMD tables match exactly. |
| [results.json](results.json) | Combined original numerical reports, tolerances, input hashes and integration-check paths. |

The integration checks also cover CELL-DINO inference on two cells and `phenotype` on 192 paper observations.
They check finite vectors, preserved identities and export agreement.
They do not establish CELL-DINO biological accuracy.

The current default uses independent per-timepoint normalization without smoothing.
Historical normalization and downstream PCA/MMD equality results do not validate this changed default.
`verify_normalization.py` reports differences against the archived smoothed recipe and exits nonzero when they differ.

## Repeat a check

Run scripts from the VisCy repository root.
Use `--help` for each script's required inputs.
Use a new output directory.

```bash
uv run --no-sync python applications/dynaclr/examples/organelle_dynamics/verification/verify_paper.py \
  --reference "$PAPER_REFERENCE" --output results/verify-mmd
```

`PAPER_REFERENCE` is the archived `direct_pal_lineage_native30_same_cohort_alignment` directory.
It contains the curve table, matched-sample manifest, PCA scores and run parameters.
Inference and normalization checks require the local datasets listed in the original paper analysis script.
The steps check requires a reference generated with the same normalization mode as the current run.

## Comparison boundaries

Embedding and classifier probabilities use `atol=5e-4`, `rtol=1e-4`.
Classifier labels must match exactly.
The maximum recorded probability difference is 0.0002074.

MMD checks reuse saved PCA scores, matched observations, bandwidths and random seeds.
They use 200 permutations and 50 paired subsamples.
The maximum recorded MMD difference is 2.85e-14.
The paper comparison uses its saved onset-selected population; the default skill compares all treated cells.

BLAS thread settings affect PCA roundoff and exact MMD agreement.
The original thread settings reproduce the archived PCA scores exactly.

Generated outputs remain under:

```text
/hpc/projects/intracellular_dashboard/organelle_dynamics/tmp-test-skill/
```

The cleanup archive is `pr-cleanup-2026-11-23/` within that directory.
It preserves the original separate reports, comparison image and checksums.
The PR retains their numerical evidence in `results.json` without the generated image.
