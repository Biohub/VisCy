---
name: organelle-normalization
description: Normalize raw organelle embeddings against each dataset's own controls and save both representations with fitted median/MAD parameters before UMAP or MMD analysis.
---

Read the shared [contract](../../examples/organelle_dynamics/CONTRACT.md). Set the manifest's embeddings to raw inference predictions.zarr and run:

```sh
uv run --no-sync python -m dynaclr.evaluation.organelle_workflow normalize --manifest manifest.yaml --output new-normalization-directory
```

Keep raw encoder X unchanged. Fit control medians and MAD independently per marker, experiment and control_group using the input dataset's controls. The paper recipe uses time-matched smoothed medians and a fixed robust scale from control residuals; require controls at two HPI values per group. Do not substitute pooled controls from another marker or acquisition. Specify the cohort before fitting; any later cohort change requires a refit.

Check normalized.zarr obsm/X_control_mad and embeddings.parquet ctrl-norm-embedding agree, with the original vectors retained in X and the embeddings Parquet column. Keep normalization.json (control centers, scales, counts and checksums) with the output. Saved classifiers use raw encoder X. Pass normalized.zarr to organelle-plotting or organelle-analysis with unchanged metadata and control label.
