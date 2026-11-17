---
name: organelle-plotting
description: Generate PCA and UMAP plots of control-normalized organelle embeddings colored by treatment and available infection annotations or classifier predictions, without running MMD.
---

Read the shared [contract](../../examples/organelle_dynamics/CONTRACT.md). Set embeddings to organelle-normalization's normalized.zarr and run:

```sh
uv run --no-sync python -m dynaclr.evaluation.organelle_workflow plot --manifest manifest.yaml --output new-plot-directory
```

Use the saved control-normalized vectors with PCA80 before seeded 2D UMAP. Plotting rejects missing or stale normalization; run normalize when the cohort or controls change. Each marker has a separate fit shared by its conditions and HPI values. Do not compare axes between markers, or describe this as the movie's joint PCA50/UMAP3D layout.

Keep treatment, provided infection annotations and predicted infection labels separate. Never manufacture infection labels from treatment. Save coordinate Parquet tables, PNG/PDF figures, fitted PCA/UMAP models and provenance. Use organelle-analysis for MMD curves; MMD must use the pre-UMAP representation.
