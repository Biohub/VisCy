---
name: organelle-analysis
description: Plot organelle embeddings by infection state and calculate physical-HPI control-versus-treated population MMD from standardized AnnData outputs.
---

Read the shared [contract](../../examples/organelle_dynamics/CONTRACT.md). Run:

```sh
uv run --no-sync python -m dynaclr.evaluation.organelle_workflow analyze --manifest manifest.yaml --output new-analysis-directory
```

Use all treated observations regardless of classifier predictions. Match controls by experiment, control_group and exact physical HPI. Keep marker/model feature spaces separate. Never invent missing controls or infection labels. Inspect insufficient-cell rows in mmd_curves.csv and report them.

Prefer the explicit organelle-normalization step first and set embeddings to its normalized.zarr. The combined analyze command also fits and saves normalization for raw inputs; it validates and reuses existing workflow normalization. PCA and seeded UMAP are visualizations; MMD uses control-normalized PCA80 features by default, never UMAP; representation: raw selects encoder X. The default cohort contains all treated cells, unlike the paper onset-selected cohort. Label subsampling bands as stability intervals, not biological replicate confidence intervals. For tracked data set sampling_unit: lineage_id (or track_id); require real fov_name and temporal identities.

For numerical paper reproduction, use the verified frozen-cohort commands in [VALIDATION.md](../../examples/organelle_dynamics/VALIDATION.md). They recompute the archived HPI curves with the same estimator used by this skill. Keep cohort selection, PCA fit, bandwidth and seeds fixed for that comparison. Alignment is outside this skill.
