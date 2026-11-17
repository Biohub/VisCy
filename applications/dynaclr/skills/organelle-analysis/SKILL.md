---
name: organelle-analysis
description: Plot organelle embeddings by infection state and calculate physical-HPI control-versus-treated population MMD from standardized AnnData outputs.
---

Read the shared [contract](../../examples/organelle_dynamics/CONTRACT.md). Run:

```sh
uv run --no-sync python -m dynaclr.evaluation.organelle_workflow analyze --manifest manifest.yaml --output new-analysis-directory
```

Use all treated observations regardless of classifier predictions. Match controls by experiment, control_group and exact physical HPI. Keep marker/model feature spaces separate. Never invent missing controls or infection labels. Inspect insufficient-cell rows in mmd_curves.csv and report them.

PCA and seeded UMAP are visualizations; MMD uses control-normalized PCA80 features by default, never UMAP; representation: raw selects encoder X. The default cohort contains all treated cells, unlike the paper onset-selected cohort. Label subsampling bands as stability intervals, not biological replicate confidence intervals. For tracked data set sampling_unit: lineage_id (or track_id); require real fov_name and temporal identities.

For paper reproduction, consult the paper provenance and original analyze_direct_pal_aligned_organelle_mmd.py script. Do not compare numeric curves across different cohorts or claim the portable preset reproduces the paper. Alignment is outside this skill.
