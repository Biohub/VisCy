---
name: phenotype
description: Normalize cell embeddings against matched controls and generate UMAP plots and MMD comparisons of population shifts.
---

Read the [input/output contract](../../examples/organelle_dynamics/CONTRACT.md).
Set `embeddings` to the embed output's predictions.zarr.
Supply the control label and experiment, marker, control-group and timing metadata.

```sh
uv run --no-sync python -m dynaclr.evaluation.organelle_workflow phenotype --manifest phenotype.yaml --output new-phenotype-directory
```

The command saves control-normalized embeddings, fits PCA80, generates UMAP plots and calculates MMD per physical HPI.
PCA80 retains at least 80% of feature variance.

Use each marker, experiment and control group's own controls for median/MAD normalization.
The current time-matched recipe requires controls at two or more HPI values per group.
Preserve raw encoder embeddings alongside normalized embeddings.
Keep normalization parameters and checksums with the output.
Refit normalization after changing controls or the observation population.

Use all treated observations for population comparisons.
Match controls by experiment, control group and exact HPI.
Keep different model feature spaces separate.
Use PCA features for MMD.
Use UMAP coordinates for visualization.
Report insufficient-cell status rows.
Describe subsampling bands as stability intervals.

Optional saved classifiers consume their compatible raw feature space.
Supply trusted classifier files with matching marker, feature-space identity and checksums.
Keep treatment, supplied infection annotations and classifier predictions separate.
The runner does not currently produce witness scores or witness predictions.

Use the normalize command to save normalization separately.
Use the plot command to generate figures without MMD.
These commands are workflow steps, not separate skills.
Tracking is optional unless selecting track or lineage sampling.
Temporal alignment is outside this skill.

See [validation](../../examples/organelle_dynamics/VALIDATION.md) for component comparisons.
Those checks do not establish complete Figure 6 reproduction.
