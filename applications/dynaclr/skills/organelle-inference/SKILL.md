---
name: organelle-inference
description: Generate organelle dynamics embeddings and pretrained classifier predictions from AI-ready images and a cell-coordinate Parquet using pinned DynaCLR artifacts.
---

Read the shared [contract](../../examples/organelle_dynamics/CONTRACT.md). Run from the VisCy repository with its locked uv environment:

```sh
uv run --no-sync python -m dynaclr.evaluation.organelle_workflow infer --manifest manifest.yaml --output new-output-directory
```

Validate source-image correspondence on a small control/treated cohort before full inference. Use `inference_recipe: paper` (the default) for the verified FOV-normalized, focus-centered crop/resampling recipe. Use the explicit `training` recipe only when that preprocessing is intended. Never substitute random weights, a different checkpoint, or arbitrary normalization. Cell coordinates do not imply tracks. Missing biological identifiers remain missing.

Use only trusted classifier joblib artifacts with explicit hashes and matching feature-space identities. The runner writes new outputs and refuses existing destinations. Report failures and absent classifier coverage; do not call treatment labels infection predictions. Classifier transfer to a new acquisition is not a validated biological label.

The output predictions.zarr is the input to organelle-analysis. Keep the resolved configuration, provenance and source linkage with the bundle. Time alignment is a separate workflow requiring real temporal identities.
