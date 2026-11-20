---
name: organelle-inference
description: Generate organelle dynamics embeddings and pretrained classifier predictions from AI-ready images and a cell-coordinate Parquet using pinned DynaCLR artifacts.
---

Read the shared [contract](../../examples/organelle_dynamics/CONTRACT.md). Run from the VisCy repository with its locked uv environment:

```sh
uv run --no-sync python -m dynaclr.evaluation.organelle_workflow infer --manifest manifest.yaml --output new-output-directory
```

Ask the user for the hours post perturbation at the first frame (t=0) and the frame interval when the dataset does not record them; the start HPI differs between datasets and is never assumed. Set them as `timing: {first_frame_hpi, interval_minutes}`.

Without a cell-coordinate Parquet, build one from nuclear labels in the image store (`labels/<name>` at each position):

```sh
uv run --no-sync python -m dynaclr.evaluation.organelle_workflow cells-from-labels --manifest manifest.yaml --output new-cells-directory
```

The manifest's `labels` block names the store, the label, the `channels` (zarr name and marker) and `perturbation_wells`. Each nucleus at each frame becomes one untracked observation at its label centroid. Label values are per-frame instance IDs, never track IDs. Nuclei whose crop would cross the image border are listed in excluded_cells.parquet; report those counts. Set the inference manifest's `cell_index` to the output cells.parquet.

Validate source-image correspondence on a small control/treated cohort before full inference. Use `inference_recipe: paper` (the default) for the verified FOV-normalized, focus-centered crop/resampling recipe. Use the explicit `training` recipe only when that preprocessing is intended. Never substitute random weights, a different checkpoint, or arbitrary normalization. Cell coordinates do not imply tracks. Missing biological identifiers remain missing.

Use only trusted classifier joblib artifacts with explicit hashes and matching feature-space identities. The runner writes new outputs and refuses existing destinations. Report failures and absent classifier coverage; do not call treatment labels infection predictions. Classifier transfer to a new acquisition is not a validated biological label.

Save raw embeddings: predictions.zarr keeps encoder X and embeddings.parquet exports the same vectors as embeddings. The output predictions.zarr is the input to organelle-normalization, followed by organelle-plotting or organelle-analysis. Keep the resolved configuration, provenance and source linkage with the bundle. Time alignment is a separate workflow requiring real temporal identities.
