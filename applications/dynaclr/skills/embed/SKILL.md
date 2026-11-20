---
name: embed
description: Generate cell embeddings with CELL-DINO or DynaCell/DynaCLR representation checkpoints, including the selected model's image preprocessing.
---

Read the [input/output contract](../../examples/organelle_dynamics/CONTRACT.md).
Select the checkpoint and its matching configuration before inference.

| Model | Manifest settings |
| --- | --- |
| Paper DynaCLR | `model_family: dynaclr`, `inference_recipe: paper`, checkpoint and SHA-256. |
| Other DynaCell/DynaCLR representation checkpoint | `model_family: dynacell`, `inference_recipe: training`, matching `training_config`, checkpoint and SHA-256. |
| Frozen CELL-DINO | `model_family: cell_dino`, CELL-DINO `training_config`, local `.pth` checkpoint and SHA-256. |

The [CELL-DINO configuration](../../examples/organelle_dynamics/cell_dino_recipe.yaml) supplies its model and image preprocessing settings.
The manifest checkpoint replaces its `weights_path`.
Use the matching training configuration for other representation checkpoints.
Set `predict.embedding_key: projections` only when the requested representation uses a trained projection head.
Include the selected output and preprocessing recipe in `feature_space`.

Supply AI-ready images, cell coordinates and the [required metadata](../../examples/organelle_dynamics/CONTRACT.md).
Keep cropping, focus selection, resizing, channel selection and image intensity normalization inside this step.
Preserve the checkpoint's preprocessing recipe.
Check source-image correspondence on a small sample before full inference.

```sh
uv run --no-sync python -m dynaclr.evaluation.organelle_workflow embed --manifest embed.yaml --output new-embedding-directory
```

The runner checks the checkpoint checksum before loading weights.
It rejects existing output directories and missing observation identities.
It saves raw vectors in predictions.zarr and embeddings.parquet.
Keep the checkpoint identity, resolved configuration and source metadata with these outputs.

Put classifier models in the phenotype manifest.
Pass predictions.zarr to phenotype for control-relative embedding normalization, UMAP and MMD.
Cell coordinates do not establish track identities.
