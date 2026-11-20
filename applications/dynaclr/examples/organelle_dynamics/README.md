# Embed and phenotype

Use two skills to extract cell embeddings and measure population changes.

| Skill | Required inputs | Outputs |
| --- | --- | --- |
| [embed](../../skills/embed/SKILL.md) | Images, cell-coordinate Parquet, metadata, selected checkpoint, checksum and matching preprocessing configuration. | Raw embeddings and source metadata. |
| [phenotype](../../skills/phenotype/SKILL.md) | Embeddings, control label and experiment, marker, control-group and timing metadata. | Control-normalized embeddings, PCA/UMAP figures and MMD comparisons. |

Cell coordinates are sufficient for these skills.
Track or lineage sampling requires genuine identifiers.

## Normalization

| Operation | Skill | Purpose |
| --- | --- | --- |
| Image cropping, focus selection, resizing, channel selection and intensity normalization | `embed` | Prepare images for the selected checkpoint. |
| Control-relative median/MAD normalization of embedding vectors | `phenotype` | Compare treated cells with their own controls. |

Keep each checkpoint's image preprocessing configuration.
Image normalization and embedding normalization act on different inputs.

## Checkpoint selection

| Model | Manifest settings |
| --- | --- |
| Paper DynaCLR | `model_family: dynaclr`, `inference_recipe: paper`, checkpoint and SHA-256. |
| Other DynaCell/DynaCLR representation checkpoint | `model_family: dynacell`, `inference_recipe: training`, matching `training_config`, checkpoint and SHA-256. |
| Frozen CELL-DINO | `model_family: cell_dino`, matching `training_config`, local `.pth` checkpoint and SHA-256. |

The paper recipe uses the verified DynaCLR architecture and image preprocessing.
Other representation checkpoints require their matching model configuration.
These configurations must expose the VisCy embedding prediction interface.

CELL-DINO uses the [embedding configuration](cell_dino_recipe.yaml).
The selected manifest checkpoint replaces the configuration's `weights_path`.
The runner checks its checksum before loading weights.

Use `predict.embedding_key: projections` when selecting a trained projection-head representation through the training recipe.
Include the checkpoint, preprocessing and selected representation in `feature_space`.
Keep different feature spaces separate during analysis.

## Required resources

| Resource | Requirements |
| --- | --- |
| Software | VisCy revision and `uv.lock`. |
| Images | AI-ready OME-Zarr with channel, pixel-size, focus and normalization metadata. |
| Cell-coordinate Parquet | Source paths, cell identifiers, frames, coordinates, channels, image dimensions and physical pixel sizes. |
| Metadata | `cell_id`, `experiment`, `marker`, `perturbation`, `control_group`, `replicate`, `hours_post_perturbation`. |
| Encoder | Selected checkpoint, SHA-256 checksum and matching model/image configuration. |
| Optional classifiers | Trusted model files, checksums, matching marker and feature-space identity. Supply these to `phenotype`. |

The experiment, cell identifier and marker combination must be unique.
Use `columns` to rename external metadata fields.
Use `metadata` to supply constants for absent fields.
Supply physical timing explicitly.

See the [input/output contract](CONTRACT.md) for coordinate fields and output formats.

## Run the skills

Run commands from the VisCy repository root.
Install the locked environment once:

```bash
uv sync --locked --package dynaclr --extra eval
```

Prepare an embedding manifest from the [DynaCell template](embed_dynacell.yaml) or [CELL-DINO template](embed_cell_dino.yaml).
Replace the example paths and checkpoint checksum.
Use a new output directory for each command.

```bash
# Images and checkpoint-specific preprocessing → raw embeddings.
uv run --no-sync python -m dynaclr.evaluation.organelle_workflow embed \
  --manifest embed.yaml --output results/embedding

# Raw embeddings → control normalization, PCA/UMAP and MMD.
uv run --no-sync python -m dynaclr.evaluation.organelle_workflow phenotype \
  --manifest phenotype.yaml --output results/phenotype
```

Set `embeddings` in the [phenotype manifest](manifest.yaml) to `results/embedding/predictions.zarr`.
Paths resolve against the manifest's `root`.
Keep metadata and feature-space identity consistent between skills.
The `embed` command rejects classifier configurations.
Optional classifiers run inside `phenotype` using compatible raw embeddings.

The `normalize` and `plot` commands remain available for separate intermediate outputs.
The `infer` and `analyze` commands remain available for existing manifests.
These commands do not require additional skills.

## Saved outputs

| Output | Contents |
| --- | --- |
| `predictions.zarr` | Raw encoder `X`, metadata and optional phenotype classifier predictions. Phenotype also stores `obsm/X_control_mad`. |
| `embeddings.parquet` | Raw `embeddings` vectors. Phenotype also stores normalized `ctrl-norm-embedding` vectors. |
| `normalization.json` | Phenotype control parameters and source/output checksums. |
| Coordinate Parquet and PNG/PDF files | Phenotype PCA/UMAP coordinates and figures. |
| `mmd_curves.csv` and `mmd_samples.parquet` | Phenotype population comparisons and selected observations. |
| Resolved configuration and provenance | Input paths, checkpoint identity, settings and software revision. |

Phenotype fits normalization separately for each marker, experiment and control group.
Each group requires controls at two or more HPI values.
HPI means hours post infection in the paper data.
The method uses smoothed control medians and median absolute deviation (MAD) scaling.

MMD compares all treated cells with controls at the same experiment, control group and HPI.
PCA retains at least 80% of normalized feature variance by default.
MMD uses these PCA features.
UMAP provides visualization coordinates.
Subsampling intervals describe stability, not biological replicate confidence.

## Resources and validation

The [resource list](CONTRACT.md#artifact-inventory) identifies paper checkpoints, configurations, classifiers and cached embeddings.
The [validation report](VALIDATION.md) records component comparisons and rerun commands.

| Check | Recorded result |
| --- | --- |
| Paper DynaCLR inference | 96 cells match within float32 tolerance. Maximum absolute difference: 0.000383. |
| CELL-DINO embedding | Two real cell crops produce finite 1,024-feature vectors. Parquet preserves the raw vectors exactly. |
| Phenotype entrypoint | 192 observations retain exact raw vectors, normalized exports and previous MMD tables. |
| Saved infection classifiers | 96/96 labels match. |
| Control normalization and cached PCA | Exact agreement across 257,856 observations. |
| Saved ZIKV HPI MMD | 3,339 comparisons pass across 371 bins. |
| Separate normalization and plotting | 192 observations preserve raw vectors, normalized exports, coordinates and previous MMD results. |

These checks validate workflow components.
They do not establish complete Figure 6 reproduction.
Witness scores, witness predictions and temporal alignment are outside the current skills.

Validation outputs:

```text
/hpc/projects/intracellular_dashboard/organelle_dynamics/tmp-test-skill/
```
