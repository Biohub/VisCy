# Input and output contract

Use the [start page](README.md) for commands and file navigation.

## Inputs

| Input | Required fields or contents |
| --- | --- |
| Manifest | `schema_version: 1`, `root`, `feature_space`, `control_label`. |
| Observation metadata | `cell_id`, `experiment`, `marker`, `perturbation`, `control_group`, `replicate`, `hours_post_perturbation`. |
| Embed inputs | `cell_index`, `checkpoint`, `checkpoint_sha256`, model selection and image preprocessing settings. |
| Phenotype input | `embeddings`: AnnData Zarr with raw vectors in `X` and observation metadata in `obs`. |
| Optional classifiers | Trusted files with `path`, `sha256`, `feature_space`, `marker` and `task`. |

Paths resolve against `root`, relative to the manifest location.
The experiment, cell identifier and marker combination must be unique.
Use `columns` for source-to-canonical field mappings.
Use `metadata` for constants in absent fields.

HPI means hours post infection in the paper data.
The general field is `hours_post_perturbation`.
When absent, `timing` computes it from `t * interval_minutes / 60 + first_frame_hpi`.
Supply timing per observation when experiments use different time references.

Cell-coordinate Parquet must follow the [viscy-data schema](../../../../packages/viscy-data/src/viscy_data/cell_index.py).
Each row identifies one cell, frame and channel.
Coordinates use zero-based native image indices.
`store_path/well/fov` identifies the OME-Zarr position.
Image dimensions, channels, pixel sizes, focus and normalization metadata must agree with the source images.
Keep absent biological track identifiers null.

## Models

| Selection | Configuration |
| --- | --- |
| Paper DynaCLR | `model_family: dynaclr`, `inference_recipe: paper`. |
| Other DynaCell/DynaCLR representation checkpoint | `model_family: dynacell`, `inference_recipe: training`, matching `training_config`. |
| Frozen CELL-DINO | `model_family: cell_dino`, [cell_dino_recipe.yaml](cell_dino_recipe.yaml), local `.pth` checkpoint. |

The paper recipe uses FOV mean/std normalization and 160-pixel crops.
It uses Phase3D focus, a 16-slice reference window, offset 0.3 and fluorescence maximum-intensity projection.
Reference sampling is 0.1494 µm in XY and 0.174 µm in Z.
The runner rounds coordinates to native pixels and rejects excluded observations.

Training configurations must expose the VisCy embedding prediction interface.
CELL-DINO loads the verified checkpoint through `weights_path`.
Its wrapper resizes images and applies per-image, per-channel spatial z-score normalization.
Set `predict.embedding_key: projections` only for a trained projection-head representation through the training recipe.
Record checkpoint, preprocessing and representation selection in `feature_space`.

`embed` rejects classifiers.
Phenotype classifiers must expose `predict`, `predict_proba` and class labels.
They use compatible raw embeddings.

## Analysis

Normalization fits controls independently for each marker, experiment and control group.
It requires controls at two or more HPI values per group.
It interpolates control medians across HPI and smooths them with Gaussian sigma 1 using nearest boundaries.
It subtracts these centers and scales by `1.4826 * MAD` of control residuals across HPI.
The scale floor is the first percentile of positive MAD values.
Missing controls or all-zero MAD values fail normalization.

PCA fits each marker separately and retains at least 80% of variance.
Conditions and timepoints share that marker's fit.
MMD compares all treated cells with controls from the same experiment, control group and exact HPI.
It uses one median bandwidth per marker.
UMAP provides visualization coordinates only.

| Setting | Default |
| --- | --- |
| `seed` | 42 |
| `min_cells` / `max_cells` | 20 / 180 |
| `permutations` / `repeats` | 200 / 50 |
| `representation` | `control_pca80`; `raw` selects unnormalized encoder vectors. |
| `sampling_unit` | `observation`; `track_id` or `lineage_id` requires genuine identifiers and `fov_name`. |

Insufficient matched cells produce explicit status rows.
Subsampling intervals describe stability, not biological replicate confidence.
Treatment, infection annotations and classifier predictions remain separate.

## Outputs

| Output | Contents |
| --- | --- |
| `predictions.zarr` | Raw `X`, metadata and optional classifier predictions. Phenotype also includes `obsm/X_control_mad`. |
| `embeddings.parquet` | Raw `embeddings`; phenotype also includes normalized `ctrl-norm-embedding`. |
| `predictions.parquet` | Metadata, labels and probabilities. |
| `normalization.json` | Fitted control parameters and source/output checksums. |
| Coordinate Parquet, PNG/PDF, fitted models | Per-marker PCA/UMAP results. |
| `mmd_curves.csv`, `mmd_samples.parquet` | Population comparisons and selected observation identities. |
| `resolved_manifest.yaml`, `provenance.json`, `artifacts.json` | Input paths, settings and output inventory. |

`observation_id` links embedding vectors to observation metadata.
Hashed filenames prevent marker labels from creating unsafe paths.
Use new output directories.
Failures omit the completion provenance file.

The `normalize` command separately writes `normalized.zarr`.
The `plot` command skips MMD.
The older `infer` and `analyze` commands remain available.
Refit normalization after changing the input population, controls or raw vectors.

## Paper resources

The [inventory helper](tools/build_paper_inventory.py) lists local checkpoints, configurations, classifiers and cached embeddings.
The [paper example](tools/smoke_paper.py) selects a small SEC61B population with a saved infection classifier.
Both accept `--prediction-root` and `--output`.
Use the paper's `epoch105-step84800` prediction directory as `--prediction-root`.
Image and coordinate paths also appear in run manifests.
The current resource inventory uses HPC paths without public download URLs.
The recorded inventory is:

```text
/hpc/projects/intracellular_dashboard/organelle_dynamics/tmp-test-skill/paper_inventory.json
```
