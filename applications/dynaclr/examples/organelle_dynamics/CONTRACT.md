# Input and output contract

Use the [start page](README.md) for commands and file navigation.

## Inputs

| Input | Required fields or contents |
| --- | --- |
| Manifest | `schema_version: 1`, `root`, `feature_space`, `control_label`. |
| Observation metadata | `cell_id`, `experiment`, `marker`, `perturbation`, `control_group`, `replicate`, `hours_post_perturbation`. |
| Embed inputs | Either `cell_index` or `labels`, plus `checkpoint`, `checkpoint_sha256`, model selection and image preprocessing settings. |
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

## Prepare external data

The agent prepares missing inputs before running the skills.
The runner converts supported nuclear labels into cell coordinates when configured through `labels`.
Image conversion and segmentation remain preparation steps.

1. Inspect the supplied images, coordinates, embeddings and experimental metadata.
2. Ask for missing channel identities, control assignments, pixel sizes and time references.
3. Select the checkpoint and its matching image recipe.
4. Convert only the inputs that do not satisfy the contract below.
5. Check several extracted cell crops against their source images before full inference.

### Images and coordinates

Convert source images into an iohub-compatible HCS OME-Zarr store.
Use the source format's reader and the installed iohub writer.
Preserve axis order `T,C,Z,Y,X`, channel names and physical pixel sizes.
Record the source-to-output position mapping.
Do not guess missing axes or pixel sizes.

Use existing cell coordinates when available.
Otherwise, run an appropriate segmentation or detection method and extract cell centers.
Record the method and parameters.
Convert crop-relative coordinates into native image coordinates.
Tracking is optional for observation sampling.
Keep unavailable track identifiers null.

Build a Parquet table with one row per cell, frame and selected channel.
Include these fields alongside the experimental metadata above:

| Fields | Meaning |
| --- | --- |
| `store_path`, `well`, `fov` | Absolute image-store path and position components. |
| `t`, `y`, `x`, `z` | Native coordinates; `t` is an integer frame index. |
| `channel_name`, `microscope`, `tracks_path` | Source descriptors; retain a null `tracks_path` when tracks are absent. |
| `T_shape`, `C_shape`, `Z_shape`, `Y_shape`, `X_shape` | Dimensions read from the source image. |
| `pixel_size_xy_um`, `pixel_size_z_um`, `z_focus_mean` | Calibration and focus values required by the selected recipe. |

Use unique `cell_id` values for cell/frame observations.
Keep biological `track_id` separate.
Map CSV column names explicitly instead of deriving treatments from filenames.
For registered tracking datasets, `dynaclr build-cell-index --help` describes the existing collection converter.
For other tables, use pandas and preserve custom metadata with `DataFrame.to_parquet`.
The fixed-schema cell-index writer omits additional fields such as `control_group` and `replicate`.

### Data without tracking

For nuclear-label input, `embed` creates tracking-format Parquet tables and a canonical cell index before inference.
For other detection formats, the agent prepares the canonical cell index.
Each detected cell is an independent instance.
This applies to single-timepoint images and independently segmented frames without temporal links.
Use one row per cell/frame, shared across the selected image channels.

| Column | Value |
| --- | --- |
| `id` | Unique integer observation identifier within the FOV. |
| `track_id` | The same independent instance identifier as `id`; never reused across frames. |
| `t` | Native frame index; zero for a single-frame image. |
| `y`, `x` | Cell centroid in native image pixels. |
| `parent_track_id`, `parent_id` | `-1`, indicating no parent relationship. |

Example for two cells in one single-frame image:

```text
track_id,t,y,x,id,parent_track_id,parent_id
1,0,120.5,200.0,1,-1,-1
2,0,310.0,145.5,2,-1,-1
```

The runner writes each FOV table to `tracks/<row>/<col>/<fov>/tracks_<row>_<col>_<fov>.parquet` within the output directory.
Preserve the mapping from generated identifiers to source image, frame and segmentation label.
Do not reuse segmentation label values as identifiers when labels restart in another frame.
Preparation provenance records `track_identity: independent_instances` and the generated table location.
These identifiers describe observations; they do not establish biological continuity.

Build the canonical cell index from these observations for `embed`.
Preserve a unique `cell_id` containing the source position and instance identity.
Keep biological track and lineage fields null in the canonical index and embedding outputs.
Keep its `tracks_path` null.
Use observation sampling for these data.
Do not use the generated identifiers for temporal pairing, track sampling, lineage sampling or time alignment.

Set `hours_post_perturbation` from the experimental time reference.
Frame zero does not imply zero hours post perturbation.
Single-timepoint embedding does not require temporal tracking.
For single-timepoint phenotype analysis, select `per_timepoint` or `pooled` normalization.

To use nuclear labels, replace `cell_index` with this block in the embedding configuration:

```yaml
labels:
  store: images.zarr
  label: nuclei
  channels:
    - name: GFP
      marker: SEC61B
  perturbation_wells:
    uninfected: [A/1]
    treated: [A/2]
metadata:
  experiment: experiment-1
  control_group: acquisition-1
  replicate: replicate-1
  microscope: microscope-1
timing:
  first_frame_hpi: 24
  # interval_minutes: 30  # Required only when images contain multiple frames.
```

Replace the example values with the acquisition metadata.
Run the same `dynacell embed --config … --output …` command.
Labels must occupy `labels/nuclei/0` with shape `(T, 1, 1, Y, X)` and integer values.
Zero denotes background; positive values identify cells within each frame.
Image and label dimensions must agree in T, Y and X.
The training recipe supports label input for Z=1 images; supply coordinates for training-recipe Z-stacks.
The paper recipe retains its existing focus requirements for Z-stacks.
The runner filters boundary cells using the selected recipe's crop geometry.
`cells.parquet` and `excluded_cells.parquet` preserve `instance_id` and `nucleus_label` for source mapping.
`cells_summary.json` records counts and the absence of temporal identity.
The FOV tracking-format tables include all detected instances, including those excluded from inference.

### Image statistics and focus

The runner currently requires image normalization metadata for every encoder, including CELL-DINO.
Compute missing statistics on the prepared store:

```python
from viscy_utils.meta_utils import generate_normalization_metadata

generate_normalization_metadata("/absolute/path/images.zarr", num_workers=1)
```

This writes normalization attributes into the image store.
It computes both FOV and per-timepoint statistics.
Use FOV statistics for the paper recipe and per-timepoint statistics for the training recipe.

For recipes requiring focus estimation, use `qc run -c focus.yml` with the [focus configuration](../../../qc/configs/focus.yml).
Install the `viscy-qc` workspace package when this command is needed.
Set optical parameters from the acquisition metadata.
The paper recipe requires Phase3D focus metadata.
Read `focus_slice` metadata into the cell table when the selected recipe needs it.
Check crop boundaries using the selected checkpoint's crop size and pixel-size conversion.
Record excluded observations explicitly.

### Existing embeddings and metadata

For existing embeddings, prepare AnnData Zarr with raw vectors in `X` and aligned metadata in `obs`.
Preserve the checkpoint and preprocessing identity in `feature_space`.
Do not treat already normalized vectors as raw embeddings.
Images and segmentation are unnecessary for this route.

Use `columns` to rename source fields and `metadata` for known constants:

```yaml
columns:
  object_id: cell_id
  treatment: perturbation
metadata:
  experiment: experiment-1
  control_group: acquisition-1
  replicate: replicate-1
timing:
  interval_minutes: 30
  first_frame_hpi: 3
```

Replace these example values with actual experimental facts.
Preserve per-observation values when experiments differ.
Single-timepoint data supports `per_timepoint` and `pooled` normalization.

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
Select the temporal grouping in the phenotype configuration:

```yaml
normalization:
  mode: per_timepoint  # Alternatives: pooled, time_matched.
```

| Mode | Center and scale | Required controls |
| --- | --- | --- |
| `per_timepoint` | Control median and `1.4826 * MAD` fitted separately at each exact HPI. No smoothing or interpolation. | Controls at every observed HPI. |
| `pooled` | One control median and `1.4826 * MAD` across all control observations in the group. | Controls somewhere within the group. |
| `time_matched` | Default paper recipe: smoothed temporal centers and pooled residual MAD. | Controls at two or more HPI values. |

Pooled normalization weights each control observation equally; timepoints with more controls contribute more observations.
The paper recipe interpolates control medians across HPI and smooths them with Gaussian sigma 1 using nearest boundaries.
It scales centered embeddings by `1.4826 * MAD` of control residuals pooled across HPI.
The scale floor is the first percentile of positive MAD values.
Each fitted scale uses its own floor; missing required controls or all-zero MAD values fail normalization.
Both new modes support a single timepoint when controls have nonzero variation.
Raw embeddings remain unchanged.
`normalization.json` records the mode, reference timepoints, fitted centers, scales and control counts.
Each pooled fit stores one center and scale; each per-timepoint fit stores its own center and scale.
Saved normalization from another mode is rejected.
Start from the raw embedding output or explicitly rerun `normalize` when changing modes.

PCA fits each marker separately and retains at least 80% of variance.
Conditions and timepoints share that marker's fit.
MMD compares all treated cells with controls from the same experiment, control group and exact HPI.
This matching rule also applies when normalization pools controls across timepoints.
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
