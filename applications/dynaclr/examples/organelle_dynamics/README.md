# Organelle dynamics workflow

Run inference, control normalization, plotting and population analysis on paper data or external microscopy data.
This workflow uses DynaCLR checkpoints and optional saved classifiers.

## Skills

| Skill | Required inputs | Outputs |
| --- | --- | --- |
| [organelle-inference](../../skills/organelle-inference/SKILL.md) | Images, cell-coordinate Parquet, metadata, encoder checkpoint and checksum. Optional classifier models. | Raw embeddings and classifier predictions. |
| [organelle-normalization](../../skills/organelle-normalization/SKILL.md) | Raw embeddings, metadata and the dataset's own controls. | Normalized embeddings, preserved raw embeddings and fitted control parameters. |
| [organelle-plotting](../../skills/organelle-plotting/SKILL.md) | Normalized embeddings and available infection labels. | PCA/UMAP coordinates, figures and fitted models. |
| [organelle-analysis](../../skills/organelle-analysis/SKILL.md) | Embeddings and experiment, control and timing metadata. | PCA/UMAP figures and control-versus-treated MMD curves. |

Cell coordinates are sufficient for inference and population analysis.
Track or lineage sampling requires genuine track or lineage identifiers.

## Required resources

| Resource | Requirements |
| --- | --- |
| Software | VisCy revision and `uv.lock`. |
| Images | AI-ready OME-Zarr with channel, pixel-size, focus and normalization metadata. Required for inference. |
| Cell-coordinate Parquet | Source image paths, cell identifiers, frames, coordinates, channels, image dimensions and physical pixel sizes. Required for inference. |
| Encoder | DynaCLR checkpoint, SHA-256 checksum and inference recipe. |
| Classifiers | Trusted model files, checksums, marker and feature-space identity. Required for new classifier predictions. |
| Metadata | `cell_id`, `experiment`, `marker`, `perturbation`, `control_group`, `replicate`, `hours_post_perturbation`. |
| Manifest | YAML input paths, metadata mappings, control label and analysis settings. See the [template](manifest.yaml). |

The experiment, cell identifier and marker combination must be unique.
Use `columns` to rename external metadata fields.
Use `metadata` to supply constants for absent fields.
Supply physical timing explicitly.

The [input/output contract](CONTRACT.md) specifies coordinate fields, inference recipes, classifier requirements and output formats.

## Run the workflow

Run commands from the VisCy repository root.
Install the locked environment once:

```bash
uv sync --locked --package dynaclr --extra eval
```

Use a new output directory for each command.

```bash
# Extract raw embeddings and optional classifier predictions.
uv run --no-sync python -m dynaclr.evaluation.organelle_workflow infer \
  --manifest inference.yaml --output results/inference

# Normalize embeddings against the dataset's own controls.
uv run --no-sync python -m dynaclr.evaluation.organelle_workflow normalize \
  --manifest normalize.yaml --output results/normalized

# Generate PCA/UMAP figures.
uv run --no-sync python -m dynaclr.evaluation.organelle_workflow plot \
  --manifest analysis.yaml --output results/plots

# Generate figures and MMD tables.
uv run --no-sync python -m dynaclr.evaluation.organelle_workflow analyze \
  --manifest analysis.yaml --output results/analysis
```

| Manifest | Input configuration |
| --- | --- |
| `inference.yaml` | Set `cell_index`, `checkpoint` and `checkpoint_sha256`. The default recipe is `paper`. |
| `normalize.yaml` | Set `embeddings` to `results/inference/predictions.zarr`. |
| `analysis.yaml` | Set `embeddings` to `results/normalized/normalized.zarr`. |

Paths resolve against the manifest's `root`.
Keep the metadata and control label consistent across stages.
Remove `classifiers` when predictions already exist.
The explicit `training` inference recipe also requires `training_config`.

## Saved embeddings and predictions

| Output | Contents |
| --- | --- |
| `predictions.zarr` | Raw encoder `X`, metadata and classifier predictions. Plotting and normalized analysis also retain `obsm/X_control_mad`. |
| `normalized.zarr` | Raw encoder `X` and normalized `obsm/X_control_mad`. |
| `embeddings.parquet` | Raw `embeddings` vectors. Normalized runs also include `ctrl-norm-embedding` vectors. |
| `predictions.parquet` | Metadata, classifier labels and probabilities. |
| `normalization.json` | Fitted control parameters and source/output checksums. |

Normalization fits each marker, experiment and control group separately.
Each group requires controls at two or more HPI values.
HPI means hours post infection in the paper data.
The method uses smoothed control medians and median absolute deviation (MAD) scaling.
Classifiers use raw encoder embeddings.

Plotting checks the saved normalization against the input data.
Run normalization again after changing the cell population, controls or raw embeddings.
The combined `analyze` command also accepts raw embeddings and saves their normalization.

MMD measures differences between populations.
The default analysis compares all treated cells with controls at the same experiment, control group and HPI.
Treatment labels, infection annotations and classifier predictions occupy separate fields.

PCA retains 80% of normalized feature variance by default.
MMD uses these PCA features.
UMAP provides visualization coordinates.
Subsampling intervals describe stability, not biological replicate confidence.

## Paper resources

The [resource list](CONTRACT.md#artifact-inventory) identifies checkpoints, configurations, classifier models and cached embeddings.
Set `PREDICTION_ROOT` to the paper's `epoch105-step84800` directory containing `embeddings/` and `classifier/`.

```bash
# List local resources and file checksums.
uv run --no-sync python applications/dynaclr/examples/organelle_dynamics/build_paper_inventory.py \
  --prediction-root "$PREDICTION_ROOT" --output paper_inventory.json

# Run a small SEC61B analysis with a saved infection classifier.
uv run --no-sync python applications/dynaclr/examples/organelle_dynamics/smoke_paper.py \
  --prediction-root "$PREDICTION_ROOT" --output results/paper-smoke
```

The inventory records local resource paths and file checksums.
Image paths and cell-coordinate Parquet paths also appear in the run manifests.
The inventory has no public download locations.

## Validation

The [validation report](VALIDATION.md) contains comparison methods, tolerances, saved results and rerun commands.

| Check | Recorded result |
| --- | --- |
| Fresh inference | 96 cells across six marker/dataset pairs match within float32 tolerance. Maximum absolute difference: 0.000383. |
| Infection classifiers | 96/96 labels match. Maximum probability difference: 0.0002074. |
| Normalization and cached PCA | Exact agreement across 257,856 observations. |
| Paper MMD | 3,339 comparisons pass across 371 HPI bins. Maximum absolute difference: 2.85e-14. |
| Separate normalization and plotting | 192 cells retain exact raw vectors, normalized exports, plotting coordinates, classifier predictions and previous MMD results. |
| Regression checks | 16 tests and four skill validators pass. |

Paper comparisons use the saved cell population, PCA models, matched samples and random seeds.
The default workflow compares all treated cells.

Validation outputs:

```text
/hpc/projects/intracellular_dashboard/organelle_dynamics/tmp-test-skill/publication-validation/
```

## Proposed changes

The proposed names are `dynacell-embedding` and `dynacell-profiling`.
Consolidation remains undecided.
CELL-DINO selection and separate witness scores and predictions are proposed additions to this workflow.
The four skills listed above remain the implemented interface.
Temporal alignment is outside this workflow.
