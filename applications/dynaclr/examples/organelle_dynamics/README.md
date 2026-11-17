# Organelle dynamics: inference and analysis

Run the organelle dynamics workflow on the paper data or your own microscopy data. Two skills connect image inference to infection-colored embedding plots and control-versus-treated population comparisons over hours post infection (HPI).

```text
Images + cell coordinates + metadata + pretrained models
                         │
                 organelle-inference
                         │
              Embeddings + classifier predictions
                         │
                  organelle-analysis
                         │
             PCA/UMAP plots + MMD curves per HPI
```

You can start at analysis if you already have compatible embeddings. Segmentation and tracking are upstream: per-frame cell coordinates suffice for inference and population analysis. Tracking is required only when explicitly selecting track/lineage sampling. Time alignment is planned as a separate skill.

## Skills and inputs

| Skill | What it does | Required inputs | Main outputs |
| --- | --- | --- | --- |
| [organelle-inference](../../skills/organelle-inference/SKILL.md) | Extract embeddings from cell-centered image crops and optionally apply saved classifiers. | AI-ready OME-Zarr images with normalization metadata; cell-coordinate Parquet; experiment metadata; encoder checkpoint and SHA-256; the paper inference recipe (default), or a resolved training configuration for the explicit training recipe. For new infection/remodeling predictions, provide compatible classifier artifacts and hashes. | `predictions.zarr`, `predictions.parquet`, resolved configuration and provenance. |
| [organelle-analysis](../../skills/organelle-analysis/SKILL.md) | Plot embeddings by treatment and available infection labels; compare all treated cells with matched controls per HPI. | AnnData embeddings; experiment/control/timing metadata; infection annotations or saved predictions for infection-colored plots. Classifier artifacts are needed only if generating new predictions. | PCA/UMAP coordinates and PNG/PDF plots; `mmd_curves.csv`, `mmd_samples.parquet`, MMD figures and provenance. |

Treatment condition, annotated infection state, predicted infection state, and remodeling predictions remain separate fields. MMD uses **all treated cells**, independently of predicted infection state, and matches controls by experiment, control group and exact HPI.

## What to prepare

| Input | Required information |
| --- | --- |
| Experiment metadata | `experiment`, `marker`, `perturbation`, `control_group`, `replicate`, and `hours_post_perturbation`. HPI can instead be derived from frame `t`, acquisition interval and first-frame HPI. |
| Observation identities | `cell_id` identifies a cell at a frame. The combination of experiment, cell ID and marker must be unique. |
| Cell-coordinate Parquet — inference only | One row per cell/frame/channel in the VisCy cell-index format: source store, well, FOV, channel, native pixel coordinates and frame, image dimensions, physical pixel sizes, and required schema fields. Keep absent tracking fields null. See the [full contract](CONTRACT.md). |
| Images — inference only | AI-ready OME-Zarr with FOV normalization and focus metadata for the paper recipe. Image dimensions and channel names must agree with the cell index. Crops crossing an image boundary are rejected. |
| Models — inference/classification | Pinned encoder checkpoint, an explicit inference recipe, and optional classifier joblib pipelines. The default paper recipe uses FOV mean/std normalization, a focus-centered 16-slice window, physical pixel-size rescaling, and 160 × 160 model crops. Classifiers must match the encoder feature space and marker. |
| Manifest | YAML tying these inputs together, with `schema_version: 1`, a root for relative paths, feature-space identity and the control label. See the [analysis template](manifest.yaml). |

For external data, use the manifest's `columns` mapping to rename metadata fields and `metadata` to supply constants for absent fields. Explicitly supply the infection-time reference; it is not inferred from folder names. Different experiment timings require correctly computed per-observation HPI.

## Run the workflow

Run from the VisCy repository root. Set up the environment once:

```bash
uv sync --locked --package dynaclr --extra eval
```

Use a new output directory for every run. Point your agent at the linked `SKILL.md`, or run the same underlying commands directly:

```bash
# Images and coordinates → embeddings and optional classifier predictions
uv run --no-sync python -m dynaclr.evaluation.organelle_workflow infer \
  --manifest inference.yaml --output results/inference

# Embeddings and metadata → plots and MMD tables
uv run --no-sync python -m dynaclr.evaluation.organelle_workflow analyze \
  --manifest analysis.yaml --output results/analysis
```

For the second stage, set `embeddings` in `analysis.yaml` to the first stage's `predictions.zarr`. Remove the `classifiers` list if predictions already exist. Paths resolve relative to the manifest's `root`, so adjust that root when moving a bundle. The supplied template is for analysis; inference adds `cell_index`, `checkpoint`, and `checkpoint_sha256`. It defaults to `inference_recipe: paper`; `inference_recipe: training` additionally requires `training_config`. See the [contract](CONTRACT.md).

Analysis defaults to control-normalized PCA retaining 80% variance for MMD, with a fixed bandwidth per marker. UMAP is used only for visualization. Optional `analysis.representation: raw` uses raw embeddings; `analysis.sampling_unit: track_id` or `lineage_id` requires genuine temporal identities and `fov_name`. Subsampling bands describe stability, not biological-replicate confidence intervals.

## Start with the paper data

`PREDICTION_ROOT` below is the existing `epoch105-step84800` prediction directory containing `embeddings/` and `classifier/`, not the downstream `paper_figure_pseudotime/` directory.

```bash
# List available local paper artifacts and file checksums
uv run --no-sync python applications/dynaclr/examples/organelle_dynamics/build_paper_inventory.py \
  --prediction-root "$PREDICTION_ROOT" --output paper_inventory.json

# Analyze a small real SEC61B cohort with the saved infection classifier
uv run --no-sync python applications/dynaclr/examples/organelle_dynamics/smoke_paper.py \
  --prediction-root "$PREDICTION_ROOT" --output results/paper-smoke
```

The [artifact inventory section](CONTRACT.md#artifact-inventory) identifies the paper checkpoint, configuration, classifiers, embeddings and teacher-label Parquet dependencies. The inventory records the available asset locations and file checksums; supply these paths through your manifest.

## Publication comparison

The workflow was checked against the archived paper data across SEC61B, TOMM20 and G3BP1. The comparison identified and corrected the inference normalization/crop recipe, the MMD null standard deviation (`ddof=1`), and matched-pair subsampling.

| Comparison | Result |
| --- | --- |
| Fresh coordinate-only inference | 96 cells across six marker/dataset pairs agree with archived embeddings within float32 tolerance (`atol=5e-4`, `rtol=1e-4`); maximum absolute difference 3.83e-4. |
| Saved infection classifiers | 96/96 labels agree; maximum probability difference 0.0002074. |
| Control normalization and cached PCA transforms | Exact agreement across 257,856 observations. |
| Paper HPI MMD, uncertainty bands and permutation p-values | All 3,339 metric comparisons passed across 371 bins; maximum absolute difference 2.85e-14. |
| Matched cohorts | Counts, dataset/HPI pairing and unique lineage membership checked for every reference bin. |

See the [validation report and rerun commands](VALIDATION.md) for measured results, tolerances and machine-readable evidence. All generated comparison outputs are under `/hpc/projects/intracellular_dashboard/organelle_dynamics/tmp-test-skill/publication-validation/`.

The default analysis compares all treated cells. To reproduce the paper's plotted cohort, the verification command uses its frozen onset-selected lineages, matched samples, PCA scores, bandwidths and random seeds. These are explicit cohort choices: use the same cohort and fitted representation when comparing numerical curves. The checks establish computational reproduction on the paper data; infection annotations and classifier predictions remain distinct outputs. Time alignment remains a separate workflow.
