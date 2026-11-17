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
| [organelle-inference](../../skills/organelle-inference/SKILL.md) | Extract embeddings from cell-centered image crops and optionally apply saved classifiers. | AI-ready OME-Zarr images with normalization metadata; cell-coordinate Parquet; experiment metadata; encoder checkpoint and SHA-256; resolved training configuration. For new infection/remodeling predictions, provide compatible classifier artifacts and hashes. | `predictions.zarr`, `predictions.parquet`, resolved configuration and provenance. |
| [organelle-analysis](../../skills/organelle-analysis/SKILL.md) | Plot embeddings by treatment and available infection labels; compare all treated cells with matched controls per HPI. | AnnData embeddings; experiment/control/timing metadata; infection annotations or saved predictions for infection-colored plots. Classifier artifacts are needed only if generating new predictions. | PCA/UMAP coordinates and PNG/PDF plots; `mmd_curves.csv`, `mmd_samples.parquet`, MMD figures and provenance. |

Treatment condition, annotated infection state, predicted infection state, and remodeling predictions remain separate fields. MMD uses **all treated cells**, independently of predicted infection state, and matches controls by experiment, control group and exact HPI.

## What to prepare

| Input | Required information |
| --- | --- |
| Experiment metadata | `experiment`, `marker`, `perturbation`, `control_group`, `replicate`, and `hours_post_perturbation`. HPI can instead be derived from frame `t`, acquisition interval and first-frame HPI. |
| Observation identities | `cell_id` identifies a cell at a frame. The combination of experiment, cell ID and marker must be unique. |
| Cell-coordinate Parquet — inference only | One row per cell/frame/channel in the VisCy cell-index format: source store, well, FOV, channel, native pixel coordinates and frame, image dimensions, physical pixel sizes, and required schema fields. Keep absent tracking fields null. See the [full contract](CONTRACT.md). |
| Images — inference only | AI-ready OME-Zarr with per-frame normalization metadata. Image dimensions and channel names must agree with the cell index. Crops crossing an image boundary are rejected. |
| Models — inference/classification | Pinned encoder checkpoint, matching training/preprocessing configuration, and optional classifier joblib pipelines. Classifiers must match the encoder feature space and marker. |
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

For the second stage, set `embeddings` in `analysis.yaml` to the first stage's `predictions.zarr`. Remove the `classifiers` list if predictions already exist. Paths resolve relative to the manifest's `root`, so adjust that root when moving a bundle. The supplied template is for analysis; inference adds `cell_index`, `training_config`, `checkpoint`, and `checkpoint_sha256` as described in the [contract](CONTRACT.md).

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

The [artifact inventory section](CONTRACT.md#artifact-inventory) identifies the paper checkpoint, configuration, classifiers, embeddings and teacher-label Parquet dependencies. These are currently HPC-local assets; public download locations remain to be established.

## Validation and current limits

Test outputs are under `/hpc/projects/intracellular_dashboard/organelle_dynamics/tmp-test-skill/portable-v1/`:

| Check | Result |
| --- | --- |
| Raw-image inference without track IDs | Eight cells produced finite 768-dimensional embeddings, unique source identities and infection predictions. |
| Real-data analysis | 192 SEC61B observations produced classifier predictions, PCA/UMAP plots and MMD curves at 12, 18 and 24 HPI. |
| External-style metadata | Renamed fields and no tracking metadata reproduced the normalized MMD tables exactly. |
| Artifact inventory | 131 local artifacts were available; file checksums were recorded. |

Five focused tests and pre-commit checks passed. Smoke MMD uses reduced sampling counts; see the [validation details](CONTRACT.md#validated-example).

**This is a draft workflow, not yet a numerical reproduction of the full paper.** The paper's onset-selected cohort differs from the default all-treated population. Generic inference uses resolved training preprocessing; the paper's distinct crop/resampling configuration still needs reconciliation. These execution checks do not establish classifier accuracy on a new acquisition. Time alignment is not included.
