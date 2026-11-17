# Organelle workflow contract v1

Start with the [workflow README](README.md) for the skill list, inputs and commands. This document specifies the detailed data contract.

Skills live under `applications/dynaclr/skills/`; install or point your agent at their SKILL.md files. Run commands from the VisCy root. Sync the locked environment once (`uv sync --locked --package dynaclr --extra eval`); use `uv run --no-sync` during jobs.

A YAML manifest has `schema_version: 1`, `root` (relative to the manifest), `feature_space` (immutable encoder identity), `control_label`, and `embeddings` for cached analysis. Paths resolve against root. `columns` maps source column names to canonical names; `metadata` supplies explicit constants only for absent columns.

Required observation fields: `cell_id` (unique cell/frame identity), `experiment`, `marker`, `perturbation`, `control_group`, `replicate`, `hours_post_perturbation`. The composite experiment/cell_id/marker key is unique. Optional infection_state is an independent annotation; predicted_infection_state is classifier output. Track/lineage fields are optional and never synthesized. If HPI is absent, `timing: {interval_minutes: 30, first_frame_hpi: 3}` computes it from zero-based `t`. Do not combine experiments with different timing using one constant.

Inference additionally requires `cell_index`, `checkpoint`, `checkpoint_sha256`, and the canonical viscy-data cell-index schema (one row per cell/frame/channel). Coordinates t/y/x are zero-based native image indices. well is the plate-relative well (e.g. A/2), and fov is its leaf name (e.g. fov0), so store_path/well/fov addresses the image position. T_shape/C_shape/Z_shape/Y_shape/X_shape dimensions are required. Image store paths, FOV, channel names, physical pixel sizes, focus and normalization metadata must agree with the training data loader. Supply AI-ready OME-Zarr; conversion and segmentation are upstream. Preserve absent tracking fields as null.

`classifiers` is a list of trusted artifacts with path, sha256, feature_space, marker, task (e.g. infection_state), exposing predict/predict_proba and classes_ (or classifier.classes_). These may include preprocessing pipelines. Never load untrusted pickle/joblib. Predicted labels follow the saved model's decision rule; probability columns retain class labels. A classifier list can be omitted for already classified embeddings.

`analysis` defaults: seed 42, min_cells 20, max_cells 180, permutations 200, repeats 50. Controls match experiment/control_group/exact HPI. The portable v1 preset samples observations and uses control-normalized PCA retaining 80% variance, with one fixed median bandwidth per marker. Control medians are smoothed with sigma 1 across HPI and scaled by fixed control MAD with a 1% floor. Set representation: raw to use encoder X, or sampling_unit: track_id / lineage_id to select one observation per temporal unit per exact HPI. Temporal sampling requires fov_name and the requested identity. The default cohort contains all treated cells. The paper reproduction command uses the archived onset-selected cohort and fitted representation; its numerical comparisons are recorded in [VALIDATION.md](VALIDATION.md). Missing controls yield explicit status rows. Curves are separate per experiment and treatment; replicate pooling is not performed.

Outputs: predictions.zarr, predictions.parquet, resolved_manifest.yaml, provenance.json, artifacts.json; analysis adds per-marker coordinates, PCA/UMAP PNG/PDF plots, mmd_curves.csv, mmd_samples.parquet and MMD figures. Hashed figure stems avoid unsafe metadata filenames; coordinate tables carry marker identity. Output directories must not already exist. Failed directories are retained without a completion provenance file.

To analyze inference output, copy the input manifest, set embeddings to the output predictions.zarr, remove classifiers to avoid recomputation, and select a new output directory.

## Artifact inventory

Required for inference: VisCy repository/revision and uv.lock; resolved training config; encoder checkpoint with SHA-256; AI-ready image stores with normalization; cell-index Parquet; explicit experiment/control/timing metadata. Required for new classifier predictions: compatible immutable classifier pipelines, class mappings, checksums and their provenance. Required for cached analysis: embedding AnnData and the metadata above; raw images and GPU are not needed.

Paper encoder: `epoch=105-step=84800.ckpt` under `/hpc/projects/organelle_phenotyping/models/DynaCLR-2D-MIP-BagOfChannels/2d-mip-ntxent-t0p2-lr2e5-bs256-192to160-zext11-single-marker-fix-shuffler/DynaCLR-2D-MIP-BagOfChannels/jbrwhzr3/checkpoints/`. A .sha256 sidecar exists. The resolved config is `config.yaml` at the run root. Do not use mutable last.ckpt.

Paper reproduction also requires per-marker embeddings, direct-pAL teacher `predictions_by_threshold.parquet`, lineage information, paper cohort manifests and run_info.json. The supplied paper_figure_pseudotime directory contains plotting provenance referencing parent mmd_curves.csv; plots alone are insufficient to rerun inference. The inventory records the current local locations and hashes of these assets. The real SEC61B infection classifier is classifier/DynaCLR/infection_state__pAL17_to_SEC61B/pipelines/infection_state_SEC61B.joblib relative to the prediction root. The inventory generator lists other classifier artifacts and checksums. These linear classifiers predict infection from marker embeddings. The direct-pAL teacher Parquet supplies the separate infection-onset cohort used by the paper MMD analysis; do not interchange these model roles.

## Validated example

Generate an inventory with `uv run --no-sync python applications/dynaclr/examples/organelle_dynamics/build_paper_inventory.py --prediction-root PATH --output inventory.json`. Run the small real-data example with `uv run --no-sync python applications/dynaclr/examples/organelle_dynamics/smoke_paper.py --prediction-root PATH --output NEW_DIRECTORY`.

Executed validation under `/hpc/projects/intracellular_dashboard/organelle_dynamics/tmp-test-skill/portable-v1/`:

- `analysis-normalized`: 192 nonblank SEC61B observations, 32 controls and 32 treated cells at each of 12/18/24 HPI; real saved infection classifier, PCA/UMAP and MMD figures.
- `external-analysis`: removed track metadata, renamed ID/treatment columns, derived HPI from frame metadata; MMD tables matched the normalized example exactly.
- `raw-inference-v6`: eight real image crops with track IDs omitted; pinned epoch105 checkpoint; finite 768-dimensional embeddings, unique source identities and infection predictions.
- `../paper_inventory.json`: 131 available local artifacts, including checksummed classifier and encoder files.

Smoke MMD uses 20 permutations and five subsamples for speed. Publication comparisons use the full saved settings: 200 permutations and 50 paired subsamples. See [VALIDATION.md](VALIDATION.md) for the full-cohort numerical comparison and fresh inference checks.

The default `inference_recipe: paper` uses FOV mean/std normalization, Phase3D focus, a 16-reference-slice window with offset 0.3, MIP for fluorescence, 0.1494 µm XY and 0.174 µm Z reference sampling, and 160 × 160 model crops. It reuses the paper Triplet transformations with a coordinate-only adapter; internal missing-track sentinels are never exported as biological tracks. Subpixel coordinates are rounded to the nearest native pixel. The explicit `inference_recipe: training` route requires `training_config` and retains its per-frame normalization. Both routes reject missing/invalid statistics. The paper route records filtered boundary observations in `excluded_cells.parquet` and fails if any requested observation was excluded. It checks every requested observation appears exactly once; incomplete threaded-loader results fail rather than silently producing a successful bundle.
