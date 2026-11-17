# 2D virtual staining on DynaCell: plug in a new source channel, train, predict and score

This tutorial walks through one loop of the 2D DynaCell workflow on the HPC:

1. add a new label-free channel (for example, a waveorder phase reconstruction from brightfield) to the train and test data;
2. train the 2D FNet baseline on it;
3. predict on the DynaCell-lite test set;
4. score the predictions with the fast metric tier.

Every command below was run once on an A40 node on 2026-11-17, unless it is marked **(not run)**. They were run as `uv run --no-sync ...` in an already-synced checkout; the `uv run` forms shown are the documented equivalents. The run log is `/hpc/mydata/alex.kalinin/VisCy-worktrees/fnet2d-vscyto2daug/applications/dynacell/experiments/2026-11-17_tutorial-2d-check/RESULTS.md` (not in git).

The running example is nucleus, trained on iPSC. The variant tag is `wo1`: one waveorder reconstruction, stored as channel `PhaseWO1`. Use one tag per reconstruction.

## 0. The pieces

| Piece | Path |
|---|---|
| Branch | `dynacell-fnet2d-vscyto2daug` (VisCy) |
| Baseline train config | `applications/dynacell/configs/benchmarks/virtual_staining/nucleus/fnet2d_vscyto2daug/ipsc_confocal/train.yml` |
| Second seed of the baseline | `.../nucleus/fnet2d_vscyto2daug_seed1/ipsc_confocal/train.yml` |
| Baseline lite predict configs | `.../nucleus/fnet2d_vscyto2daug{,_seed1}/ipsc_confocal/predict__{ipsc_confocal,a549_mantis_h2b_{mock,denv,zikv}}_lite.yml` |
| iPSC train store (5-plane in-focus slab) | `/hpc/projects/virtual_staining/training/dynacell/ipsc/dataset_v4/train/cell_focus.zarr` |
| Its full-Z source | `/hpc/projects/virtual_staining/training/dynacell/ipsc/dataset_v4/train/cell.zarr` (500 FOVs, channels Brightfield, Nuclei, Membrane, Phase3D) |
| iPSC lite test store | `/hpc/projects/virtual_staining/training/dynacell_lite/ipsc/dataset_v4/test_cropped/cell.zarr` (50 FOVs, (1, 4, 40, 512, 512)) |
| A549 lite test stores | `/hpc/projects/virtual_staining/training/dynacell_lite/a549/mantis/test/dual_nucl_memb_{mock,DENV,ZIKV}.zarr` (12 FOVs × 5 timepoints, (5, 4, 48, 640, 960)) |
| Lite plane files (which Z planes a 2D model predicts) | `/hpc/projects/virtual_staining/training/dynacell_lite/focus2d_planes/nucleus/{ipsc,a549__mock,a549__denv,a549__zikv}__h1.json` |
| Tool to build stores with new channels | `applications/dynacell/tools/assemble_channel_store.py` |
| Tool to cut the 5-plane train slab | `applications/dynacell/tools/extract_focus_slab_store.py` |

**The baseline recipe (`fnet2d_vscyto2daug`):**
- **Model and input:** FNet run in 2D, one in-focus plane per sample, 384 × 384 patches.
- **Augmentations:** the VSCyto2D training augmentations (affine, contrast, intensity, noise and blur), cropped from 600 × 600.
- **Training:** batch 48, MSE loss, 200k steps, bf16-mixed, cuDNN benchmark and channels-last.
- **Measured speed:** 0.308 s/step on an A100 and 0.505 s/step on an A40, so 200k steps take about 17 h and 28 h.

**Plane files and lite caches already exist.** Every test store carries a Brightfield channel, so you can reconstruct phase on both the train and the test side. The plane files are picked from the ground-truth nuclei, not from the input channel, so one file serves every reconstruction.

## 1. Set up once

```sh
git clone git@github.com:Biohub/VisCy.git && cd VisCy
git checkout dynacell-fnet2d-vscyto2daug
uv sync --all-packages --all-extras --frozen   # needs the HPC uv cache symlink, see CLAUDE.md
uv run wandb login                             # training logs to wandb project "dynacell"
```

- **Run from your own checkout.** Jobs run in the repository of the launcher script you call. If you run Alex's copy, your jobs use his tree and his environment.
- **Write permissions.** You need write access, through `group.virtual.staining`, to:
  - `/hpc/projects/virtual_staining/training/dynacell/ipsc/dataset_v4/train/`;
  - `/hpc/projects/virtual_staining/training/dynacell_lite/`;
  - `/hpc/projects/comp.micro/virtual_staining/models/dynacell/ipsc/nucleus/`.

  You cannot create new top-level folders under `.../models/dynacell/`.
- **GPU work outside sbatch.** Run preprocessing, stats, predict and eval inside a GPU allocation, for example `srun -p gpu --gres=gpu:1 -c 16 --mem=128G -t 4:00:00 --pty bash`.

## 2. Add a reconstruction as a new channel

### 2.1 What the waveorder output must look like

- **Positions and shape:** the same position names (`row/col/fov`) and the same T, Z, Y and X as the store it was reconstructed from. waveorder keeps position names when you reconstruct `<store>/*/*/*`.
- **Dimension:** use `reconstruction_dimension: 3`. A 2D reconstruction writes one Z plane named `Phase2D`. That breaks the plane files and the slab extractor, and `assemble_channel_store.py` refuses it.
- **Data type:** float32.
- **Scale:** keep the input's scale, so do not pass `--write-config-scale-to-output`. The iPSC stores carry a placeholder `[1, 1, 1, 1, 1]` scale, so waveorder will warn about a mismatch with the physical pixel size. That warning is expected.
- **Channel name:** a 3D phase reconstruction is named `Phase3D`, the same as the existing channel. The next step renames it.

Reconstruct each train and test store you need: `train/cell.zarr`, the lite iPSC `cell.zarr` and, for out-of-domain tests, the three A549 lite stores. Writing waveorder outputs to stores and running waveorder itself are **(not run)** here.

### 2.2 Build a train store with the new channel

The 5-plane train store is cut from a full-Z store, and its statistics come from that full-Z store. So assemble the full-Z store first, compute statistics, then cut the slab. Never write into or `--overwrite` `cell_focus.zarr`, because running fits read it.

```sh
T=/hpc/projects/virtual_staining/training/dynacell/ipsc/dataset_v4/train
# 1. Slim full-Z store: Phase3D (picks the in-focus slab), Nuclei (target), PhaseWO1 (new source)
uv run python applications/dynacell/tools/assemble_channel_store.py \
  --base $T/cell.zarr --keep Phase3D Nuclei \
  --add <waveorder_train_output.zarr>:Phase3D:PhaseWO1 \
  --output $T/cell_wo1.zarr
# 2. Normalization statistics for the new channel (needs write access)
uv run viscy preprocess --data_path $T/cell_wo1.zarr --channel_names '["PhaseWO1"]' --num_workers 8
# 3. Cut the 5-plane in-focus slab; the slab is picked from Phase3D, like the production store
uv run python applications/dynacell/tools/extract_focus_slab_store.py \
  --input $T/cell_wo1.zarr --output $T/cell_wo1_focus.zarr \
  --phase-channel Phase3D --halfwidth 2 --pixel-size 0.108
```

**Checks on 8 FOVs, running these three commands with `Phase3D` itself as the stand-in for a waveorder output:**
- The slab's `Phase3D`, `Nuclei` and the new channel were bitwise identical to the production `cell_focus.zarr` (the new channel to its `Phase3D`).
- The new channel's per-FOV statistics equalled the production `Phase3D` statistics.
- Timing at 8 FOVs: assemble 8 s, statistics 28 s, slab extraction 8 s.

**Notes:**
- **Several reconstructions:** add them all at once (`--add a.zarr:Phase3D:PhaseWO1 --add b.zarr:Phase3D:PhaseWO2`) and list them all in `--channel_names`. Each new channel adds roughly 21 GB full-Z (for all 500 FOVs) and about 2.5 GB to the slab.
- **Smoke test first:** `--limit 8` copies only 8 FOVs.
- **Not tested:** a real waveorder output. The checks used `Phase3D` itself as the stand-in.

### 2.3 Build test stores with the new channel

Predict reads only the source channel, and the eval reads only the prediction and the registered ground truth. So a test-side store needs only the new channel.

```sh
L=/hpc/projects/virtual_staining/training/dynacell_lite
uv run python applications/dynacell/tools/assemble_channel_store.py \
  --base $L/ipsc/dataset_v4/test_cropped/cell.zarr \
  --add <waveorder_ipsc_lite_output.zarr>:Phase3D:PhaseWO1 \
  --output $L/ipsc/dataset_v4/test_cropped/cell_wo1.zarr
uv run viscy preprocess --data_path $L/ipsc/dataset_v4/test_cropped/cell_wo1.zarr --channel_names '["PhaseWO1"]' --num_workers 8
# A549 (out of domain): repeat for dual_nucl_memb_{mock,DENV,ZIKV}.zarr -> dual_nucl_memb_{mock,DENV,ZIKV}_wo1.zarr
```

**Checks:** running these two commands with `Phase3D` as the stand-in (all 50 FOVs; assemble 10 s, statistics 28 s), then predicting from the new store, gave predictions bitwise identical to predicting from the registered store (max abs diff 0). Its recomputed statistics also equalled the registered ones.

## 3. Train

### 3.1 Write a variant config

Make one config per reconstruction. Put it at `applications/dynacell/configs/benchmarks/virtual_staining/nucleus/fnet2d_vscyto2daug_wo1/ipsc_confocal/train.yml`.

The `fnet2d_vscyto2daug_` name prefix matters. The test suite gives the speed flags to every model whose name starts with it, and a config placed here is checked by the repository tests.

```yaml
# Per-reconstruction variant of nucleus/fnet2d_vscyto2daug/ipsc_confocal: same recipe,
# source channel PhaseWO1 read from the new 5-plane store.
base:
  - ../../fnet2d_vscyto2daug/ipsc_confocal/train.yml

benchmark:
  model_name: fnet2d_vscyto2daug_wo1
  experiment_id: nucleus__ipsc_confocal__fnet2d_vscyto2daug_wo1

data:
  init_args:
    source_channel: PhaseWO1
    data_path: /hpc/projects/virtual_staining/training/dynacell/ipsc/dataset_v4/train/cell_wo1_focus.zarr
    normalizations:
      - class_path: viscy_transforms.NormalizeSampled
        init_args: {keys: [PhaseWO1], level: fov_statistics, subtrahend: mean, divisor: std}
      - class_path: viscy_transforms.NormalizeSampled
        init_args: {keys: [Nuclei], level: fov_statistics, subtrahend: mean, divisor: std}
    augmentations:
      - class_path: viscy_transforms.RandWeightedCropd
        init_args: {keys: [PhaseWO1, Nuclei], w_key: Nuclei, spatial_size: [1, 600, 600], num_samples: 4}
    val_augmentations:
      - class_path: viscy_transforms.CenterSpatialCropd
        init_args: {keys: [PhaseWO1, Nuclei], roi_size: [1, 384, 384]}

trainer:
  logger:
    init_args:
      name: FNet2D_VSAug_iPSC_NUCL_wo1
      save_dir: /hpc/projects/comp.micro/virtual_staining/models/dynacell/ipsc/nucleus/fnet2d_vscyto2daug_wo1
  callbacks:
    - class_path: lightning.pytorch.callbacks.LearningRateMonitor
      init_args: {logging_interval: step}
    - class_path: lightning.pytorch.callbacks.ModelCheckpoint
      init_args:
        monitor: loss/validate
        filename: "epoch={epoch}-step={step}-loss={loss/validate:.3f}"
        auto_insert_metric_name: false
        every_n_epochs: 1
        save_top_k: 4
        save_last: true
        dirpath: /hpc/projects/comp.micro/virtual_staining/models/dynacell/ipsc/nucleus/fnet2d_vscyto2daug_wo1/checkpoints

launcher:
  job_name: FNet2DVSAug_wo1
  run_root: /hpc/projects/comp.micro/virtual_staining/models/dynacell/ipsc/nucleus/fnet2d_vscyto2daug_wo1
```

**Why a copy, not `--override`:**
- Lists are replaced whole. Overriding `normalizations` with one entry silently drops the `Nuclei` one.
- The checkpoint folder sits inside the callbacks list and does not follow `run_root`. Variants launched with overrides alone would share one checkpoint folder.

**What to change per variant:** the tag (`wo1`) everywhere, `source_channel`, `data_path` and the channel name in the three lists. The recipe (precision, speed flags, steps, augmentations) comes from the base config.

**Checks:** this config composed, with bf16-mixed and both speed flags in effect, and the repository's config tests passed with it in place. The 40-step test fit used a smoke copy of it (TensorBoard logger, 8 FOVs); it trained and wrote checkpoints.

### 3.2 Launch

```sh
LEAF=applications/dynacell/configs/benchmarks/virtual_staining/nucleus/fnet2d_vscyto2daug_wo1/ipsc_confocal/train.yml
uv run python applications/dynacell/tools/submit_benchmark_job.py $LEAF --print-resolved-config   # preview, writes nothing
uv run python applications/dynacell/tools/submit_benchmark_job.py $LEAF --dry-run                 # stages resolved YAML + sbatch, no submit
uv run python applications/dynacell/tools/submit_benchmark_job.py $LEAF                           # submit
uv run python applications/dynacell/tools/submit_benchmark_job.py $LEAF --resume                  # continue a killed fit from its last checkpoint
```

- **Check before submitting:** in `--print-resolved-config`, look for `precision: bf16-mixed`, `benchmark: true`, `channels_last: true`, your `source_channel` and your `dirpath`.
- **Job resources:** 1 GPU of any type on the `gpu` partition, 32 CPUs, 128 GB, 20-day wall limit.
- **Following the run:** progress is in wandb, project `dynacell`, group = `trainer.logger.init_args.name`.
- **Is it done?** A fit is done when its last checkpoint reaches 200k steps. wandb's `finished` state alone also appears for killed runs.
- **Throughput:** the resolved config runs the training job at the speeds in §0. Alex submitted the baseline this way; submitting a variant this way is **(not run)**.

## 4. Predict on DynaCell-lite

### 4.1 Variant predict configs

Make one per test set, next to the train config: `nucleus/fnet2d_vscyto2daug_wo1/ipsc_confocal/predict__ipsc_confocal_lite.yml`. A new channel is not in the dataset manifest, so set `dataset_ref: null` and give the data fields inline.

```yaml
# Lite 2D predict of the PhaseWO1 variant on the iPSC lite test set.
base:
  - ../../../_internal/shared/model/predict_sets/ipsc_confocal_lite.yml
  - ../../../_internal/shared/model/targets/nucleus.yml
  - ../../../_internal/shared/model/model_overlays/fnet2d_predict.yml
  - ../../../_internal/shared/model/launcher_profiles/mode_predict.yml
  - ../../../_internal/shared/model/launcher_profiles/hardware_predict_any_gpu.yml
  - ../../../_internal/shared/model/launcher_profiles/runtime_shared.yml

benchmark:
  task: virtual_staining
  organelle: nucleus
  trained_on: ipsc_confocal
  predict_set: ipsc_confocal_lite
  model_name: fnet2d_vscyto2daug_wo1
  experiment_id: nucleus__ipsc_confocal__fnet2d_vscyto2daug_wo1__ipsc_confocal_lite
  dataset_ref: null

model:
  init_args:
    ckpt_path: /hpc/projects/comp.micro/virtual_staining/models/dynacell/ipsc/nucleus/fnet2d_vscyto2daug_wo1/checkpoints/last.ckpt

data:
  init_args:
    data_path: /hpc/projects/virtual_staining/training/dynacell_lite/ipsc/dataset_v4/test_cropped/cell_wo1.zarr
    source_channel: PhaseWO1
    target_channel: Nuclei
    predict_z_planes: /hpc/projects/virtual_staining/training/dynacell_lite/focus2d_planes/nucleus/ipsc__h1.json
    normalizations:
      - class_path: viscy_transforms.NormalizeSampled
        init_args: {keys: [PhaseWO1], level: fov_statistics, subtrahend: mean, divisor: std}
    augmentations: []

trainer:
  callbacks:
    - class_path: viscy_utils.callbacks.prediction_writer.HCSPredictionWriter
      init_args:
        output_store: /hpc/projects/virtual_staining/training/dynacell_lite/nucleus/fnet2d_vscyto2daug_wo1/ipsc/ipsc/prediction.zarr

launcher:
  job_name: FNet2DVSAug_wo1_PRED_NUCL_LITE
  run_root: /hpc/projects/virtual_staining/training/dynacell_lite/nucleus/fnet2d_vscyto2daug_wo1/ipsc/ipsc
```

**A549 lite (out of domain):** copy the config and change three things.
- `predict_sets/a549_mantis_h2b_<c>_lite.yml` in `base:`.
- `predict_set` and `data_path` to `.../dynacell_lite/a549/mantis/test/dual_nucl_memb_<C>_wo1.zarr`.
- `predict_z_planes` to `.../focus2d_planes/nucleus/a549__<c>__h1.json`, with the output under `.../ipsc/a549__<c>/`.

The baseline's A549 lite predict configs show the layout. An A549 predict from a new channel is **(not run)**; the A549 path itself was run with the baseline `Phase3D` configs.

### 4.2 Run the predict

```sh
P=applications/dynacell/configs/benchmarks/virtual_staining/nucleus/fnet2d_vscyto2daug_wo1/ipsc_confocal/predict__ipsc_confocal_lite.yml
# a) through sbatch
uv run python applications/dynacell/tools/submit_benchmark_job.py $P --ckpt best --override launcher.sbatch.constraint=a40 --dry-run
uv run python applications/dynacell/tools/submit_benchmark_job.py $P --ckpt best --override launcher.sbatch.constraint=a40
# b) directly, inside a GPU allocation (what was tested; each predict takes ~1 min)
mkdir -p ~/dynacell-resolved
uv run python applications/dynacell/tools/submit_benchmark_job.py $P --ckpt best --print-resolved-config > ~/dynacell-resolved/wo1-pred-ipsc.yml
uv run dynacell predict -c ~/dynacell-resolved/wo1-pred-ipsc.yml
```

- **GPU type:** `--override launcher.sbatch.constraint=a40` pins the GPU type (checked with `--print-script`). Use the same type for every predict you compare (§6); route (b) does this by running everything in one allocation.
- **Checkpoint:** `--ckpt best` replaces the pinned `last.ckpt` with the checkpoint that had the best validation loss. `--ckpt last` and `--ckpt <path>` also work.
- **Other settings:** predict runs at fp32 and needs neither bf16 nor channels-last.
- **Planes predicted:** only the 3 in-focus planes per FOV and timepoint are predicted. The others stay 0 in the store.
- **Re-predicting into the same store** needs `--overwrite`.
- **Timing:** 36 s for iPSC lite (50 FOVs) and 54–68 s per A549 lite condition. Route (a) is **(not run)**.

**Baseline predicts.** The two baseline models already have lite predict configs. Run them once their fits finish, which is expected by about 2026-11-19:

```sh
for m in fnet2d_vscyto2daug fnet2d_vscyto2daug_seed1; do
  for t in ipsc_confocal a549_mantis_h2b_mock a549_mantis_h2b_denv a549_mantis_h2b_zikv; do
    uv run python applications/dynacell/tools/submit_benchmark_job.py \
      applications/dynacell/configs/benchmarks/virtual_staining/nucleus/$m/ipsc_confocal/predict__${t}_lite.yml \
      --ckpt best --override launcher.sbatch.constraint=a40
  done
done
```

## 5. Score (fast tier)

The fast tier computes:
- **pixel metrics:** PCC, SSIM, NRMSE, PSNR, their scale-invariant versions, spectral PCC and `FRC_Resolution`;
- **segmentation metrics** on the nucleus masks: Dice and IoU;
- **instance AP.**

It skips the deep-feature and CP metrics. Run it inside a GPU allocation with at least 40 GB of GPU memory:

```sh
R=/hpc/projects/virtual_staining/training/dynacell_lite
M=fnet2d_vscyto2daug_wo1
uv run --frozen --package dynacell --extra eval --extra eval_gpu dynacell evaluate \
  target=nucleus +benchmark.dataset_ref.dataset=aics-hipsc-lite \
  io.pred_path=$R/nucleus/$M/ipsc/ipsc/prediction.zarr \
  io.pred_cache_dir=$R/ipsc/eval_cache_pred/nucleus/$M/ipsc/ipsc \
  save.save_dir=$R/eval_focus2d_h1/nucleus/$M/ipsc/ipsc \
  focus2d.halfwidth=1 segmentation.backend=cpdino compute_instance_ap=true \
  compute_feature_metrics=false compute_microssim=false force_recompute.final_metrics=true \
  force_recompute.pred_masks=true force_recompute.pred_instances=true
```

**A549 lite:** replace the first line's dataset settings with `target_name=nucleus +benchmark.dataset_ref.dataset=a549-mantis-h2b-<c>-lite +benchmark.dataset_ref.target=h2b`. Then point the three paths at `.../ipsc/a549__<c>`.

**Outputs** in `save.save_dir`:
- `pixel_metrics.csv` (one row per FOV and timepoint);
- `mask_metrics.csv`;
- `metrics_provenance.json`;
- plots and the segmentation masks.

**Timing:** about 2.5 min per test set; 156 s for iPSC lite, 137–141 s per A549 condition.

**Checks:** scoring the existing `fnet2d` checkpoint this way reproduced the full-benchmark 2D eval on the same FOVs.
- Pixel metrics matched to ≤5e-4 per FOV.
- Dice matched to ≤1e-3.
- Mean mAP matched to ≤0.003.

The command was run with `uv run --no-sync` in an already-synced environment. The `--frozen ... --extra` form shown is the documented one.

**Why the `force_recompute.pred_*` flags:** prediction caches are not invalidated when you re-predict into the same store. Without these flags, a second iteration would be scored on stale masks. Alternatively, give every iteration its own store and cache folder, as the variant tag does.

## 6. Read the results

- **Compare against both baseline seeds.** `fnet2d_vscyto2daug` and `fnet2d_vscyto2daug_seed1` differ only in the random seed. A reconstruction's effect is credible only if it is larger than the gap between them on the same test set.
- **Keep one GPU type for predict and eval** across everything you compare. Different GPU types change predictions by up to about 3e-3 and the segmentation masks by up to about 0.01 mean Dice.
- **Lite is validated for model rankings,** not for small differences. It has half the iPSC FOVs and half the A549 timepoints, so the standard error is about 1.4× the full benchmark's. To confirm a small effect, re-score on the full 2D benchmark: full dataset names, `.../dynacell/focus2d_planes/...` plane files.
- **A549 is out of domain for an iPSC-trained model.**
  - The old `fnet2d` (64 × 64 patches, flips only) fails there completely, with PCC < 0.1 and Dice < 0.01.
  - The VSCyto2D augmentations are expected to help, based on the 3D result. The baseline's A549 numbers will show it.
- **Useful columns:**
  - `PCC` and `SI_SSIM` for intensity agreement;
  - `FRC_Resolution` (µm, lower is better) for resolved detail;
  - `Dice` and `mAP` in `mask_metrics.csv` for nucleus shape and instances.

## 7. A new dataset or a new target channel (not run end to end)

The steps above reuse the DynaCell test sets, whose plane files, lite subsets and ground-truth caches already exist. For a dataset of your own:

- **Training** needs no manifest. Point a variant config's `data_path`, `source_channel` and `target_channel` at your store, re-key the three channel lists as in §3.1, and run `viscy preprocess` for both channels. A 3D store needs a 5-plane slab from `extract_focus_slab_store.py` (pass `--pixel-size`); a store that is already single-plane (Z=1) should work directly, since the 2D configs read one plane per sample (not run).
- **Predict** on a 3D test store needs a plane file. `dynacell precompute-gt ... build.focus_planes=<file>` writes one and needs a ground-truth nucleus channel. Here it was run through a registered dataset (`+benchmark.dataset_ref...`); registering yours means a manifest under `DYNACELL_MANIFEST_ROOTS`, and passing `io.gt_path` directly instead is untested. Without a plane file, set `predict_z_planes: null` to predict every plane (slower, but the eval accepts it).
- **Eval** on your own stores: pass `io.gt_path`, `io.gt_channel_name`, `io.pred_path`, `io.pred_channel_name`, `pixel_metrics.spacing` and `save.save_dir` instead of `+benchmark.dataset_ref...`. Two measured constraints (2026-11-16, synthetic stores; log in Alex's `experiments/2026-11-16_2d-native-z1-pixel-metrics/RESULTS.md`):
  - single-plane (Z=1) stores must be scored with `focus2d.halfwidth=0` plus the cpdino 2D segmentation flags (`compute_instance_ap=true segmentation.backend=cpdino segmentation.dimension=2d segmentation.slice_selection=focus segmentation.focus_slab_halfwidth=0`); without focus2d the eval crashes in SSIM;
  - the eval always segments the target, and only nucleus and membrane have a 2D segmenter.
- **For PSNR and FRC alone** on 2D arrays, `dynacell.evaluation.metrics.compute_pixel_metrics(pred, gt, spacing, fsc_kwargs={"bin_delta": 5}, ...)` returns `PSNR`, `SI_PSNR` and `FRC_Resolution` with no segmentation.

## 8. Gotchas

- **Never run `--overwrite` on, or write into, a registered store** (`cell.zarr`, `cell_focus.zarr` or any lite test store). Benchmark evals and running fits read them. Always assemble a new, tagged store.
- **Missing statistics fail loudly:** predict and fit raise `KeyError`. Stale statistics fail silently, so re-run `viscy preprocess` after rewriting a channel.
- **Shared caches:** the plane files and the lite ground-truth caches (`.../dynacell_lite/*/eval_cache/.../focus2d_h1/`) are shared. Do not delete them.
- **Don't reuse full-benchmark A549 plane files on lite.** They have 10 timepoints against 5 in the lite stores, so predict raises.
- **Writer refusals:** the prediction writer refuses to write a slab predict into a store another run wrote. Remove that store, or use `--overwrite` on your own store.
- **Logger swap fails:** swapping in a TensorBoard logger through a config fails with `Option 'project' is not accepted`, because the base config's wandb key cannot be removed by a merge. Keep wandb.
- **Stray files:** `dynacell` writes `config.yaml`, `pipeline.log`, `precompute_cli.log` and `cufile.log` into the current directory.

## 9. Not covered yet

- Running waveorder, and assembling a real waveorder output. The tool was run with `Phase3D` itself standing in for the reconstruction.
- A new dataset or target end to end (§7).
- Submitting a variant fit or predict through sbatch. Only the baseline fits were submitted that way; predicts and evals were run directly on a GPU node.
- The full metric tier (deep features, CP features, KID). On lite it needs a CP reference that includes the lite datasets, which does not exist yet.
- A549-trained or joint 2D baselines.
