# 3D virtual staining on DynaCell: train, profile, predict and score

This tutorial is for engineers who want to profile and speed up training of the DynaCell 3D
virtual-staining models. It covers one loop:

1. set up a checkout and pick a model;
2. launch a 3D fit on one GPU or on 4 GPUs with DDP;
3. run a short, repeatable fit for profiling, and measure seconds per step;
4. predict full 3D volumes of the iPSC test set;
5. score the predictions with the fast metric tier, to check that a faster run still trains an
   equally good model.

**How the commands were checked (2026-11-23 and 2026-11-25):**
- every preview command (`--print-resolved-config`, `--print-script`, `--dry-run`, Hydra
  `--cfg job`) was run and exited 0;
- the data path of §3.1–3.2 was run against the public release: the manifest rewrite, config
  resolution to the downloaded archive, and the profiling recipe of §5 for 30 steps on one A40
  reading the training archive directly, with the A40 changes listed in §5;
- commands marked **(not run)** were not executed;
- commands ran as the checkout's virtualenv binaries (`.venv/bin/python`, `.venv/bin/dynacell`); the
  `uv run` forms shown are the equivalents.

The running example is nucleus, trained on iPSC. The data come from the public DynaCell release
on AWS (§3.1). The committed dataset manifests point at the lab's own storage, so §3.2 makes a
copy that points at your download. `<data-dir>` is where you put the download and `<run-root>` is
any directory you can write to, both given as absolute paths.

## 0. The pieces

| Piece | Path (repository-relative) |
|---|---|
| Branch | `dynacell-fnet2d-vscyto2daug` (VisCy) |
| Primary single-GPU fit | `applications/dynacell/configs/benchmarks/virtual_staining/nucleus/fnet3d_vscyto3daug/ipsc_confocal/train.yml` |
| Primary 4-GPU DDP fit | `.../nucleus/fcmae_vscyto3d_scratch/ipsc_confocal/train.yml` |
| Shared config fragments | `.../virtual_staining/_internal/shared/model/{train_sets,targets,data_overlays,model_overlays,launcher_profiles,predict_sets}/` |
| Base recipes (model, trainer, GPU topology) | `applications/dynacell/configs/recipes/{models,trainer,topology}/` |
| Launcher | `applications/dynacell/tools/submit_benchmark_job.py` |
| sbatch template it renders | `applications/dynacell/tools/sbatch_template.sbatch` |
| Config composition (`base:`) | `packages/viscy-utils/src/viscy_utils/compose.py` |
| Manifest lookup (`benchmark.dataset_ref`) | `applications/dynacell/src/dynacell/_compose_hook.py` |
| Lightning CLI and the callbacks it adds | `packages/viscy-utils/src/viscy_utils/cli.py` |
| Data module and dataset | `packages/viscy-data/src/viscy_data/hcs.py` (`HCSDataModule`), `.../sliding_window.py` (`SlidingWindowDataset`) |
| Joint-dataset data module | `packages/viscy-data/src/viscy_data/combined.py` (`BatchedConcatDataModule`) |
| Augmentations | `packages/viscy-transforms/src/viscy_transforms/` (`RandWeightedCropd`, `Batched*`) |
| Models (Lightning modules) | `applications/dynacell/src/dynacell/engine.py` (`DynacellUNet`, `DynacellFlowMatching`, `DynacellGAN`) |
| Dataset manifests | `applications/dynacell/src/dynacell/_manifests/{aics-hipsc,aics-hipsc-lite,a549-mantis-*}/manifest.yaml` |

### How a config is put together

A runnable config is a *leaf*. A leaf lists fragments under `base:`. The fragments are merged in
order, then the leaf's own keys are merged on top:

- **Dicts merge key by key.**
- **Lists are replaced whole.** A leaf that sets `normalizations`, `augmentations` or
  `trainer.callbacks` replaces the inherited list, not one entry of it.
- **A fragment can have its own `base:`**, so a leaf can name another leaf as its base (§4.1).
- **Data paths come from the manifest.** After merging, `benchmark.dataset_ref: {dataset, target}`
  is looked up in the manifest, and `data_path`, `source_channel` and `target_channel` are written
  into `data.init_args`: the train store for `fit`, the test store for `predict`. A leaf that also
  sets one of these three fields fails with a conflict error.
- **Two keys are reserved.** `launcher:` (sbatch settings) and `benchmark:` (metadata) are removed
  before Lightning sees the config.
- **`--override key.path=value` merges after all of the above**, with the same rules. It cannot
  address a list entry (`callbacks[0]`), and it cannot re-point `benchmark.dataset_ref`, because
  the manifest lookup has already happened.

## 1. The 3D model families

All values are from the resolved nucleus/iPSC leaves (`--print-resolved-config` and
`--print-script`).

| Code name (paper name) | Engine, architecture | Input patch (Z, Y, X) | Batch per GPU (crops per window) | Budget | Precision, speed flags | GPUs the launcher asks for |
|---|---|---|---|---|---|---|
| `fnet3d_paper` (FNet3D) | `DynacellUNet`, FNet3D | 32×64×64 | 48 (8) | 200k steps | fp32, benchmark + channels-last | 1, any type |
| `fnet3d_bigpatch` (ablation) | `DynacellUNet`, FNet3D | 32×384×384 | 8 (4) | 200k steps | bf16-mixed, benchmark + channels-last | 1 H200 |
| **`fnet3d_vscyto3daug`** (FNet3D-VSCyto3DAug) | `DynacellUNet`, FNet3D | 32×600×600 crop → affine → 32×384×384 | 8 (4) | 200k steps | bf16-mixed, benchmark + channels-last | 1 H100 or H200 |
| **`fcmae_vscyto3d_scratch`** (UNeXt2) | `DynacellUNet`, fcmae (ConvNeXt V2 U-Net) | 20×600×600 crop → affine → 15×384×384 | 32 per rank (4) | 200 epochs | 16-mixed, benchmark | 4 H100 or H200, DDP |
| `fcmae_vscyto3d_pretrained` (VSCyto3D) | as UNeXt2, encoder loaded from an FCMAE checkpoint | as UNeXt2 | 32 per rank (4) | 200 epochs | 16-mixed, benchmark | 4 H100 or H200, DDP |
| `unetvit3d` (UNetViT3D) | `DynacellUNet`, UNetViT3D | 13×624×624 crop → affine → 8×512×512 | 4 (2) | 20 epochs | bf16-mixed | 1 H200 |
| `celldiff` (CELL-Diff) | `DynacellFlowMatching` | as UNetViT3D | 4 (2) | 20 epochs | bf16-mixed | 1 H200 |
| `pix2pix3d_unetvit` (pix2pix3d) | `DynacellGAN`, UNetViT3D generator + 3D PatchGAN | as UNetViT3D | 4 per rank (2) | 40 epochs | bf16-mixed | 4 H200, DDP |

- **Which leaf trained the published pix2pix3d weights:** `train_4gpu_modernized.yml`, not
  `train.yml` (see `applications/dynacell/CLAUDE.md`).
- **Speed flags:** "benchmark" is `trainer.benchmark: true` (cuDNN autotuning). "channels-last" is
  `model.init_args.channels_last: true`. `DynacellUNet` also accepts
  `model.init_args.compile_model: true` (`torch.compile` of the backbone during fit); no committed
  leaf turns it on.
- **The published fits mostly ran without the speed flags.** The leaves turn them on now, but of
  the fits timed in Appendix A only one (FNet3D-VSCyto3DAug, membrane) trained with them.
- **Budgets:** the step budget in one fit is `trainer.max_steps` or `trainer.max_epochs`. For the
  nucleus/iPSC store (§3.3):
  - FNet3D-VSCyto3DAug runs 2,600 steps per epoch, so 200k steps are ~77 epochs;
  - VSCyto3D runs 312 steps per epoch on 4 GPUs.

  Both figures are worked out in §3.5. They match the step counts in the published checkpoint
  names (`epoch=28-step=75400`, `epoch=89-step=28080`).

**Recommended models for profiling:**
- **Single GPU: `fnet3d_vscyto3daug`.** A plain 3D U-Net with the full augmentation stack, bf16 and
  both speed flags. A step runs through every stage of §6: CPU cropping, GPU augmentations, model
  and optimizer.
- **Multi-GPU: `fcmae_vscyto3d_scratch`** (UNeXt2, 4-GPU DDP). Its pretrained sibling
  `fcmae_vscyto3d_pretrained` (VSCyto3D) loads an FCMAE encoder checkpoint from the lab's storage,
  and that checkpoint is not part of the public release. The two resolve to the same config apart
  from that file (checked by diffing the two resolved configs: only `ckpt_path`, `encoder_only`
  and the output names differ), so the scratch fit has the same per-step work.

## 2. Set up once

```sh
git clone https://github.com/Biohub/VisCy.git && cd VisCy
git checkout dynacell-fnet2d-vscyto2daug
uv sync --all-packages --all-extras --frozen
uv run wandb login                 # or: export WANDB_MODE=offline
```

- **Use your own checkout.** The rendered sbatch script runs `uv run` from the repository that holds
  the launcher you called.
- **Module loading is site-specific.** The sbatch template loads uv with `ml uv`. On a cluster
  without that module, put `uv` on `PATH` and edit that line in your copy (§4.2 lists the other
  site settings).
- **What you need:**
  - the data of §3.1 (about 92 GB for the nucleus/membrane training archive and 7.5 GB for its test
    archive) on a filesystem the GPU nodes can read;
  - write access to your own `<run-root>`;
  - host memory for the staged training copy (§3.6), about 101 GB for nucleus/iPSC.
- **Interactive GPU work** (predict, eval, short fits) runs inside an allocation, for example
  `srun -p <partition> --gres=gpu:1 -c 16 --mem=128G -t 4:00:00 --pty bash`.

## 3. Data

### 3.1 Download the iPSC data from AWS

The DynaCell release is a public S3 bucket, `s3://dynacell/v1/`, readable without an AWS account
(registry entry: <https://registry.opendata.aws/dynacell/>). The iPSC data used here are in
`data/aics-hipsc/`:

| Archive | Training split | Test split | Targets it serves |
|---|---|---|---|
| `cell.ozx` | 85 GiB (500 FOVs) | 7.0 GiB (100 FOVs) | nucleus, membrane |
| `SEC61B.ozx` | 83 GiB (423 FOVs) | 8.0 GiB | ER |
| `TOMM20.ozx` | 96 GiB (500 FOVs) | 7.9 GiB | mitochondria |

For the nucleus example you need only `cell.ozx`, both splits:

```sh
mkdir -p <data-dir>/aics-hipsc/train <data-dir>/aics-hipsc/test
aws s3 cp --no-sign-request s3://dynacell/v1/data/aics-hipsc/train/cell.ozx <data-dir>/aics-hipsc/train/
aws s3 cp --no-sign-request s3://dynacell/v1/data/aics-hipsc/test/cell.ozx  <data-dir>/aics-hipsc/test/
```

- **Without the AWS CLI**, each file is also served over HTTPS at
  `https://dynacell.s3.us-west-2.amazonaws.com/v1/<path>`, for example
  `.../v1/data/aics-hipsc/train/cell.ozx`.
- **Keep the layout** `<data-dir>/aics-hipsc/{train,test}/<name>.ozx`; §3.2 assumes it.
- **Browse the rest** with `aws s3 ls --no-sign-request s3://dynacell/v1/`. The bucket README
  lists every archive. The A549 data (`data/biohub-a549/`, 577 GiB) are not needed here.
- **Licence.** The iPSC data are a derivative of the Allen Institute hiPSC Single-cell Image
  Dataset and are redistributed under the Allen Institute Terms of Use, which permit research and
  noncommercial use (`data/aics-hipsc/README.md` in the bucket). The A549 data are CC-BY-4.0.

**What an `.ozx` file is.** It is a single-file OME-Zarr: a zip archive, stored without
compression, that holds the Zarr v3 store. The code opens it in place through iohub, so there
is no need to unpack it: the 30-step fit of §5 read the training archive directly, and the
evaluation resolves its ground truth to the test archive (§8). Checked against the lab's working
stores (2026-11-25):
- all three training archives and the `cell.ozx` test archive hold the same FOVs, and the planes
  compared (one FOV each) are bitwise identical;
- the normalization statistics match in every FOV. The lab copy of `cell.zarr` carries one extra
  statistic, `otsu_threshold`, which no 3D leaf reads.

### 3.2 Point the configs at your download

The configs never name a data path. They name a dataset and a target
(`benchmark.dataset_ref`), and the code looks up the paths in a dataset manifest
(`_manifests/aics-hipsc/manifest.yaml`). The committed manifest points at the lab's storage. Make
a copy that points at `<data-dir>`, then tell the code to search your copy first. Save this as
`rewrite_manifest.py` and run it from the repository root:

```python
import shutil
import sys
from pathlib import Path

import yaml

data = Path(sys.argv[1]).resolve()  # <data-dir>
src = Path("applications/dynacell/src/dynacell/_manifests/aics-hipsc")
dst = data / "manifests" / "aics-hipsc"
shutil.copytree(src, dst, dirs_exist_ok=True)
manifest = yaml.safe_load((dst / "manifest.yaml").read_text())
for name, target in manifest["targets"].items():
    stores = target["stores"]
    archive = Path(stores["train"]).stem + ".ozx"  # cell, SEC61B or TOMM20
    target["stores"] = {
        "train": str(data / "aics-hipsc" / "train" / archive),
        "test": str(data / "aics-hipsc" / "test" / archive),
        "gt_cache_dir": str(data / "eval_cache" / name),
    }
(dst / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
print(f"export DYNACELL_MANIFEST_ROOTS={data / 'manifests'}")
```

```sh
uv run python rewrite_manifest.py <data-dir>
export DYNACELL_MANIFEST_ROOTS=<data-dir>/manifests     # the line the script prints
```

- **`DYNACELL_MANIFEST_ROOTS`** is a list of directories (separated by `:`) searched before the
  committed manifests; the first `<root>/<dataset>/manifest.yaml` found wins
  (`dynacell/data/resolver.py`). Set it in every shell that renders a config (§4), predicts (§7)
  or evaluates (§8).
- **The rewrite drops `cell_segmentation`**, the lab's cleaned cell masks of the test set. They
  are not in the release. The evaluation requires them only with `compute_feature_metrics=true`
  or `compute_cell_similarity=true`, and the fast tier of §8 sets neither.
- **`gt_cache_dir`** is where the evaluation caches ground-truth masks. It starts empty; the
  first evaluation fills it (§8).
- **Check it:** `--print-resolved-config` (§4.2) should show
  `data_path: <data-dir>/aics-hipsc/train/cell.ozx`.

### 3.3 The training store

- **Format:** HCS OME-Zarr v0.5 (Zarr v3), inside the `.ozx` archive (§3.1).
- **Which store:** nucleus and membrane train on the full-Z `cell.ozx` (`targets.nucleus.stores.train`
  in the manifest). ER and mitochondria train on separate archives (`SEC61B.ozx`, `TOMM20.ozx`),
  whose target channel is `Structure`.
- **Shape:**
  - 500 FOVs, each `(T, C, Z, Y, X) = (1, 4, 44, 624, 924)`, float32;
  - chunks `(1, 1, 8, 512, 512)`, grouped into one shard per FOV and channel;
  - channels `Brightfield`, `Nuclei`, `Membrane`, `Phase3D`.
- **Source and target:** source `Phase3D`, target `Nuclei`.
- **Voxel spacing** (manifest): 0.290 µm in Z, 0.108 µm in Y and X.

### 3.4 Normalization statistics

- **Computing them:** `uv run viscy preprocess --data_path <store> --channel_names '["Phase3D"]'`
  writes, into every FOV's attributes under `normalization`, per-channel `dataset_statistics`,
  `fov_statistics` and `timepoint_statistics`.
- **Using them:** `NormalizeSampled` (`viscy_transforms/_normalize.py`) reads the level and the
  statistics a config names:
  - FNet3D: `fov_statistics` mean/std for both channels;
  - VSCyto3D, UNeXt2 and UNetViT3D: mean/std for `Phase3D` and median/IQR for `Nuclei`
    (`targets/nucleus.yml`).
- **The released archives already carry them** (§3.1). Do not run `preprocess` on them.

### 3.5 How a training batch is built

1. **Dataset item.** `SlidingWindowDataset` serves one window per index: one FOV and timepoint,
   `z_window_size` consecutive planes at full Y and X, for source and target. The dataset length is
   `Σ T × (Z − z_window_size + 1)` over the FOVs in the split.
2. **Split.** `HCSDataModule._setup_fit` shuffles the FOVs with `torch.randperm` (seeded by
   `seed_everything`) and keeps the first 80% (`split_ratio: 0.8`) for training.
3. **CPU transforms** run in the DataLoader workers, in this order: `NormalizeSampled`, then
   `RandWeightedCropd`. `RandWeightedCropd` takes `num_samples` crops per window, with crop
   centres drawn with probability proportional to the target intensity.
4. **Collation.** The training DataLoader batch is `batch_size / num_samples` windows. The collate
   function flattens the crops, so each step trains on `batch_size` patches. `batch_size` must be
   divisible by `num_samples`.
5. **GPU transforms.** `HCSDataModule.on_after_batch_transfer` applies `gpu_augmentations` on the
   device, after the host-to-device copy:
   - `BatchedRandAffined`: rotation, shear and scale, resampled on the GPU;
   - `BatchedCenterSpatialCropd`: crop to the model input;
   - `BatchedRandAdjustContrastd`, `BatchedRandScaleIntensityd`, `BatchedRandGaussianNoised`
     and `BatchedRandGaussianSmoothd`: intensity changes on the source only.
6. **Validation:**
   - FNet3D center-crops on the CPU (`val_augmentations`);
   - FCMAE moves whole 20-plane windows to the GPU and center-crops there
     (`val_gpu_augmentations`);
   - the validation DataLoader batch is `batch_size` windows; it is **not** divided by
     `num_samples`.

**Worked numbers for nucleus/iPSC** (400 training FOVs, T = 1, Z = 44):

| | Windows per FOV | Training windows | Windows per DataLoader batch | Steps per epoch |
|---|---|---|---|---|
| FNet3D-VSCyto3DAug (Z 32, batch 8, 4 crops) | 13 | 5,200 | 2 | 2,600 |
| VSCyto3D (Z 20, batch 32 per rank, 4 crops, 4 ranks) | 25 | 10,000 (2,500 per rank) | 8 | 312 |

**DDP:**
- **Per-rank batch.** `batch_size` is per rank: VSCyto3D trains on 4 × 32 = 128 patches per optimizer
  step.
- **Sharding.** Lightning adds a `DistributedSampler`, so the ranks see disjoint windows.
- **Changing the GPU count** changes the global batch and the steps per epoch.

**Joint (iPSC + A549) fits:**
- These leaves use `BatchedConcatDataModule`, whose `train_dataloader` does not divide by
  `num_samples`. A joint leaf therefore gets `batch_size × num_samples` patches per step.
- The repository root `CLAUDE.md` (*Joint vs single-set training batch semantics*) has the table.

### 3.6 Loading, caching and worker settings

These are the `HCSDataModule` arguments that set the I/O path (`hcs.py`):

| Argument | 3D leaves | Effect |
|---|---|---|
| `mmap_preload` | `true` (`train_sets/ipsc_confocal.yml`) | Before training, `prepare_data` copies every FOV's source and target channels into one `tensordict.MemoryMappedTensor`. It uses 16 reader threads and writes a `.done` marker. The training and validation datasets then read windows from that buffer instead of from Zarr. Fit and validate only; predict ignores it. |
| `scratch_dir` | `/dev/shm` | Where the buffer goes: `<scratch_dir>/<SLURM_JOB_ID>/<store>_<hash>/data.mmap`. Staging runs once per node (`prepare_data_per_node`), and all DDP ranks share it. |
| `num_workers` | 4 (FNet3D-VSCyto3DAug, FCMAE, UNetViT3D), 8 (`fnet3d_paper`) | DataLoader worker processes per rank. |
| `persistent_workers` | `true` | Workers survive across epochs. |
| `pin_memory` | `true` (default) | Pinned host buffers for the copy to the GPU. |
| `prefetch_factor` | unset (PyTorch default 2) | Batches prefetched per worker. |
| `min_nonzero_fraction` | 0 (off) | Optional resampling of near-empty patches. |
| `include_fov_names` / `exclude_fov_names` | unset | FOV filters, applied before the split and the staging. |

**Size of the staged copy for nucleus/iPSC** (DERIVED):
500 FOVs × 2 channels × 44 × 624 × 924 voxels × 4 bytes ≈ 101 GB (94.5 GiB).

**Memory accounting:** a tmpfs copy counts toward the job's memory limit. That is why the
FNet3D and 4-GPU leaves request 512 GB of host memory (UNetViT3D and CELL-Diff request 256 GB).

**Other notes:**
- `mmap_preload: false` reads every window from Zarr in the workers. Startup is faster, but the
  training data path is different.
- No other dataset cache exists.
- Predict reads each volume from Zarr once and cuts its windows in memory (`cache_volume`, on when
  `num_workers: 0`, as in every predict leaf).

## 4. Launch a fit

### 4.1 Make your own copy of the leaf

The committed leaves write logs, checkpoints and sbatch files into the lab's model directories,
which do not exist on your cluster. Make a copy whose base is the committed leaf, with your own
output directories:

`applications/dynacell/configs/benchmarks/virtual_staining/nucleus/fnet3d_vscyto3daug_hpc/ipsc_confocal/train.yml`:

```yaml
# Copy of nucleus/fnet3d_vscyto3daug/ipsc_confocal: same recipe, own output dirs.
base:
  - ../../fnet3d_vscyto3daug/ipsc_confocal/train.yml

benchmark:
  model_name: fnet3d_vscyto3daug_hpc
  experiment_id: nucleus__ipsc_confocal__fnet3d_vscyto3daug_hpc

trainer:
  logger:
    init_args:
      name: FNet3D_VSAug_iPSC_NUCL_hpc
      save_dir: <run-root>/fnet3d_vscyto3daug_hpc
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
        dirpath: <run-root>/fnet3d_vscyto3daug_hpc/checkpoints

launcher:
  job_name: FNet3DVSAug_hpc
  run_root: <run-root>/fnet3d_vscyto3daug_hpc
```

**Why a copy, not `--override launcher.run_root=...`:** the checkpoint folder sits inside the
`trainer.callbacks` list. An override cannot reach a list entry, so the checkpoints would still
go to the lab's folder.

**For UNeXt2 (4-GPU):** make `nucleus/fcmae_vscyto3d_scratch_hpc/ipsc_confocal/train.yml` the
same way:
- `base: [../../fcmae_vscyto3d_scratch/ipsc_confocal/train.yml]`;
- `save_top_k: 5`;
- your own names and `<run-root>` paths.

Both copies were composed against the public-data manifest of §3.2 and previewed
(`--print-resolved-config`, `--print-script` with the site overrides below); the FNet3D copy was
also staged with `--dry-run`.

### 4.2 Preview, then submit

```sh
LEAF=applications/dynacell/configs/benchmarks/virtual_staining/nucleus/fnet3d_vscyto3daug_hpc/ipsc_confocal/train.yml
uv run python applications/dynacell/tools/submit_benchmark_job.py $LEAF --print-resolved-config  # resolved Lightning config, writes nothing
uv run python applications/dynacell/tools/submit_benchmark_job.py $LEAF --print-script           # rendered sbatch, writes nothing
uv run python applications/dynacell/tools/submit_benchmark_job.py $LEAF --dry-run                # writes <run_root>/resolved/ and <run_root>/slurm/, no submit
uv run python applications/dynacell/tools/submit_benchmark_job.py $LEAF                          # submit (not run)
```

**Check in `--print-resolved-config`:**
- `data_path: <data-dir>/aics-hipsc/train/cell.ozx` (§3.2);
- `precision: bf16-mixed`, `benchmark: true` and `channels_last: true`;
- `max_steps: 200000`;
- your `dirpath` and `save_dir`.

The launcher resolves the data path when it renders the job and writes it into
`<run_root>/resolved/fit_*.yml`, so the job itself does not need `DYNACELL_MANIFEST_ROOTS`.

**What the rendered sbatch does:**
1. sets the `#SBATCH` lines from the `launcher.sbatch` block;
2. runs a short NCCL check, `tools/nccl_smoke_test.py`;
3. runs `srun --cpu-bind=none uv run python -m dynacell fit --config <run_root>/resolved/fit_<job>_<time>.yml`;
4. removes its per-job folders in the node-local temp directory and in `/dev/shm` on exit.

**Multi-GPU invariants:**
- The launcher refuses to render unless `trainer.devices == launcher.sbatch.ntasks_per_node` and
  `launcher.sbatch.gpus == nodes × devices`.
- The template uses `--ntasks-per-node`, never `--ntasks`: Lightning's `SLURMEnvironment` rejects
  `--ntasks`.
- To change the GPU count, override all three together, for example
  `--override trainer.devices=2 --override launcher.sbatch.ntasks_per_node=2 --override launcher.sbatch.gpus=2`
  (previewed).
- A different GPU count changes the global batch (§3.5) and, for the `WarmupCosine` schedule, the
  learning-rate curve, because the schedule length is `trainer.estimated_stepping_batches`.

**Other hardware settings** come from the `launcher_profiles/hardware_*.yml` the leaf composes.
Override any of them with `--override launcher.sbatch.<key>=<value>`, where the keys are
`partition`, `constraint`, `mem`, `time`, `cpus_per_task` and `exclude`.

**Site settings to change for your cluster.** The committed profiles and template are written
for the lab's SLURM cluster:

| Setting | Committed value | Change it with |
|---|---|---|
| Partition | `gpu` | `--override launcher.sbatch.partition=<partition>` |
| GPU type | `constraint: "h100\|h200"` (4-GPU), `"h200"` or `"h100\|h200"` (1-GPU); these are the lab's node feature names | `--override launcher.sbatch.constraint=<feature>`, or `=null` to drop the line |
| Hugging Face cache | `launcher.env.HF_HUB_CACHE`, a lab directory (`runtime_shared.yml`) | `--override launcher.env.HF_HUB_CACHE=<dir>` |
| uv | `ml uv` in `tools/sbatch_template.sbatch` | edit the line in your checkout |

Wall-time limits (`time`, 4 days for most fits) and memory (`mem`) are also sized for that
cluster; check them against your partition's limits. The override forms above were previewed
with `--print-script`.

### 4.3 Run directly inside an allocation

```sh
RUN=<run-root>/fnet3d-direct && mkdir -p $RUN
uv run python applications/dynacell/tools/submit_benchmark_job.py $LEAF --print-resolved-config > $RUN/fit.yml
cd $RUN && uv run --project <checkout> dynacell fit -c $RUN/fit.yml
```

- **Run it from `$RUN`.** `dynacell` writes `config.yaml` and other stray files into the current
  directory or into `trainer.default_root_dir`.
- **The 4-GPU leaves** need one process per GPU, which the sbatch path's `srun` provides. Running
  them directly under `srun --ntasks-per-node=4` inside a 4-GPU allocation is **(not run)**.

### 4.4 Resume, follow, and decide whether a fit finished

- **Resume:** `--resume` adds `--ckpt_path=<newest last*.ckpt by modification time>` to the fit
  command. `--resume-from <ckpt>` names the checkpoint explicitly. Previewed with
  `--print-script`; a real resume is **(not run)**.
- **Why the newest file, not `last.ckpt`:** Lightning writes `last-v1.ckpt`, `last-v2.ckpt`, ... on
  each resume, so `last.ckpt` alone can be an old state.
- **Following the run:** wandb project `dynacell`, run group = `trainer.logger.init_args.name`.
  wandb's `heartbeatAt` is the reliable sign that a run is alive.
- **Did it finish?** The run log prints ``Trainer.fit` stopped: `max_steps=200000` reached.``
  wandb's `finished` state does not prove this: it also appears after a clean SIGTERM. In `sacct`,
  `COMPLETED` with exit code `0:0` is the only unambiguous success; a `scancel` shows `CANCELLED+`.

## 5. A short, repeatable fit for profiling

Render the leaf with overrides that end the run early and turn off logging and checkpoints, then
run it inside an allocation:

```sh
PROF=<run-root>/profile-fnet3d && mkdir -p $PROF
uv run python applications/dynacell/tools/submit_benchmark_job.py $LEAF --print-resolved-config \
  --override trainer.max_steps=300 \
  --override trainer.logger=false \
  --override 'trainer.callbacks=[]' \
  --override trainer.enable_checkpointing=false \
  --override trainer.default_root_dir=$PROF \
  --override trainer.profiler=simple \
  > $PROF/fit.yml
cd $PROF && uv run --project <checkout> dynacell fit -c $PROF/fit.yml
```

**What was run (2026-11-25):** this recipe for 30 steps on one A40 (48 GB), reading the public
`cell.ozx` through the manifest of §3.2, with two A40-only changes:
- `--override data.init_args.batch_size=4`: the leaf's batch 8 is sized for 80 GB cards (leaf
  comment), and batch 8 was not tried on the A40;
- `--override 'data.init_args.include_fov_names=[4/10615/1304,4/10625/1347,...]'` (the first 10
  FOVs), so the staged copy was 1.9 GB instead of ~101 GB. Staging (`mmap_preload: true`) ran.

The run ended with ``Trainer.fit` stopped: `max_steps=30` reached.`` and exit code 0.

For profiling, keep the leaf's batch, all FOVs and `mmap_preload: true` on the GPUs the launcher
asks for. A smaller FOV list is a quick way to check the pipeline end to end, but it changes the
epoch length and the staging time, so do not time with it.

**Profilers:** `trainer.profiler` also accepts `advanced` (cProfile) and `pytorch` (the torch
profiler, with traces). The SimpleProfiler prints a per-action table to stdout at the end of the fit; the check run printed these rows:
- `[_TrainingEpochLoop].train_dataloader_next`: waiting for data;
- `run_training_batch`: forward, backward and optimizer;
- `[LightningDataModule]HCSDataModule.on_after_batch_transfer`: the GPU augmentations;
- `[LightningDataModule]HCSDataModule.prepare_data`: the staging.

CUDA calls are asynchronous, so the split of GPU time between `on_after_batch_transfer` and
`run_training_batch` is approximate. Use `trainer.profiler=pytorch` for a GPU timeline.

**Through sbatch:** pass the same overrides to `submit_benchmark_job.py` without
`--print-resolved-config`, together with the site overrides of §4.2. Add
`--override 'launcher.sbatch.time="01:00:00"'` (previewed). **(not run)**

**What a short run changes:**
- **No validation pass.** Validation runs at the end of each epoch (every 2,600 steps here), so a
  run shorter than one epoch has none.
- **Startup cost.** With `mmap_preload: true`, staging (§3.6) happens before step 1. Time it
  separately from the steps.
- **Warm-up.** cuDNN autotuning (`benchmark: true`) makes the first steps slower and briefly raises
  peak GPU memory.
- **Learning rate.** For `WarmupCosine` models (FCMAE, UNetViT3D, CELL-Diff, pix2pix3d), setting
  `max_steps` shortens the learning-rate schedule. The work per step is unchanged, but loss curves
  do not match a full run. FNet3D uses a constant learning rate.

### 5.1 Measure steady seconds per step

- **Long runs: use wandb.** Fit `_runtime` (seconds since the run started) against
  `trainer/global_step` by least squares. Drop the warm-up, and stay inside one epoch: a fit across
  epoch boundaries also counts validation, image logging and checkpoint writes. That number is
  useful too, but label it "wall per step". Do not divide total wall time by the number of steps.

```python
import numpy as np
import wandb

run = wandb.Api().run("<entity>/dynacell/<run-id>")
rows = np.array(
    [(h["trainer/global_step"], h["_runtime"]) for h in run.scan_history(keys=["trainer/global_step", "_runtime"])],
    dtype=float,
)
window = (rows[:, 0] >= 200) & (rows[:, 0] < 2500)   # inside epoch 0, after warm-up
s_per_step, _ = np.polyfit(rows[window, 0], rows[window, 1], 1)
print(f"{s_per_step:.3f} s/step")
```

  This snippet was run as printed against an existing FNet3D-VSCyto3DAug run (trained before the
  speed flags were on) and returned a slope.

- **Short runs without wandb:** use the profiler table above. Its means include the warm-up steps,
  so run enough steps (a few hundred) for them to wash out.

### 5.2 What "reproducible" means here

- **Fixed by `seed_everything`:** the model initialization, the train/validation split and the FOV
  order. FNet3D uses seed 0; the other families use 42.
- **Not fixed:** the crop centres. MONAI's `RandWeightedCropd` draws them from a random state seeded
  from OS entropy, and nothing in the code calls `monai.utils.set_determinism`. Two runs with the
  same seed therefore see different batches.
- **Also not fixed:** cuDNN autotuning (`benchmark: true`) can pick different kernels per run.
- **Consequence:** compare an accelerated run with the reference by predicted-image metrics (§8),
  not by loss curves or weights.

## 6. Where the time goes

No per-stage breakdown has been measured; Appendix A gives whole-step rates per model. This is
the path a step takes, with the place to look for each stage:

| Stage | When | Code | Settings |
|---|---|---|---|
| Config compose, manifest lookup | startup | `compose.py`, `_compose_hook.py` | none |
| NCCL check | startup (sbatch only) | `tools/nccl_smoke_test.py` | none |
| Staging the store to `/dev/shm` | startup, once per node | `HCSDataModule.prepare_data` | `mmap_preload`, `scratch_dir` |
| Worker start, sanity validation (2 batches) | startup | Lightning | `num_workers`, `persistent_workers`, `trainer.num_sanity_val_steps` |
| Window read: clone from the mmap view, cast to float32 | every step, CPU workers | `SlidingWindowDataset._read_img_window` | `num_workers`, `prefetch_factor` |
| `NormalizeSampled`, `RandWeightedCropd` (600×600 crops) | every step, CPU workers | `viscy_transforms` | `augmentations` list |
| Collate, pin, copy to the GPU | every step | `_collate_samples`, DataLoader | `pin_memory` |
| Affine resample, center crop, intensity augmentations | every step, GPU | `HCSDataModule.on_after_batch_transfer` | `gpu_augmentations` list |
| Forward, loss, backward, AdamW | every step, GPU | `DynacellUNet.training_step` | `precision`, `benchmark`, `channels_last`, `compile_model` |
| Loss logging (`sync_dist=True`: one all-reduce per step under DDP) | every step | `training_step` | `trainer.log_every_n_steps` sets only the logger write rate |
| Gradient all-reduce | every step, DDP | `DDPStrategy` (`find_unused_parameters: true` for FCMAE) | `trainer.strategy` |
| AMP scale and loss checks | every step | `viscy_utils.callbacks.OptimizerHealthCheck`, added to every fit by `cli.py` | none |
| Validation pass | every epoch | `validation_step` | `trainer.limit_val_batches`, `check_val_every_n_epoch` |
| Sample images to wandb (first 8 batches) | every epoch | `_log_samples` | `model.init_args.log_batches_per_epoch` |
| Checkpoints: top-k by `loss/validate`, plus `latest-*` and `last.ckpt` | every epoch | `ModelCheckpoint`, extra callback added by `cli.py` | `every_n_epochs`, `save_top_k` |

**Measured numbers already in the configs:**
- `model_overlays/fnet3d_paper_fit.yml` records 0.493 s/step with `benchmark` and `channels_last`
  against 0.918 s/step without them, on one H200, with training loss within the run-to-run spread.
- The `fnet3d_vscyto3daug` leaf records about 41 GB of GPU memory for a 384×384×32 patch at
  bf16-mixed, measured on the `fnet3d_bigpatch` arm.
- `hardware_4gpu.yml` records a peak of about 110 GB for single-store staging.

## 7. Predict full 3D volumes of the iPSC test set

The test split of §3.1 (`test/cell.ozx`) holds the 100 held-out iPSC FOVs, each
`(1, 4, 40, 512, 512)`. The committed leaves also predict *DynaCell-lite*, a 50-FOV subset that
the lab uses for fast comparisons, but its stores are not in the public release; use the full
test set.

**The committed predict leaf** for this model,
`nucleus/fnet3d_vscyto3daug/ipsc_confocal/predict__ipsc_confocal.yml`, points its checkpoint and
output at the lab's storage. Write a copy next to your fit copy, with your own paths:

`.../nucleus/fnet3d_vscyto3daug_hpc/ipsc_confocal/predict__ipsc_confocal.yml`:

```yaml
# 3D predict of the HPC-team FNet3D-VSCyto3DAug fit on the iPSC test set.
base:
  - ../../../_internal/shared/model/predict_sets/ipsc_confocal.yml
  - ../../../_internal/shared/model/targets/nucleus.yml
  - ../../../_internal/shared/model/model_overlays/fnet3d_paper_predict.yml
  - ../../../_internal/shared/model/launcher_profiles/mode_predict.yml
  - ../../../_internal/shared/model/launcher_profiles/hardware_predict_any_gpu.yml
  - ../../../_internal/shared/model/launcher_profiles/runtime_shared.yml

benchmark:
  task: virtual_staining
  organelle: nucleus
  trained_on: ipsc_confocal
  predict_set: ipsc_confocal
  model_name: fnet3d_vscyto3daug_hpc
  experiment_id: nucleus__ipsc_confocal__fnet3d_vscyto3daug_hpc__ipsc_confocal

model:
  init_args:
    ckpt_path: <run-root>/fnet3d_vscyto3daug_hpc/checkpoints/last.ckpt   # --ckpt best replaces it

data:
  init_args:
    normalizations:
      - class_path: viscy_transforms.NormalizeSampled
        init_args: {keys: [Phase3D], level: fov_statistics, subtrahend: mean, divisor: std}
    augmentations: []

trainer:
  callbacks:
    - class_path: viscy_utils.callbacks.prediction_writer.HCSPredictionWriter
      init_args:
        output_store: <run-root>/predict/nucleus/fnet3d_vscyto3daug_hpc/ipsc/ipsc/prediction.zarr

launcher:
  job_name: FNet3DVSAug_hpc_PRED_NUCL
  run_root: <run-root>/predict/nucleus/fnet3d_vscyto3daug_hpc/ipsc/ipsc
```

With `DYNACELL_MANIFEST_ROOTS` set (§3.2), the predict set resolves to
`<data-dir>/aics-hipsc/test/cell.ozx`.

**Run it:**

```sh
P=applications/dynacell/configs/benchmarks/virtual_staining/nucleus/fnet3d_vscyto3daug_hpc/ipsc_confocal/predict__ipsc_confocal.yml
uv run python applications/dynacell/tools/submit_benchmark_job.py $P --print-resolved-config     # previewed
uv run python applications/dynacell/tools/submit_benchmark_job.py $P --ckpt best \
  --override launcher.sbatch.partition=<partition> --override launcher.sbatch.constraint=<gpu-type>   # (not run)
```

**What predict does:**
- **Settings:** fp32, batch 1, `num_workers: 0`, every Z plane.
- **FNet3D** (`predict_method: full_image`) runs each 32-plane window over the whole 512×512 field.
  Windows step one plane at a time, and the writer blends the overlapping planes.
- **VSCyto3D and UNeXt2** do the same with 15-plane windows.
- **UNetViT3D and pix2pix3d** tile each window into 8×512×512 patches (`sliding_window`, overlap
  `[4, 256, 256]`). pix2pix3d also averages 4 shifted tilings (`predict_phase_shifts`).
- **CELL-Diff** samples iteratively, which makes it by far the slowest to predict.

**Flags:**
- **`--ckpt best`:** reads `best_model_path` from the newest `last*.ckpt`, the checkpoint with the
  lowest `loss/validate`. `--ckpt last` and `--ckpt <path>` also work.
- **Pin the GPU type** with `launcher.sbatch.constraint`, and use the same type for every predict
  you compare (§8).
- **`--overwrite`** re-predicts into an existing store. **`--resume-predict`** continues a store
  that a time limit cut off.

## 8. Score with the fast tier, and compare with the reference

The fast tier computes:
- **pixel metrics** over the full 3D volume: PCC, SSIM, NRMSE, PSNR, their scale-invariant
  versions, spectral PCC, and XY and Z FSC resolution;
- **segmentation metrics** (Dice, IoU) and **instance AP** on cpdino nucleus masks of the in-focus
  plane of each volume (`segmentation.dimension: 2d`, `slice_selection: focus`, the defaults).

It skips the deep-feature and CellProfiler-style feature metrics. Run it inside an allocation with
at least 40 GB of GPU memory, with `DYNACELL_MANIFEST_ROOTS` set (§3.2):

```sh
OUT=<run-root>/predict/nucleus/fnet3d_vscyto3daug_hpc/ipsc/ipsc
uv run --frozen --package dynacell --extra eval --extra eval_gpu dynacell evaluate \
  target=nucleus +benchmark.dataset_ref.dataset=aics-hipsc \
  io.pred_path=$OUT/prediction.zarr \
  io.pred_cache_dir=$OUT/eval_cache_pred \
  save.save_dir=$OUT/eval \
  segmentation.backend=cpdino compute_instance_ap=true \
  compute_feature_metrics=false compute_microssim=false force_recompute.final_metrics=true \
  force_recompute.pred_masks=true force_recompute.pred_instances=true
```

- **Checked (2026-11-25):** the command composes (Hydra `--cfg job`, exit 0), and with the
  manifest of §3.2 the dataset lookup supplies the ground truth `<data-dir>/aics-hipsc/test/cell.ozx`,
  the cache folder `<data-dir>/eval_cache/nucleus`, the channels `Nuclei_prediction` / `Nuclei`
  and the spacing `[0.29, 0.108, 0.108]`.
- **Not run:** the evaluation itself.
- **No `focus2d` setting:** that is the 2D benchmark. Leaving it unset scores pixel metrics on the
  full volume.
- **Model downloads.** `cellpose` downloads the cpdino (Cellpose-DINO) weights on first use from
  its public Hugging Face repository into `~/.cellpose/models`, or `$CELLPOSE_LOCAL_MODELS_PATH`
  (read from `cellpose/models.py`; not tried from outside the lab). Compute nodes without internet
  access need that file copied in beforehand.
- **Caches:**
  - ground-truth masks are cached in the manifest's `gt_cache_dir`. Your copy starts empty, so the
    first evaluation segments the ground truth and fills it (`io.require_complete_cache: false`);
    later evaluations reuse it;
  - prediction caches go to your `io.pred_cache_dir`;
  - the `force_recompute.pred_*` flags stop a re-predicted store from being scored with stale masks.
- **Outputs** in `save.save_dir`:
  - `pixel_metrics.csv`, one row per FOV and timepoint;
  - `mask_metrics.csv`;
  - `metrics_provenance.json`.

**To check an accelerated run:**
1. Train it from the same leaf, changing only what you accelerate.
2. Predict both the accelerated run and a reference fit with `--ckpt best`, on the same GPU type.
3. Score both with the command above.
4. Compare `PCC`, `SI_SSIM`, `Spectral_PCC`, `XY_FSC_Resolution` and `Z_FSC_Resolution` in
   `pixel_metrics.csv`, and `Dice` and `mAP` in `mask_metrics.csv` (column names read from an
   existing 3D evaluation).
5. Judge any difference against the spread between two reference fits that differ only in the seed.
   Crop sampling is random (§5.2), so a single pair of runs never matches exactly.

**Notes:**
- **GPU type matters.** cpdino segments in bf16, so instance masks differ by GPU type: per-row Dice
  medians by ≤ 3e-4, single frames by up to ~0.1.
- **Published reference checkpoints.** The release holds trained iPSC nucleus checkpoints of the
  FNet3D (paper), UNeXt2 and VSCyto3D families, each with its training config, under
  `s3://dynacell/v1/models/ipsc/nucleus/{fnet3d,unext2,vscyto3d}/`. No FNet3D-VSCyto3DAug
  checkpoint is published, so a reference for that model has to be trained.
- **Out of domain (A549).** The lab also scores iPSC-trained models on A549 cells. The A549 test
  archives are in the release (`data/biohub-a549/test/H2B_<condition>.ozx`), but the A549
  manifests have a different store layout, and rewriting them for the public archives was not
  checked.

## 9. Gotchas

- **Checkpoints: `last.ckpt` is the newest weights; the best are in the top-k files.**
  - Lightning's `ModelCheckpoint` writes `last.ckpt` only in an epoch that also enters the top-k.
    Once validation loss stops improving, a monitored callback alone stops updating `last.ckpt`.
  - `viscy_utils/cli.py` therefore adds a second, unmonitored checkpoint callback to every fit. It
    owns `last.ckpt` and writes `latest-epoch=*` on the same cadence.
  - Use `--ckpt best` for predicts.
  - Do not read training progress from the highest `epoch=` in the folder: those are the best
    epochs, not the latest.
- **A rank killed for running out of memory can hang a DDP fit.** The other ranks wait in a
  collective with their GPUs busy, so `squeue` shows RUNNING and CPU use looks normal. Watch the
  wandb heartbeat. Size `--mem` for the staged copy plus the workers (§3.6).
- **A NODE_FAIL requeue restarts from scratch.** SLURM requeues the job with its original
  sbatch script, which has no `--ckpt_path` unless it was submitted with `--resume`. Cancel the
  requeued job and resubmit with `--resume`.
- **Lightning swallows SIGTERM.** Its signal handler logs `Received SIGTERM` and keeps running, so
  `scancel` takes effect only at the KILL escalation, minutes later. A killed fit can still end in
  wandb's `finished` state.
- **The staged copy is node-wide shared memory.** `/dev/shm` is a tmpfs shared with other jobs on
  the node, and its size varies by node.
  - Another job can fill it. Staging then fails with SIGBUS (`Bus error`, exit 135) within a minute
    or two, which looks like your own memory bug but is not.
  - A hard kill skips the cleanup trap and leaves `/dev/shm/<jobid>` behind. Check
    `df -h /dev/shm` on the node.
- **Node-local temp directory.** wandb writes logged images under `$TMPDIR`, by default the node's
  local temp directory. A failing
  local disk there makes a write hang until the 3-hour NCCL timeout fires. `sinfo -R` shows a
  drained node's reason.
- **Resuming across a speed-flag change.** A resume re-resolves the leaf. To keep kernels unchanged
  when resuming a fit that ran without the flags, pass `--override trainer.benchmark=false`, plus
  `--override model.init_args.channels_last=false` for FNet3D.
- **Swapping the logger by override fails.** A dict override merges into the wandb logger's
  `init_args` and leaves `project` behind (`Option 'project' is not accepted`). Use
  `trainer.logger=false` or keep wandb (`WANDB_MODE=offline` works without an account).
- **Runs are not batch-reproducible.** MONAI crop centres are not seeded (§5.2). Compare runs by
  metrics, never by loss curves.
- **Stray files.** `dynacell` writes `config.yaml`, `cufile.log` and similar files into the current
  directory. Run it from a scratch directory.

## 10. Not covered or not run

- Anything on a cluster other than the lab's. The download was not repeated: the checks read the
  lab's byte-for-byte mirror of the S3 bucket.
- Submitting any fit or predict through sbatch from these copies; only previews and `--dry-run`
  ran.
- A 4-GPU fit, or a direct `srun` DDP launch.
- A predict from the public test archive, and the fast-tier evaluation itself (only their
  configuration was checked).
- A profiling run with all FOVs staged and the leaf's batch size (the A40 check used 10 FOVs and
  batch 4).
- Reference metric values for the baseline fits.
- A per-stage timing breakdown (Appendix A has whole-step rates only).

## Appendix A: training times and hardware

These are the fits behind the published DynaCell 3D models, surveyed on 2026-11-23. The survey
covers 79 trained fits: 35 iPSC (nucleus, membrane, ER, mitochondria), 22 A549 and 22 joint
iPSC + A549.

### A.1 How the numbers were obtained

- **s/batch (measured).** This is the slope of a least-squares fit of wandb `_runtime` against
  `trainer/global_step`, after dropping the first 10% of the run's steps (§5.1).
  - `_runtime` is wall-clock time, so the slope includes validation, image logging and checkpoint
    writes.
  - One batch is one pass of the training data loader. That equals one optimizer step for every
    family except pix2pix3D, which takes a discriminator step and a generator step per batch.
    Lightning's `global_step` and the checkpoint names count optimizer steps, so pix2pix3D's
    checkpoint step counts are twice its batch counts.
- **Patches/s and Mvox/s (derived).**
  - Patches/s = global patches per batch ÷ s/batch, where global patches per batch = `batch_size`
    × GPUs (§3.5).
  - Mvox/s = patches/s × the voxels in the network patch.
  - Patch sizes differ by up to 26× between families, so compare families by Mvox/s, not by
    patches/s.
- **Projected wall h (derived)** = s/batch × budget in batches ÷ 3600. This is the cost of one
  uninterrupted fit. It leaves out startup and staging.
- **GPU-h spent (measured)** comes from SLURM accounting: elapsed time × allocated GPUs, summed
  over every job in the chain that produced the final checkpoint, resumes included. It therefore
  includes startup, staging, restarts and any steps run again after a resume from an older
  checkpoint.
- **Complete** means the chain reached its budget: SLURM `COMPLETED`, and the final step equals
  `max_steps` or `max_epochs`.
- **Agreement.** For single-job fits that ran to completion, measured wall time matched the
  projection to within ±2% in 53 of 58 fits (all 58 fall between −1.8% and +7.3%).

### A.2 Hardware requested per family

These are the sbatch requests and data settings of the fits that were timed.

| Family | GPUs | CPUs (per task × tasks) | Host memory | Loader workers per GPU | Precision | Network patch (Z×Y×X) | CPU read crop | Patches per batch (global) | Budget (batches, iPSC nucleus) |
|---|---|---|---|---|---|---|---|---|---|
| FCMAE VSCyto3D / UNeXt2, pretrained and scratch | 4 × H100 or H200, DDP | 8 × 4 | 512 GB–1 TB | 4 | fp16 mixed | 15×384×384 | 20×600×600 | 32 × 4 = 128 | 200 epochs × 312 = 62,400 |
| FNet3D (paper) | 1, any type | 32 | 256–512 GB | 8 | fp32 | 32×64×64 | 32×64×64 | 48 | 200,000 |
| FNet3D-VSCyto3DAug | 1 × H100 or H200 | 32 | 512 GB | 4 | bf16 mixed | 32×384×384 | 32×600×600 | 8 | 200,000 |
| FNet3D big patch | 1 × H200 | 32 | 512 GB | 4 | bf16 mixed | 32×384×384 | 32×384×384 | 8 | 200,000 |
| UNetViT3D | 1 × H200 | 32 | 256 GB | 4 | bf16 mixed | 8×512×512 | 13×624×624 | 4 | 20 epochs × 6,400 = 128,000 |
| CELL-Diff 3D | 1 × H200 | 32 | 256 GB | 4 | bf16 mixed | 8×512×512 | 13×624×624 | 4 | 20 epochs × 6,400 = 128,000 |
| pix2pix3D | 4 × H200, DDP | 8 × 4 | 512 GB | 4 | bf16 mixed | 8×512×512 | 13×624×624 | 4 × 4 = 16 | 40 epochs × 1,600 = 64,000 |

Every family stages its train store to `/dev/shm` (§3.6). The stores are 68–104 GB (apparent size
measured with `du`; iPSC: `cell.zarr` 92 GB, `SEC61B.zarr` 90 GB, `TOMM20.zarr` 104 GB), which is
why the host-memory requests are large. The public iPSC training archives of §3.1 are 89–104 GB.

### A.3 iPSC-trained fits

Each cell lists four values in the order nucleus / membrane / ER / mitochondria. A single value
means the family was trained on one target only.

| Model | GPU type × count | s/batch (measured) | Patches/s (derived) | Mvox/s (derived) | Projected wall h (derived) | GPU-h spent (measured) | Complete? |
|---|---|---|---|---|---|---|---|
| FCMAE VSCyto3D, pretrained | H200 / H100 / H200 / H100, ×4 | 5.51 / 5.55 / 5.67 / 6.01 | 21–23 | 47–51 | 96 / 96 / 83 / 104 | 334 / 397 / 292 / 196 | membrane only; the others were cancelled at 87% / 87% / 47% |
| FCMAE VSCyto3D, scratch (UNeXt2) | H100 / H200 / H200 / H200, ×4 | 5.74 / 5.94 / 5.58 / 5.15 | 22–25 | 48–55 | 99 / 103 / 82 / 89 | 324 / 355 / 298 / 293 | none; cancelled at 70% / 83% / 84% / 82% |
| FCMAE VSCyto3D, scratch, re-run (nucleus, membrane) | H200 ×4 | 4.92 / 4.80 | 26.0 / 26.7 | 58 / 59 | 85 / 83 | 358 / 337 | yes |
| FNet3D (paper) | H200 / A6000 / A100 80 GB / A40, ×1 | 0.277 / 0.886 / 0.488 / 0.765 | 173 / 54 / 98 / 63 | 23 / 7 / 13 / 8 | 15 / 49 / 27 / 43 | 16 / 50 / 37 / 44 | yes |
| FNet3D-VSCyto3DAug | H100 / H100 / H200 / H200, ×1 | 1.037 / 0.613\* / 0.949 / 0.956 | 7.7 / 13.0 / 8.4 / 8.4 | 36 / 62 / 40 / 40 | 58 / 34 / 53 / 53 | 58 / 34 / 53 / 53 | yes |
| FNet3D big patch (nucleus) | H200 ×1 | 0.918 | 8.7 | 41 | 51 | 51 | yes |
| UNetViT3D | H200 ×1 | 1.180 / 1.174 / 1.139 / 1.202 | 3.3–3.5 | 7.0–7.4 | 42 / 42 / 34 / 43 | 43 / 42 / 35 / 44 | yes |
| pix2pix3D | H200 ×4 | 1.884 / 1.811 / 1.893 / 1.807 | 8.5–8.9 | 18–19 | 33 / 32 / 28 / 32 | 134 / 129 / 117 / 129 | yes |
| CELL-Diff 3D | H200 ×1 | 1.244 / 1.228 / 1.234 / 1.229 | 3.2–3.3 | 6.7–6.8 | 44 / 44 / 37 / 44 | 45 / 44 / 37 / 44 | yes |
| UNeXt2, timm encoder (ER) | H100 ×4 | 11.77 | 10.9 | 24 | 173 | 177 | no; stopped at 20% |

\* The only timed fit that trained with `trainer.benchmark` and `channels_last` on. Its nucleus
counterpart ran without them at 1.037 s/batch, but the two fits also differ in target and node, so
the gap is not attributed to the flags. An earlier nucleus FNet3D-VSCyto3DAug fit ran at
1.032 s/batch on H100 and used 57 GPU-h.

### A.4 A549-trained and joint fits

All 44 of these fits ran to completion. Each A549 fit trains on one A549 store. Each joint fit
reads one iPSC store and one A549 store, so it runs about twice as many batches per epoch. Ranges
span the four targets; UNetViT3D has ER and mitochondria fits only.

| Model | GPUs | s/batch, A549 | GPU-h spent, A549 | s/batch, joint | GPU-h spent, joint |
|---|---|---|---|---|---|
| FCMAE VSCyto3D, pretrained | H100 or H200, ×4 | 4.73–5.60 | 166–274 | 4.62–4.92 | 494–719 |
| FCMAE VSCyto3D, scratch | H100 or H200, ×4 | 4.73–5.75 | 167–278 | 4.70–5.59 | 617–736 |
| FNet3D (paper) | 1 × A40, A100 80 GB, H100 or H200 | 0.63–0.96 | 35–54 | 0.69–0.89 | 38–50 |
| UNetViT3D | H200 ×1 | 1.12–1.13 | 20–21 | 1.14–1.16 | 56–61 |
| pix2pix3D | H200 ×4 | 1.80–1.89 | 64–91 | 1.84–1.93 | 181–230 |
| CELL-Diff 3D | H200 ×1 | 1.15–1.24 | 21–30 | 1.15–1.27 | 56–76 |

The joint FCMAE fits took two or three jobs each. Their GPU-h includes restarts, and in one fit
about 39,000 batches were run again after a resume from an older checkpoint.

### A.5 Notes for profiling

- **GPU type changes within a family.** FNet3D (paper) ran on five GPU types, and within one type
  its s/batch varies by up to 2.5× between targets and stores. These tables therefore do not give
  per-GPU speed-ups. Compare GPU types only on the same leaf and store.
- **FCMAE 3D runs at similar rates on H100 and H200** (inferred, not tested). Across FCMAE 3D jobs
  with more than 5,000 fitted batches, s/batch spans 4.53–6.01 on H100 (18 jobs) and 4.62–5.94 on
  H200 (22 jobs). The two ranges overlap, which would fit a step limited by data loading or DDP
  synchronization rather than GPU compute. A per-stage timer (§5, `trainer.profiler`) or a fit on
  synthetic data on one node would test this. This 4-GPU family costs the most GPU-hours.
- **Speed flags measured in isolation.** On one H200, `fnet3d_paper` ran at 0.493 s/step with
  `benchmark` and `channels_last` and at 0.918 s/step without them. This figure comes from
  `model_overlays/fnet3d_paper_fit.yml`; the timed fits above predate the flags.
- **Precision is set per family:** fp32 for FNet3D (paper), fp16 mixed for FCMAE 3D and UNeXt2,
  and bf16 mixed for the rest.
