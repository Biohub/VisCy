# Cell-cycle prediction from phase movies with temporal foundation-model embeddings

Phase movies of tracked NIH3T3 nuclei in, the two PIP-FUCCI reporters out. The model, the data and the design choices
are described in the app README (*Temporal embeddings of foundation models*) and in
`Bill/5-cellcycle/README.md`; this file records what was run and what came out. Trained 2026-10-20, a rerun of the
runs of 2026-10-09/10, whose outputs were lost (see Caveats).

```
[p_t] --E (frozen, cached)--> [e_t] --T (RoPE transformer)--> [ê_t] --f--> Geminin-mKO, PIP-mVenus median in the nucleus
                                                                   └--g(p_t, ê_t)--> the two reporter images
```

## What was run

27 experiments (`experiments/`), one seed each, all on the same spatial split (`Bill/5-cellcycle`: 2,722 / 647 / 574
train / validation / test tracks) and the same budget: regression 15k steps, staining and multitask 30k steps, best
checkpoint on the validation split, scored on the test split (25,720 frames). T is 4 layers × 384 wide in every run;
E is one of four frozen pretrained encoders or one of two controls without pretraining, each frame embedded as
[global token | mean of the central 4×4 patch tokens].

| | MorphEm | CELL-DINO | DINOv3 | C-RADIOv4-H | Untrained ViT | ViT from scratch |
|---|---|---|---|---|---|---|
| regression f | ✓ + single-frame | ✓ + single-frame | ✓ + single-frame | ✓ + single-frame | ✓ + single-frame | ✓ + single-frame |
| virtual staining g | ✓ | ✓ | ✓ | ✓ + single-frame + ungated | ✓ | ✓ |
| multitask f + g | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

plus `stain_unconditioned`, the same UNet from phase alone. *Single-frame* removes T's layers (every ê_t reads its own
frame only); *ungated* applies the FiLM conditioning to every pixel instead of around the tracked nucleus.

*Untrained ViT* (`vit-untrained`) is the frozen control: MorphEm's ViT-S/16 architecture with its own random
initialization (seed 0), frozen and cached like the others (384 token + 384 centre patch mean), so pretraining is the
only difference from MorphEm. The UNet's encoder is trained from scratch in every run (no ImageNet weights), so
`stain_vit-untrained` has no pretrained weights anywhere. *ViT from scratch* (`vit-scratch`) is the same ViT, same
seed, trained with the model (its runs started on Intel nodes, where the seeded init draws different weights than on
the AMD node that embedded the frozen cache, so the two controls start from two draws of the same initialization):
`phase_frames` feeds every frame's centred crop, E runs in every step and its weights are kept in the checkpoint.

Scores (`dynaformer.cellcycle_metrics`): Pearson r per reporter over all test frames; track r, the per-track Pearson r
averaged over tracks (the trajectory, which frame-to-frame jitter destroys); PIP-FUCCI phase accuracy and macro F1
over the frames whose true phase is clear (G1 / S / G2/M). Always answering G1 scores phase accuracy **0.687** and F1
0.27. The stain is read out as the median of the predicted image inside the true nucleus mask, in the regression
head's units; pixel r is the per-crop image correlation.

## Results — test split

### Regression head f(ê_t)

| encoder | r Geminin | r PIP | R² | track r | phase acc | phase F1 |
|---|---|---|---|---|---|---|
| MorphEm | 0.702 | 0.655 | 0.448 | **0.573** | 0.867 | 0.770 |
| CELL-DINO | 0.716 | **0.655** | **0.457** | 0.568 | **0.873** | **0.785** |
| DINOv3 | 0.728 | 0.651 | 0.450 | 0.436 | 0.823 | 0.689 |
| C-RADIOv4-H | **0.738** | 0.610 | 0.445 | 0.512 | 0.868 | 0.777 |
| Untrained ViT | 0.618 | 0.463 | 0.282 | 0.340 | 0.732 | 0.565 |
| ViT from scratch | 0.642 | 0.477 | 0.286 | 0.403 | 0.798 | 0.668 |
| *single-frame:* MorphEm | 0.657 | 0.510 | 0.329 | 0.271 | 0.699 | 0.544 |
| CELL-DINO | 0.647 | 0.528 | 0.331 | 0.271 | 0.729 | 0.564 |
| DINOv3 | 0.577 | 0.464 | 0.247 | 0.222 | 0.682 | 0.513 |
| C-RADIOv4-H | 0.618 | 0.504 | 0.296 | 0.246 | 0.710 | 0.538 |
| Untrained ViT | 0.410 | 0.325 | 0.106 | 0.120 | 0.630 | 0.384 |
| ViT from scratch | 0.575 | 0.465 | 0.231 | 0.229 | 0.712 | 0.564 |

### Virtual staining g(p_t, ê_t), read out in the nucleus

| model | r Geminin | r PIP | R² | track r | phase acc | phase F1 | pixel r |
|---|---|---|---|---|---|---|---|
| UNet, phase only | 0.688 | 0.558 | 0.374 | 0.367 | 0.777 | 0.644 | 0.614 |
| + C-RADIO, single frame | 0.659 | 0.557 | 0.332 | 0.347 | 0.757 | 0.624 | 0.611 |
| + untrained ViT, temporal | 0.695 | 0.617 | 0.417 | 0.441 | 0.788 | 0.638 | 0.608 |
| + ViT from scratch, temporal | 0.689 | 0.564 | 0.374 | 0.379 | 0.775 | 0.634 | 0.610 |
| + MorphEm, temporal | 0.647 | 0.624 | 0.373 | 0.572 | 0.855 | 0.754 | 0.613 |
| + CELL-DINO, temporal | 0.677 | **0.673** | 0.434 | **0.594** | **0.872** | **0.775** | **0.621** |
| + DINOv3, temporal | **0.742** | 0.637 | **0.466** | 0.564 | 0.859 | 0.759 | 0.619 |
| + C-RADIOv4-H, temporal | 0.717 | 0.638 | 0.449 | 0.474 | 0.847 | 0.736 | 0.586 |
| + C-RADIOv4-H, temporal, ungated | 0.695 | 0.624 | 0.399 | 0.492 | 0.840 | 0.708 | 0.601 |

### Multitask f + g (each readout from the checkpoint selected on it)

| encoder | f: r Gem / r PIP | f: track r | f: phase acc | g: r Gem / r PIP | g: track r | g: phase acc | g: pixel r |
|---|---|---|---|---|---|---|---|
| MorphEm | 0.703 / 0.635 | 0.545 | 0.870 | 0.708 / 0.640 | 0.588 | 0.871 | 0.621 |
| CELL-DINO | 0.707 / 0.655 | 0.544 | **0.881** | 0.703 / 0.655 | 0.578 | **0.879** | 0.619 |
| DINOv3 | 0.719 / 0.607 | 0.433 | 0.840 | 0.725 / 0.609 | 0.491 | 0.841 | 0.586 |
| C-RADIOv4-H | 0.731 / 0.597 | 0.514 | 0.866 | 0.727 / 0.591 | 0.563 | 0.866 | 0.619 |
| Untrained ViT | 0.651 / 0.484 | 0.347 | 0.755 | 0.703 / 0.564 | 0.445 | 0.805 | 0.595 |
| ViT from scratch | 0.559 / 0.449 | 0.331 | 0.730 | 0.689 / 0.586 | 0.437 | 0.820 | 0.616 |

### Single-frame linear probe of each embedding (`scripts/probe_embeddings.py`)

Ridge regression from one frame's embedding, test-split r averaged over the two reporters:

| encoder | global token | central patch mean | both |
|---|---|---|---|
| CELL-DINO | 0.506 | 0.554 | **0.557** |
| MorphEm | 0.481 | 0.527 | 0.538 |
| C-RADIOv4-H | 0.471 | 0.523 | 0.530 |
| DINOv3 | 0.429 | 0.497 | 0.511 |
| Untrained ViT | 0.268 | 0.330 | 0.348 |

## What the numbers say

1. **Temporal integration is the largest effect, for every pretrained encoder.** The RoPE transformer roughly
   doubles track r (0.22-0.27 → 0.44-0.57) and lifts phase accuracy from 0.68-0.73, around always-G1, to
   0.82-0.87. The single-frame models cannot see S phase: PIP-mVenus drops within minutes at S onset with no visible
   change in one phase frame. `figures/trajectories.png` shows it: per-frame predictions jitter, temporal ones follow
   the G1/S transitions.
2. **The pretrained encoders end up close, and their order below the top is not stable.** With temporal integration
   the four regression heads are within 0.04 of each other in r per reporter (Geminin 0.70-0.74, PIP 0.61-0.66) and
   within 0.05 in phase accuracy (0.82-0.87); track r spreads more (0.44-0.57). CELL-DINO is at or near the top of
   most columns in every task (phase accuracy 0.872-0.881), and MorphEm, a ViT-S, keeps up with the ViT-L/H models
   throughout. Below that the order changes between runs: here DINOv3's regression head is the weakest (phase
   accuracy 0.823, track r 0.44) and its stain model second in phase accuracy (0.859); in the first run its stain
   model was the weakest (0.825; Caveats). Per frame, the two microscopy models carry the most reporter information (linear probe).
3. **The regression head needs a pretrained E.** The untrained ViT, MorphEm's network with random weights, reaches
   phase accuracy 0.732, above always-G1 but below every pretrained encoder (0.823-0.873; 0.867 for MorphEm itself),
   and track r 0.34 against 0.44-0.57 (validation phase accuracy 0.717 against 0.802-0.865). One frame's random
   features carry less of the reporters (linear probe r 0.35 against 0.51-0.56). T still integrates something from
   them (single-frame → temporal: track r 0.12 → 0.34, R² 0.11 → 0.28), but the result is only as good as the best
   pretrained single frame: R² 0.28 against 0.25-0.33, phase accuracy 0.732 against 0.682-0.729. Training the same
   ViT from scratch with f recovers part of the gap, not all of it: phase accuracy 0.798 (validation 0.773) and track
   r 0.40, though R² stays at 0.29. The scored checkpoint, best on validation r, is at 6.5k of 15k steps, and
   validation phase accuracy stays at 0.78-0.79 from about 8k steps on, so more steps would not close the rest.
   Single-frame, it matches the pretrained single-frame models in phase accuracy (0.712 against 0.682-0.729) and
   track r (0.23 against 0.22-0.27) but not in R² (0.23 against 0.25-0.33).
4. **Staining needs pretraining less, but still gains from it**, as the UNet learns its own features from the phase
   image (phase only: phase accuracy 0.777, track r 0.37). Conditioning on the untrained ViT adds little phase
   accuracy (0.788; validation 0.745 → 0.766) but lifts PIP r from 0.56 to 0.62 and track r from 0.37 to 0.44; every
   pretrained encoder does more (phase accuracy 0.847-0.872, track r 0.47-0.59). Training that ViT from scratch
   through the stain loss alone gains nothing over the phase-only UNet: phase accuracy 0.775, PIP r 0.56, track r
   0.38 and pixel r 0.610 against 0.777, 0.56, 0.37 and 0.614. Its validation r, the checkpoint monitor, rises slowly
   to 0.548 at 22k of 30k steps, against 0.535 for the phase-only UNet and 0.643 for the frozen control, at a similar
   training loss (0.031 against 0.032 and 0.030 at 20-25k steps). With random features, training f together with g
   helps both a little: f's phase accuracy 0.732 → 0.755 (validation 0.717 → 0.738) and g's 0.788 → 0.805
   (validation 0.766 → 0.778), at about the same track r (0.34 → 0.35, 0.44 → 0.45). Trained from scratch with f and
   g together, the ViT's stain readout does better than through the stain loss alone: phase accuracy 0.820, track r
   0.437 and pixel r 0.616 against 0.775, 0.379 and 0.610 (validation phase accuracy 0.794 against 0.748), level with
   the frozen untrained ViT's multitask readout (0.805, 0.445, 0.595). Its regression head does worse than when the
   ViT trains with f alone: track r 0.331 and phase accuracy 0.730 against 0.403 and 0.798, below the frozen
   control's multitask head (0.347, 0.755). f does not fit its training targets here: its training loss is still
   0.13 at the end (25-30k steps), where the frozen control's multitask head and the ViT trained with f alone end at
   0.035 and 0.034.
5. **The nucleus-centred patch tokens matter.** For every encoder the mean of the central patch tokens predicts the
   reporters better than the global token, and the concatenation used here is best.
6. **Conditioning the UNet on ê_t helps where the phase image cannot: PIP and the cycle's timing.** Against the
   phase-only UNet, the pretrained encoders take PIP r from 0.56 to 0.62-0.67, track r from 0.37 to 0.47-0.59 and
   phase accuracy from 0.777 to 0.85-0.87; Geminin, which the UNet partly reads from the phase image, does not gain
   consistently (0.69 against 0.65-0.74). The per-frame embedding alone adds nothing (single-frame row: phase
   accuracy 0.757, track r 0.35), so the gain is temporal. Pixel r stays at 0.59-0.62 (phase only 0.614): the
   conditioning acts on the tracked nucleus, not the whole crop. On the test frames that are truly in S, the
   phase-only UNet's PIP readout has a median of 56 counts and is below the off-level (44) in 45% of them;
   conditioned over time on C-RADIOv4-H or CELL-DINO it is 26 and 10 counts, off in 57% and 73%
   (`figures/stain_track*.png` shows three tracks).
7. **The centre gate makes little difference here.** Conditioning every pixel instead of around the tracked nucleus
   gives phase accuracy 0.840 against 0.847 with the gate (S-phase F1 0.66 against 0.68), track r 0.49 against 0.47
   and pixel r 0.601 against 0.586 (C-RADIOv4-H). The first run's gap (0.740 against 0.863, S-phase F1 0.17) did not
   recur, so a single run per configuration does not establish an effect of the gate on phase calls.
8. **Multitask is not a consistent win.** Against the single-task runs, it raises the stain readout's track r for
   C-RADIOv4-H (0.47 → 0.56) and lowers it for DINOv3 (0.56 → 0.49), and leaves the regression head's at or below
   single-task (MorphEm 0.57 → 0.55, CELL-DINO 0.57 → 0.54). Phase accuracy moves by at most 0.02 for the pretrained
   encoders. The best models overall are CELL-DINO's: phase accuracy 0.872-0.881, F1 0.775-0.789 and track r
   0.54-0.59 in all three tasks.

## Caveats

- **One seed per configuration, and the runs vary with the GPU.** A run reproduces exactly on the same GPU model but
  not across models, and checkpoint selection amplifies the difference. All 27 runs were retrained on 2026-10-20 after
  the first runs' outputs were lost. The two that ran on the same GPU model as before (C-RADIOv4-H regression,
  temporal and single-frame) reproduced their scores exactly; the others moved by up to 0.10 in phase accuracy and
  0.11 in track r (on average 0.02 and 0.03). That overturned three findings of the first run: the centre gate (ungated phase
  accuracy 0.740 → 0.840), the untrained ViT's regression head below always-G1 (0.651 → 0.732) and the stain-only ViT
  from scratch overfitting early (validation r peak at 4k → 22k steps; phase accuracy 0.674 → 0.775). Differences of
  a few hundredths between encoders are within this variation. One random draw per ViT control: seed 0 for both, but
  drawn on different CPU types (above), so not the same weights.
- **One checkpoint was selected on a partial validation.** A requeue saved the multitask run from scratch in the
  middle of a validation, and the resumed job finished that validation on 180 of its 7,165 stain frames: the score
  (0.653) became the best, so the stain readout comes from step 29k (full validation r 0.609) instead of the run's
  best full validation (0.615 at 25k). The requeue checkpoint keeps the loop's place but not the frames scored so
  far. No other checkpoint here or in `../cellcycle_leger` was selected this way.
- **Frozen pretrained encoders.** No pretrained E was fine-tuned; the comparison is of the representations as
  published.
- **The from-scratch control is not tuned.** It uses the frozen runs' hyperparameters (lr 3e-4, weight decay 0.05, 500
  warmup steps), not ones chosen for training a ViT. It is also far slower: E runs in every step (0.5-1 s per step on
  an H100), where the cached runs never run it.
- **Bidirectional T uses future frames**, which suits offline analysis of a recorded movie. Transitions in the last
  ~25 frames of the movie, with little future context, are the commonest misses (`figures/trajectories.png`, track
  69), and a track that continues through a division into a daughter gets smoothed over the division
  (`figures/stain_track1712.png`, t = 0).
- **One experiment.** All tracks come from one 14.5 h mosaic; the split is spatial (blocks, lineages kept together,
  a guard band so no pixel appears in crops of two splits), not across experiments.
- **Earlier runs are superseded.** The first round trained with a sampler that silently skipped 70% of the training
  tracks (all tracks shorter than the 64-frame window); those runs were deleted. Their W&B runs (started before
  2026-10-09 17:20 PDT) remain in the project, as do those of the first corrected runs (2026-10-09 to 2026-10-11),
  which this rerun replaces.

## Where things are

- Runs: `/hpc/projects/comp.micro/virtual_staining/models/dynaformer/runs/cellcycle/<experiment>/` (checkpoints,
  `eval/best_<readout>/{predictions,metrics}_{val,test}.{csv,json}`); tables `summary_{val,test}.csv`; figures in
  `figures/`.
- W&B: project `dynaformer-cellcycle` (runs started after 2026-10-20 17:30 PDT; a run that was requeued appears as one
  W&B run per allocation).
- Reproduce: `run_embed.slurm` per encoder (`vit-untrained` too; `vit-scratch` runs E inside training and needs no
  cache), then `./submit.sh --time=0:30:00 [--constraint="h200|h100|a100"] -- <leaves>` (the `vit-scratch` leaves
  with `--constraint=intel_8480c`, the CPU type their runs started on); `python ../../scripts/summarize_cellcycle.py
  --split test`, `plot_cellcycle.py`, `render_stain_examples.py`. To retrain a leaf from scratch, move its run
  directory away first: `run_fit.slurm` resumes from what it finds there.
