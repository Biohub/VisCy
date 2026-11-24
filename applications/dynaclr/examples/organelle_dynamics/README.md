# Start here: embed and phenotype

Use `embed` to extract raw embeddings.
Use `phenotype` to compare treated cells with controls.

```mermaid
flowchart LR
    I["Images + cell coordinates"] --> E["embed"]
    C["Checkpoint + image recipe"] --> E
    E --> R["Raw embeddings"]
    R --> P["phenotype"]
    M["Controls + timing metadata"] --> P
    P --> N["Normalized embeddings"]
    P --> U["PCA / UMAP figures"]
    P --> D["MMD per HPI"]
```

## Run

Run commands from the VisCy repository root.

Give the agent your completed configuration and data paths.
If no configuration exists, the agent requests one using the appropriate template:

- [DynaCell/DynaCLR](embed_dynacell.yaml) or [CELL-DINO](embed_cell_dino.yaml) for embedding.
- [Phenotype](phenotype.yaml) for population analysis.

Fill the template's paths, metadata and checkpoint identity.
Any YAML filename works; copying or renaming it is optional.
The agent checks required fields and unresolved placeholders before execution.
For other data formats, follow [data preparation](CONTRACT.md#prepare-external-data).
For nuclear labels without tracking, use the [label-input configuration](CONTRACT.md#data-without-tracking) with the same `embed` command.

Install the environment once:

```bash
uv sync --locked --package dynacell --extra profiler
source .venv/bin/activate
```

Run with your configuration paths:

```bash
dynacell embed --config /path/to/embedding.yml --output results/embed
dynacell phenotype --config /path/to/phenotype.yml --output results/phenotype
```

Set the phenotype configuration's `embeddings` to the generated `predictions.zarr`.
Relative input paths resolve against the configuration's `root`, relative to its directory.
This includes CELL-DINO's `training_config` recipe path.
Use a new output directory for each run.

Without activation, prefix commands with `uv run --locked --package dynacell --extra profiler`.
`--no-sync` is optional for environments already synchronized before a job.
It skips environment synchronization; the default instructions do not require it.

## File map

| File or folder | Purpose |
| --- | --- |
| [embed/SKILL.md](../../skills/embed/SKILL.md) | Agent instructions for checkpoint selection and image preprocessing. |
| [phenotype/SKILL.md](../../skills/phenotype/SKILL.md) | Agent instructions for control normalization, UMAP and MMD. |
| [CONTRACT.md](CONTRACT.md) | Input fields, normalization rules and output formats. |
| [dynacell CLI](../../../dynacell/src/dynacell/__main__.py) | Routes `embed` and `phenotype` to the workflow runner. |
| [organelle_workflow.py](../../src/dynaclr/evaluation/organelle_workflow.py) | Command entry point, metadata checks, classifier calls and plotting. |
| [organelle_inference.py](../../src/dynaclr/evaluation/organelle_inference.py) | Paper DynaCLR image preprocessing and inference. |
| [organelle_cells.py](../../src/dynaclr/evaluation/organelle_cells.py) | Label centroids, independent-instance tables and crop exclusions. |
| [organelle_normalization.py](../../src/dynaclr/evaluation/organelle_normalization.py) | Control normalization and embedding export. |
| [organelle_mmd.py](../../src/dynaclr/evaluation/organelle_mmd.py) | MMD statistics and uncertainty calculations. |
| [tools/](tools/) | Optional paper-data inventory and small example. |
| [verification/](verification/README.md) | Paper comparison scripts and numerical evidence. |
| [test_organelle_workflow.py](../../tests/test_organelle_workflow.py) | Automated regression tests. |

Image normalization belongs to `embed`.
Control-relative embedding normalization belongs to `phenotype`.
MMD uses PCA features, not UMAP coordinates.

The [verification report](verification/README.md) describes component tests, not complete Figure 6 reproduction.
Witness scoring and temporal alignment remain outside these skills.
