from pathlib import Path

import anndata as ad
import pandas as pd
import yaml

pd.options.future.infer_string = False
import argparse

from dynaclr.evaluation.organelle_workflow import run, sha256

parser = argparse.ArgumentParser(
    description="Run a small SEC61B paper-data analysis with a pinned infection classifier."
)
parser.add_argument("--prediction-root", required=True, type=Path)
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
p = args.prediction_root
o = args.output
o.mkdir(parents=True, exist_ok=False)
a = ad.read_zarr(p / "embeddings/SEC61B.zarr")
a = a[~a.obs.is_blank.astype(bool)].copy()
rows = []
for time in [12.0, 18.0, 24.0]:
    for condition in ["uninfected", "ZIKV"]:
        rows.extend(
            a.obs.reset_index(drop=True)
            .query("hours_post_perturbation == @time and perturbation == @condition")
            .index[:32]
        )
a = a[rows].copy()
a.obs["cell_id"] = a.obs.fov_name.astype(str) + ":" + a.obs.track_id.astype(str) + ":" + a.obs.t.astype(str)
a.obs["cell_id"] = a.obs.cell_id.astype(object)
a.obs.index = pd.Index(a.obs.cell_id.to_numpy(dtype=str), dtype=object)
# Real biological/source IDs retained; no manufactured infection labels.
for c in a.obs:
    if isinstance(a.obs[c].dtype, (pd.StringDtype, pd.CategoricalDtype)):
        a.obs[c] = a.obs[c].astype(object)
a.write_zarr(o / "input.zarr", convert_strings_to_categoricals=False)
model = p / "classifier/DynaCLR/infection_state__pAL17_to_SEC61B/pipelines/infection_state_SEC61B.joblib"
m = dict(
    schema_version=1,
    root=".",
    embeddings="input.zarr",
    feature_space="DynaCLR-jbrwhzr3-epoch105-step84800",
    control_label="uninfected",
    metadata=dict(control_group="2026_03_24", replicate="2026_03_24"),
    analysis=dict(min_cells=10, max_cells=32, permutations=20, repeats=5),
    classifiers=[
        dict(
            path=str(model),
            sha256=sha256(model),
            feature_space="DynaCLR-jbrwhzr3-epoch105-step84800",
            marker="SEC61B",
            task="infection_state",
        )
    ],
)
(o / "manifest.yaml").write_text(yaml.safe_dump(m))
print("Running", a.shape, flush=True)
run(o / "manifest.yaml", o / "analysis", "analyze")
print("COMPLETE", flush=True)
