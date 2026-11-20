"""Coordinate-only adapter for the paper's deterministic Triplet image recipe."""

from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd

PAPER_RECIPE = {
    "name": "paper",
    "float32_matmul_precision": "highest",
    "z_extraction_window": 16,
    "focus_channel": "Phase3D",
    "z_focus_offset": 0.3,
    "z_reduction": "mip",
    "reference_pixel_size": 0.1494,
    "reference_pixel_size_z_um": 0.174,
    "initial_yx_patch_size": [160, 160],
    "final_yx_patch_size": [160, 160],
    "normalization_level": "fov_statistics",
    "normalization_subtrahend": "mean",
    "normalization_divisor": "std",
}


def predict_paper_cells(cells, checkpoint, output, batch_size=32):
    """Apply the original paper crop/resize/normalization code to supplied coordinates.

    The legacy crop reader needs tracking-shaped index columns. Its -1 sentinels
    are adapter-only, never inferred biological tracks and never exported. The
    unique internal id maps each result back to the caller's observation row.
    """
    import torch
    from iohub.ngff import open_ome_zarr
    from lightning.pytorch import Trainer, seed_everything

    from dynaclr.evaluation.predict_triplet import _DEFAULT_ENCODER_KWARGS, _build_dynaclr_module
    from viscy_data.triplet import TripletDataModule
    from viscy_transforms import NormalizeSampled

    torch.set_float32_matmul_precision(PAPER_RECIPE["float32_matmul_precision"])
    seed_everything(42)
    model = _build_dynaclr_module(str(checkpoint), _DEFAULT_ENCODER_KWARGS, [1, 1, 1, 160, 160])
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        precision="32-true",
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
    )
    embeddings = None
    seen = np.zeros(len(cells), dtype=bool)
    cells = cells.reset_index(drop=True)
    for (store, experiment, marker, channel), group in cells.groupby(
        ["store_path", "experiment", "marker", "channel_name"], sort=True
    ):
        plate = open_ome_zarr(store, mode="r", implementation="tensorstore")
        try:
            positions, tables = [], []
            for (well, fov), rows in group.groupby(["well", "fov"], sort=True):
                positions.append(plate[f"{well}/{fov}"])
                table = rows[["t", "y", "x", "z"]].copy()
                table[["t", "y", "x"]] = np.rint(table[["t", "y", "x"]]).astype(int)
                table["id"] = rows.index.to_numpy()
                for key in ["track_id", "parent_track_id", "parent_id"]:
                    table[key] = -1
                tables.append(table)

            class CoordinateDataModule(TripletDataModule):
                def _align_tracks_tables_with_positions(self):
                    return positions, [table.copy() for table in tables]

            settings = {
                key: value
                for key, value in PAPER_RECIPE.items()
                if key
                not in [
                    "name",
                    "float32_matmul_precision",
                    "normalization_level",
                    "normalization_subtrahend",
                    "normalization_divisor",
                ]
            }
            dm = CoordinateDataModule(
                data_path=store,
                tracks_path=store,
                source_channel=[channel],
                **settings,
                batch_size=batch_size,
                num_workers=0,
                normalizations=[
                    NormalizeSampled(keys=[channel], level="fov_statistics", subtrahend="mean", divisor="std")
                ],
            )
            predictions = trainer.predict(model, datamodule=dm)
            for batch in predictions:
                values = batch["features"].cpu().numpy()
                ids = np.array([row["id"] for row in batch["index"]], dtype=int)
                if embeddings is None:
                    embeddings = np.empty((len(cells), values.shape[1]), dtype=values.dtype)
                if seen[ids].any() or len(np.unique(ids)) != len(ids):
                    raise ValueError("Repeated source rows in paper inference")
                embeddings[ids] = values
                seen[ids] = True
        finally:
            plate.close()
    if not seen.all():
        cells.loc[~seen].to_parquet(output / "excluded_cells.parquet", index=False)
        raise ValueError(
            "Paper crop pipeline excluded cells; inspect excluded_cells.parquet and choose the cohort explicitly"
        )
    if embeddings is None or not np.isfinite(embeddings).all():
        raise ValueError("Paper inference produced missing or nonfinite embeddings")
    for column in cells:
        if isinstance(cells[column].dtype, (pd.StringDtype, pd.CategoricalDtype)):
            cells[column] = cells[column].astype(object)
    cells.index = pd.Index([str(i) for i in range(len(cells))], dtype=object)
    data = ad.AnnData(embeddings, obs=cells)
    data.uns["inference_recipe"] = PAPER_RECIPE
    destination = output / "embeddings" / "embeddings.zarr"
    destination.parent.mkdir()
    data.write_zarr(destination, convert_strings_to_categoricals=False)
    (output / "inference_recipe.json").write_text(json.dumps(PAPER_RECIPE, indent=2))
    return destination
