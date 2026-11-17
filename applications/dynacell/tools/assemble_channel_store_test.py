"""Integration tests for ``assemble_channel_store.py`` on tiny synthetic HCS stores."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from assemble_channel_store import assemble, main  # noqa: E402
from iohub.ngff import TransformationMeta, open_ome_zarr

_POSITIONS = ("A/1/0", "A/1/1", "B/2/0")
_SCALE = [1.0, 1.0, 0.29, 0.108, 0.108]


def _store(path: Path, channels: list[str], *, shape=(2, 4, 16, 16), seed=0, positions=_POSITIONS, norm=True):
    """Write a plate with one random (T, C, Z, Y, X) image per position; return its arrays."""
    rng = np.random.default_rng(seed)
    data = {}
    with open_ome_zarr(path, layout="hcs", mode="w-", channel_names=channels, version="0.5") as plate:
        if norm:
            plate.zattrs["normalization"] = {c: {"dataset_statistics": {"mean": 1.0}} for c in channels}
        for name in positions:
            arr = rng.random((shape[0], len(channels), *shape[1:]), dtype=np.float32)
            pos = plate.create_position(*name.split("/"))
            pos.create_image(
                "0", arr, chunks=(1, 1, *shape[1:]), transform=[TransformationMeta(type="scale", scale=_SCALE)]
            )
            if norm:
                pos.zattrs["normalization"] = {
                    c: {"fov_statistics": {"mean": float(i), "std": 1.0}} for i, c in enumerate(channels)
                }
            data[name] = arr
    return data


def test_assemble_keeps_base_channels_and_adds_renamed_ones(tmp_path: Path) -> None:
    """Kept channels and stats come from the base, added channels from their stores, renamed."""
    base = _store(tmp_path / "base.zarr", ["Brightfield", "Nuclei", "Phase3D"], seed=0)
    recon = _store(tmp_path / "wo1.zarr", ["Phase3D"], seed=1, norm=False)
    out = tmp_path / "out.zarr"

    main(
        [
            "--base", str(tmp_path / "base.zarr"),
            "--keep", "Phase3D", "Nuclei",
            "--add", f"{tmp_path / 'wo1.zarr'}:Phase3D:PhaseWO1",
            "--output", str(out),
        ]
    )  # fmt: skip

    with open_ome_zarr(out, mode="r") as plate:
        assert plate.channel_names == ["Phase3D", "Nuclei", "PhaseWO1"]
        assert set(plate.zattrs["normalization"]) == {"Phase3D", "Nuclei"}
        names = [name for name, _ in plate.positions()]
        assert names == list(_POSITIONS)
        for name, pos in plate.positions():
            got = np.asarray(pos["0"])
            np.testing.assert_array_equal(got[:, 0], base[name][:, 2])
            np.testing.assert_array_equal(got[:, 1], base[name][:, 1])
            np.testing.assert_array_equal(got[:, 2], recon[name][:, 0])
            assert pos.scale == _SCALE
            assert pos["0"].chunks == (1, 1, 4, 16, 16)
            norm = pos.zattrs["normalization"]
            assert norm == {
                "Phase3D": {"fov_statistics": {"mean": 2.0, "std": 1.0}},
                "Nuclei": {"fov_statistics": {"mean": 1.0, "std": 1.0}},
            }


def _o(tmp_path: Path) -> Path:
    return tmp_path / "o.zarr"


def test_assemble_refuses_a_shape_mismatch(tmp_path: Path) -> None:
    """A 2D (single-Z) reconstruction cannot sit next to a full-Z base channel."""
    _store(tmp_path / "base.zarr", ["Phase3D", "Nuclei"])
    _store(tmp_path / "wo2d.zarr", ["Phase2D"], shape=(2, 1, 16, 16))
    adds = [(tmp_path / "wo2d.zarr", "Phase2D", "PhaseWO1")]
    with pytest.raises(ValueError, match="reconstruction_dimension: 3"):
        assemble(tmp_path / "base.zarr", ["Nuclei"], adds, tmp_path / "o.zarr")


def test_assemble_refuses_a_missing_position(tmp_path: Path) -> None:
    """Every base position must exist in each added store."""
    _store(tmp_path / "base.zarr", ["Phase3D", "Nuclei"])
    _store(tmp_path / "partial.zarr", ["Phase3D"], positions=_POSITIONS[:2])
    adds = [(tmp_path / "partial.zarr", "Phase3D", "PhaseWO1")]
    with pytest.raises(KeyError):
        assemble(
            tmp_path / "base.zarr",
            ["Nuclei"],
            adds,
            tmp_path / "o.zarr",
        )


def test_assemble_refuses_duplicate_names_and_existing_output(tmp_path: Path) -> None:
    """Renaming onto a kept channel and writing over an existing store both raise."""
    _store(tmp_path / "base.zarr", ["Phase3D", "Nuclei"])
    _store(tmp_path / "wo1.zarr", ["Phase3D"])
    adds = [(tmp_path / "wo1.zarr", "Phase3D", "Phase3D")]
    with pytest.raises(ValueError, match="unique"):
        assemble(tmp_path / "base.zarr", ["Phase3D"], adds, tmp_path / "o.zarr")
    assemble(tmp_path / "base.zarr", ["Nuclei"], [], _o(tmp_path), limit=1)
    with pytest.raises(FileExistsError):
        assemble(tmp_path / "base.zarr", ["Nuclei"], [], _o(tmp_path))
