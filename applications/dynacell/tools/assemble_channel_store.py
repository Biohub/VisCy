r"""Assemble a new HCS OME-Zarr store from kept channels of a base store plus channels from others.

Built for adding phase reconstructions (e.g. waveorder outputs) as new source channels to
DynaCell train and test data without touching the registered stores. The output keeps the
base store's plate layout, position names, chunks and OME scale; every added channel must
come from a store with the same position names and the same (T, Z, Y, X) per position.

Normalization statistics of the kept channels are copied from the base store. Added channels
get none: run ``viscy preprocess --channel_names '["<NEW>"]'`` on the output afterwards.

Example (train side, one reconstruction, renamed so it never collides with Phase3D)::

    python applications/dynacell/tools/assemble_channel_store.py \
        --base .../ipsc/dataset_v4/train/cell.zarr --keep Phase3D Nuclei \
        --add .../waveorder/train_wo1.zarr:Phase3D:PhaseWO1 \
        --output .../ipsc/dataset_v4/train/cell_wo1.zarr
"""

import argparse
from pathlib import Path

import numpy as np
from iohub import open_ome_zarr
from iohub.ngff.models import TransformationMeta


def _parse_add(spec: str) -> tuple[Path, str, str]:
    """Split ``STORE:SRC_CHANNEL:NEW_CHANNEL`` (the store path may itself hold colons)."""
    store, src_ch, new_ch = spec.rsplit(":", 2)
    if not (store and src_ch and new_ch):
        raise ValueError(f"--add expects STORE:SRC_CHANNEL:NEW_CHANNEL, got {spec!r}")
    return Path(store), src_ch, new_ch


def assemble(
    base: Path, keep: list[str], adds: list[tuple[Path, str, str]], output: Path, limit: int | None = None
) -> list[str]:
    """Write ``output`` and return the position names written.

    Raises
    ------
    ValueError
        If an output channel name repeats, or an added store's position has a different
        (T, Z, Y, X) than the base position.
    KeyError
        If an added store lacks a base position or the named channel.
    """
    out_channels = [*keep, *(new for _, _, new in adds)]
    if len(set(out_channels)) != len(out_channels):
        raise ValueError(f"output channel names must be unique, got {out_channels}")
    written = []
    with open_ome_zarr(base, mode="r") as src:
        others = [open_ome_zarr(store, mode="r") for store, _, _ in adds]
        try:
            with open_ome_zarr(output, layout="hcs", mode="w-", channel_names=out_channels, version="0.5") as dst:
                plate_norm = src.zattrs.get("normalization", {})
                dst.zattrs["normalization"] = {c: plate_norm[c] for c in keep if c in plate_norm}
                keep_idx = [src.channel_names.index(c) for c in keep]
                add_idx = [other.channel_names.index(ch) for other, (_, ch, _) in zip(others, adds)]
                positions = list(src.positions())[:limit]
                for name, pos in positions:
                    arr = pos["0"]
                    planes = [np.asarray(arr[:, i]) for i in keep_idx]
                    for other, i, (store, _, _) in zip(others, add_idx, adds):
                        other_arr = other[name]["0"]
                        added = np.asarray(other_arr[:, i])
                        expected = (arr.shape[0], *arr.shape[2:])
                        if added.shape != expected:
                            raise ValueError(
                                f"{store}/{name}: (T, Z, Y, X) {added.shape} != base {expected}; "
                                "reconstruct with the input's full Z (waveorder reconstruction_dimension: 3)"
                            )
                        planes.append(added.astype(arr.dtype, copy=False))
                    row, col, fov = name.split("/")
                    out = dst.create_position(row, col, fov)
                    out.create_image(
                        "0",
                        np.stack(planes, axis=1),
                        chunks=arr.chunks,
                        transform=[TransformationMeta(type="scale", scale=list(pos.scale))],
                    )
                    pos_norm = pos.zattrs.get("normalization", {})
                    out.zattrs["normalization"] = {c: pos_norm[c] for c in keep if c in pos_norm}
                    written.append(name)
                    print(f"{name} {tuple(out['0'].shape)}", flush=True)
        finally:
            for other in others:
                other.close()
    return written


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and assemble the store."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True, help="store whose layout and kept channels are copied")
    ap.add_argument("--keep", nargs="*", default=[], help="base channels to copy, e.g. Phase3D Nuclei")
    ap.add_argument(
        "--add", action="append", default=[], metavar="STORE:SRC_CH:NEW_CH", help="channel to add (repeatable)"
    )
    ap.add_argument("--output", type=Path, required=True, help="new store; refuses to overwrite")
    ap.add_argument("--limit", type=int, default=None, help="copy only the first N positions (smoke tests)")
    args = ap.parse_args(argv)
    if not args.keep and not args.add:
        ap.error("nothing to write: pass --keep and/or --add")
    assemble(args.base, args.keep, [_parse_add(a) for a in args.add], args.output, args.limit)


if __name__ == "__main__":
    main()
