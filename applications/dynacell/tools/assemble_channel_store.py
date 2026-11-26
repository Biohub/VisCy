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
from contextlib import ExitStack
from itertools import islice
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
) -> None:
    """Write ``output``, after checking every input position so a failure leaves no partial store.

    Raises
    ------
    ValueError
        If an output channel name repeats, a kept or added channel is missing from its store,
        or an added store's position has a different (T, Z, Y, X) than the base position.
    KeyError
        If an added store lacks a base position.
    """
    out_channels = [*keep, *(new for _, _, new in adds)]
    if len(set(out_channels)) != len(out_channels):
        raise ValueError(f"output channel names must be unique, got {out_channels}")
    with ExitStack() as stack:
        src = stack.enter_context(open_ome_zarr(base, mode="r"))
        others = [stack.enter_context(open_ome_zarr(store, mode="r")) for store, _, _ in adds]
        keep_idx = [src.channel_names.index(c) for c in keep]
        add_idx = [other.channel_names.index(ch) for other, (_, ch, _) in zip(others, adds)]
        positions = list(islice(src.positions(), limit))
        # Metadata-only pass over every position, so an input problem surfaces before the
        # output store exists (a "w-" store written halfway looks usable to a trainer).
        for name, pos in positions:
            shape = pos["0"].shape
            expected = (shape[0], *shape[2:])
            for other, (store, _, _) in zip(others, adds):
                added_shape = other[name]["0"].shape
                added = (added_shape[0], *added_shape[2:])
                if added != expected:
                    raise ValueError(
                        f"{store}/{name}: (T, Z, Y, X) {added} != base {expected}; "
                        "reconstruct with the input's full Z (waveorder reconstruction_dimension: 3)"
                    )
        dst = stack.enter_context(
            open_ome_zarr(output, layout="hcs", mode="w-", channel_names=out_channels, version="0.5")
        )
        plate_norm = {c: v for c, v in src.zattrs.get("normalization", {}).items() if c in keep}
        if plate_norm:
            dst.zattrs["normalization"] = plate_norm
        for name, pos in positions:
            arr = pos["0"]
            planes = [np.asarray(arr[:, i]) for i in keep_idx]
            for other, i in zip(others, add_idx):
                planes.append(np.asarray(other[name]["0"][:, i]).astype(arr.dtype, copy=False))
            row, col, fov = name.split("/")
            out = dst.create_position(row, col, fov)
            chunks = (1, 1, *arr.chunks[2:])
            # Keep the base's shard shape (an unsharded base stays unsharded); it is always a whole
            # multiple of these chunks, since chunks[:2] are 1 and the rest equal the base's.
            shards_ratio = (
                None if arr.shards is None else tuple(s // c for s, c in zip(arr.shards, chunks, strict=True))
            )
            out.create_image(
                "0",
                np.stack(planes, axis=1),
                chunks=chunks,
                shards_ratio=shards_ratio,
                transform=[TransformationMeta(type="scale", scale=list(pos.scale))],
            )
            pos_norm = {c: v for c, v in pos.zattrs.get("normalization", {}).items() if c in keep}
            if pos_norm:
                out.zattrs["normalization"] = pos_norm
            print(f"{name} {tuple(out['0'].shape)}", flush=True)


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
