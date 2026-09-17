#!/usr/bin/env python3
"""Create manifests from explicit official train/validation/test directories."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    aliases = {
        "train": ("train", "training", "Train", "Training"),
        "validation": ("validation", "val", "Validation", "Val"),
        "test": ("test", "testing", "Test", "Testing"),
    }
    for split, candidates in aliases.items():
        directory = next((args.root / name for name in candidates if (args.root / name).is_dir()), None)
        if directory is None:
            raise FileNotFoundError(
                f"could not locate an explicit {split} directory under {args.root}; "
                "do not invent a split—use the organizer-provided partition"
            )
        files = sorted(directory.rglob("*.ply"))
        if not files:
            raise FileNotFoundError(f"no PLY files found under {directory}")
        manifest = args.output_dir / f"urban_{split}.txt"
        manifest.write_text(
            "".join(f"{path.relative_to(args.root).as_posix()}\n" for path in files),
            encoding="utf-8",
        )
        print(f"{split}: {len(files)} files -> {manifest}")


if __name__ == "__main__":
    main()

