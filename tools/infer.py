#!/usr/bin/env python3
"""Run overlap-voted inference on an official test manifest."""

from __future__ import annotations

import argparse

from railway3d_seg.config import load_config, require_section
from railway3d_seg.engine import infer_files
from railway3d_seg.io import read_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/urban_railway_sparse_unet.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="outputs/urban_test_predictions")
    args = parser.parse_args()
    config = load_config(args.config)
    data = require_section(config, "data")
    paths = read_manifest(data["root"], data["test_manifest"])
    written = infer_files(paths, args.checkpoint, args.output_dir, config)
    print(f"wrote {len(written)} prediction files to {args.output_dir}")


if __name__ == "__main__":
    main()

