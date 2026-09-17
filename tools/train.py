#!/usr/bin/env python3
"""Train the sparse U-Net on the official urban railway split."""

from __future__ import annotations

import argparse

from railway3d_seg.config import load_config, require_section
from railway3d_seg.data import RailwayBlockDataset
from railway3d_seg.engine import train_model
from railway3d_seg.io import read_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/urban_railway_sparse_unet.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    data = require_section(config, "data")
    root = data["root"]
    train_paths = read_manifest(root, data["train_manifest"])
    validation_paths = read_manifest(root, data["validation_manifest"])
    common = {
        "voxel_size": float(data["voxel_size"]),
        "block_size": float(data["block_size"]),
        "max_points": int(data["max_training_points"]),
        "feature_properties": data.get("feature_properties", ["intensity"]),
        "seed": int(config.get("seed", 2025)),
        "cache_size": int(data.get("cache_size", 2)),
        "label_property": data.get("label_property"),
    }
    train_dataset = RailwayBlockDataset(
        train_paths,
        blocks_per_file=int(data.get("train_blocks_per_file", 16)),
        augment=True,
        **common,
    )
    validation_dataset = RailwayBlockDataset(
        validation_paths,
        blocks_per_file=int(data.get("validation_blocks_per_file", 4)),
        augment=False,
        **common,
    )
    checkpoint = train_model(train_dataset, validation_dataset, config)
    print(f"best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()

