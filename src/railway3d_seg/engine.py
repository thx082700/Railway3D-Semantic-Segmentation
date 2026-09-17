"""Training and bounded-memory scene inference."""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from railway3d_seg.config import require_section
from railway3d_seg.data import VoxelSample, sliding_block_indices, voxelize_arrays
from railway3d_seg.io import read_ply, save_prediction
from railway3d_seg.labels import NUM_CLASSES
from railway3d_seg.losses import semantic_loss
from railway3d_seg.metrics import confusion_matrix, metrics_from_confusion
from railway3d_seg.model import build_model, load_compatible_weights


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _dependencies():
    try:
        import MinkowskiEngine as ME
        import torch
        from torch.utils.data import DataLoader
        from tqdm import tqdm
    except ImportError as error:  # pragma: no cover - optional CUDA environment
        raise ImportError(
            "Training/inference requires PyTorch, tqdm, and MinkowskiEngine"
        ) from error
    return torch, ME, DataLoader, tqdm


def _collate(samples):
    return samples


def resolve_device(requested: str):
    torch, _, _, _ = _dependencies()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def sparse_batch(samples: list[VoxelSample], device):
    torch, ME, _, _ = _dependencies()
    coordinates = ME.utils.batched_coordinates([sample.coords for sample in samples])
    features = torch.from_numpy(np.concatenate([sample.features for sample in samples])).to(device)
    sparse = ME.SparseTensor(features=features, coordinates=coordinates, device=device)
    labels = None
    if all(sample.labels is not None for sample in samples):
        labels = torch.from_numpy(
            np.concatenate([sample.labels for sample in samples])  # type: ignore[arg-type]
        ).long().to(device)
    return sparse, labels


def _run_epoch(model, loader, optimizer, scaler, config, device, *, training: bool):
    torch, _, _, tqdm = _dependencies()
    train_config = require_section(config, "train")
    loss_config = require_section(config, "loss")
    use_amp = bool(train_config.get("amp", True)) and device.type == "cuda"
    model.train(training)
    totals = {"loss": 0.0, "cross_entropy": 0.0, "lovasz": 0.0, "batches": 0}
    confusion = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for samples in tqdm(loader, leave=False):
            sparse, target = sparse_batch(samples, device)
            if target is None:
                raise ValueError("training/validation samples require semantic labels")
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(sparse)
                loss, parts = semantic_loss(
                    logits,
                    target,
                    class_weights=loss_config.get("class_weights"),
                    lovasz_weight=float(loss_config.get("lovasz_weight", 1.0)),
                )
            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), float(train_config.get("gradient_clip", 10.0))
                )
                scaler.step(optimizer)
                scaler.update()
            prediction = logits.detach().argmax(dim=1).cpu().numpy()
            confusion += confusion_matrix(target.detach().cpu().numpy(), prediction)
            totals["loss"] += float(loss.detach())
            totals["cross_entropy"] += float(parts["cross_entropy"])
            totals["lovasz"] += float(parts["lovasz"])
            totals["batches"] += 1
    batches = max(int(totals.pop("batches")), 1)
    output = {name: value / batches for name, value in totals.items()}
    metrics = metrics_from_confusion(confusion)
    output.update({"mean_iou": metrics.mean_iou, "overall_accuracy": metrics.overall_accuracy})
    return output


def train_model(train_dataset, validation_dataset, config: dict[str, Any]) -> Path:
    torch, _, DataLoader, _ = _dependencies()
    model_config = require_section(config, "model")
    train_config = require_section(config, "train")
    data_config = require_section(config, "data")
    seed_everything(int(config.get("seed", 2025)))
    device = resolve_device(str(train_config.get("device", "cuda")))
    model = build_model(
        in_channels=int(model_config["in_channels"]),
        base_channels=int(model_config.get("base_channels", 32)),
    ).to(device)
    if model_config.get("pretrained_backbone"):
        load_compatible_weights(model, model_config["pretrained_backbone"])

    loader_options = {
        "batch_size": int(train_config.get("batch_size", 2)),
        "num_workers": int(data_config.get("num_workers", 0)),
        "collate_fn": _collate,
        "pin_memory": device.type == "cuda",
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config.get("learning_rate", 1e-3)),
        weight_decay=float(train_config.get("weight_decay", 1e-4)),
    )
    epochs = int(train_config.get("epochs", 100))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    checkpoint_dir = Path(train_config.get("checkpoint_dir", "checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint_dir / "best_miou.pth"
    best_miou = -1.0

    for epoch in range(1, epochs + 1):
        training = _run_epoch(model, train_loader, optimizer, scaler, config, device, training=True)
        validation = _run_epoch(
            model, validation_loader, optimizer, scaler, config, device, training=False
        )
        scheduler.step()
        record = {
            "epoch": epoch,
            "train": training,
            "validation": validation,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        print(json.dumps(record, sort_keys=True))
        if validation["mean_iou"] > best_miou:
            best_miou = float(validation["mean_iou"])
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "config": config}, best_path
            )
    return best_path


def _bounded_chunks(indices: np.ndarray, xyz: np.ndarray, max_points: int) -> list[np.ndarray]:
    """Recursively split dense blocks along their widest XY axis."""

    if max_points <= 0 or len(indices) <= max_points:
        return [indices]
    subset = xyz[indices, :2]
    axis = int(np.argmax(np.ptp(subset, axis=0)))
    order = indices[np.argsort(subset[:, axis], kind="stable")]
    chunks: list[np.ndarray] = []
    stride = max(int(max_points * 0.9), 1)
    for start in range(0, len(order), stride):
        chunk = order[start : start + max_points]
        if len(chunk):
            chunks.append(np.sort(chunk))
        if start + max_points >= len(order):
            break
    return chunks


def infer_files(
    paths: Sequence[str | Path],
    checkpoint: str | Path,
    output_dir: str | Path,
    config: dict[str, Any],
) -> list[Path]:
    """Predict full scenes with overlap voting and optional mirrored TTA."""

    torch, _, _, tqdm = _dependencies()
    model_config = require_section(config, "model")
    data_config = require_section(config, "data")
    infer_config = require_section(config, "inference")
    device = resolve_device(str(require_section(config, "train").get("device", "cuda")))
    model = build_model(
        in_channels=int(model_config["in_channels"]),
        base_channels=int(model_config.get("base_channels", 32)),
    ).to(device)
    load_compatible_weights(model, checkpoint)
    model.eval()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    feature_properties = tuple(data_config.get("feature_properties", ["intensity"]))
    block_size = float(data_config.get("block_size", 50.0))
    overlap = float(infer_config.get("overlap", 0.5))
    max_points = int(infer_config.get("max_points_per_pass", 800_000))
    flips = [False, True] if bool(infer_config.get("tta_x_flip", True)) else [False]

    with torch.no_grad():
        for path_value in tqdm(paths):
            cloud = read_ply(path_value, require_labels=False)
            score_sum = np.zeros((cloud.size, NUM_CLASSES), dtype=np.float32)
            votes = np.zeros(cloud.size, dtype=np.float32)
            for flip_x in flips:
                working_xyz = cloud.xyz.copy()
                if flip_x:
                    working_xyz[:, 0] *= -1.0
                blocks = sliding_block_indices(
                    working_xyz, block_size=block_size, overlap=overlap
                )
                for block in blocks:
                    for indices in _bounded_chunks(block, working_xyz, max_points):
                        sample = voxelize_arrays(
                            working_xyz,
                            cloud.attributes,
                            None,
                            indices,
                            path=path_value,
                            voxel_size=float(data_config.get("voxel_size", 0.08)),
                            feature_properties=feature_properties,
                            spatial_scale=block_size,
                        )
                        sparse, _ = sparse_batch([sample], device)
                        logits = model(sparse)
                        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
                        score_sum[sample.point_indices] += probabilities[sample.inverse]
                        votes[sample.point_indices] += 1.0
            if np.any(votes == 0):
                raise RuntimeError(f"inference failed to cover every point in {path_value}")
            labels = np.argmax(score_sum / votes[:, None], axis=1)
            written.append(save_prediction(destination / f"{Path(path_value).stem}.npy", labels))
    return written

