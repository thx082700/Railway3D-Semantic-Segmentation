"""Canonical class contract for the urban and rural WHU-Railway3D benchmarks."""

from __future__ import annotations

CLASS_NAMES: tuple[str, ...] = (
    "rails",
    "track_bed",
    "masts",
    "support_devices",
    "overhead_lines",
    "fences",
    "poles",
    "vegetation",
    "buildings",
    "ground",
    "others",
)
NUM_CLASSES = len(CLASS_NAMES)
LABEL_TO_NAME = dict(enumerate(CLASS_NAMES))
NAME_TO_LABEL = {name: label for label, name in LABEL_TO_NAME.items()}
IGNORE_LABEL = -1


def validate_labels(labels, *, allow_ignore: bool = True) -> None:
    """Raise when an array violates the competition label range."""

    import numpy as np

    values = np.asarray(labels)
    if values.size == 0:
        return
    minimum = int(values.min())
    maximum = int(values.max())
    allowed_minimum = IGNORE_LABEL if allow_ignore else 0
    if minimum < allowed_minimum or maximum >= NUM_CLASSES:
        raise ValueError(
            f"labels must be in [{allowed_minimum}, {NUM_CLASSES - 1}], "
            f"observed [{minimum}, {maximum}]"
        )

