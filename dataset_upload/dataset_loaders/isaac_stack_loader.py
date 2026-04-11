#!/usr/bin/env python3
"""
Isaac-Lab Stack Cube dataset loader for Robometer model training.

Expected input: one or more HDF5 files with structure:
    data/
        demo_0/
            obs/
                table_cam  (T, H, W, 3) uint8
                wrist_cam  (T, H, W, 3) uint8
            actions        (T, 7)
        demo_1/ ...

Each camera view is emitted as a separate trajectory.
"""

from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from dataset_upload.helpers import generate_unique_id

TASK_DESCRIPTION = "Stack the cubes on top of each other"

CAMERAS = ["table_cam", "wrist_cam"]


class IsaacStackFrameLoader:
    """Lazy loader: reads both camera views side-by-side from HDF5 on demand."""

    def __init__(self, hdf5_path: str, demo_key: str):
        self.hdf5_path = hdf5_path
        self.demo_key = demo_key  # e.g. "demo_0"

    def __call__(self) -> np.ndarray | None:
        views = []
        with h5py.File(self.hdf5_path, "r") as f:
            for camera in CAMERAS:
                path = f"data/{self.demo_key}/obs/{camera}"
                if path not in f:
                    return None
                frames = f[path][:]
                if frames.dtype != np.uint8:
                    frames = frames.astype(np.uint8)
                views.append(frames)
        # Concatenate side-by-side: (T, H, W*2, 3)
        return np.concatenate(views, axis=2)


def load_isaac_stack_dataset(
    dataset_path: str,
    max_trajectories: int | None = None,
) -> dict[str, list[dict]]:
    """Load Isaac Stack Cube HDF5 dataset.

    Args:
        dataset_path: Path to a directory containing *.hdf5 files, or a single .hdf5 file.
        max_trajectories: Maximum total trajectories to load (None for all).

    Returns:
        Dictionary mapping task description to list of trajectory dicts.
    """
    print(f"Loading Isaac Stack dataset from: {dataset_path}")
    base = Path(dataset_path).resolve()
    if not base.exists():
        raise FileNotFoundError(f"Path not found: {base}")

    if base.is_file() and base.suffix == ".hdf5":
        hdf5_files = [base]
    else:
        hdf5_files = sorted(base.glob("*.hdf5"))

    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files found in: {base}")

    print(f"  Found {len(hdf5_files)} HDF5 file(s)")

    task_data: dict[str, list[dict]] = {}
    total = 0

    for hdf5_path in tqdm(hdf5_files, desc="Loading Isaac Stack files"):
        with h5py.File(hdf5_path, "r") as f:
            if "data" not in f:
                print(f"  Skipping {hdf5_path.name}: no 'data' group")
                continue
            demo_keys = list(f["data"].keys())

        for demo_key in demo_keys:
            if max_trajectories is not None and max_trajectories != -1 and total >= max_trajectories:
                break

            with h5py.File(hdf5_path, "r") as f:
                demo = f[f"data/{demo_key}"]
                actions_key = "actions" if "actions" in demo else "obs/actions"
                actions = demo[actions_key][:].astype(np.float32)

            trajectory = {
                "id": generate_unique_id(),
                "rollout_id": generate_unique_id(),
                "frames": IsaacStackFrameLoader(str(hdf5_path), demo_key),
                "actions": actions,
                "is_robot": True,
                "task": TASK_DESCRIPTION,
                "quality_label": "successful",
                "partial_success": 1.0,
                "data_source": "isaac_stack_combined",
            }
            task_data.setdefault(TASK_DESCRIPTION, []).append(trajectory)
            total += 1

    print(f"Loaded {total} trajectories ({total} demos, both cameras side-by-side)")
    return task_data
