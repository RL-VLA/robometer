#!/usr/bin/env python3
"""
Pi0 GIM T-shirt dagger dataset loader for Robometer model training.

Expected directory structure:
    <dataset_path>/
        good/       -> successful rollouts
        dagger/     -> successful dagger rollouts
        failure/    -> failed rollouts
        positioning/  -> skipped
        discarded/    -> skipped

Each pkl is a list of timestep dicts with keys:
    timestamp, source, qpos, qpos_target, gripper_target, grip_active, image

image: {'top': {'color': bytes}, 'left_wrist': {'color': bytes}, 'right_wrist': {'color': bytes}}

Each camera view is emitted as a separate trajectory.
"""

import glob
import io
import pickle
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from dataset_upload.helpers import generate_unique_id

TASK_DESCRIPTION = "Load the t-shirt onto the small table"

# splits to include and their quality labels
SPLIT_QUALITY: dict[str, str] = {
    "good": "successful",
    "dagger": "successful",
    "failure": "failure",
}

CAMERAS = ["top", "left_wrist", "right_wrist"]


class ClothFrameLoader:
    """Lazy loader: reads the pkl and decodes one camera's frames only when called."""

    def __init__(self, pkl_path: str, camera: str):
        self.pkl_path = pkl_path
        self.camera = camera

    def __call__(self) -> np.ndarray | None:
        with open(self.pkl_path, "rb") as f:
            timesteps: list[dict] = pickle.load(f)
        frames = []
        for step in timesteps:
            try:
                jpeg_bytes = step["image"][self.camera]["color"]
                img = Image.open(io.BytesIO(jpeg_bytes)).convert("RGB")
                frames.append(np.array(img, dtype=np.uint8))
            except (KeyError, Exception):
                continue
        if not frames:
            return None
        return np.stack(frames, axis=0)  # (T, H, W, 3)


def load_cloth_dataset(
    dataset_path: str,
    max_trajectories: int | None = None,
) -> dict[str, list[dict]]:
    """Load cloth loading dataset and organize by task.

    Args:
        dataset_path: Path to the dataset root (contains good/, dagger/, failure/ subdirs).
        max_trajectories: Maximum total trajectories to load (None for all).

    Returns:
        Dictionary mapping task name to list of trajectory dicts.
    """
    print(f"Loading cloth loading dataset from: {dataset_path}")
    base = Path(dataset_path).resolve()
    if not base.exists():
        raise FileNotFoundError(f"Cloth dataset path not found: {base}")

    task_data: dict[str, list[dict]] = {}
    total = 0

    for split, quality_label in SPLIT_QUALITY.items():
        split_dir = base / split
        if not split_dir.exists():
            print(f"  Skipping missing split: {split}")
            continue

        pkl_files = sorted(split_dir.glob("*.pkl"))
        print(f"  {split}: {len(pkl_files)} rollouts -> quality_label={quality_label}")

        for pkl_path in tqdm(pkl_files, desc=f"  Loading {split}"):
            if max_trajectories is not None and max_trajectories != -1 and total >= max_trajectories:
                break

            try:
                with open(pkl_path, "rb") as f:
                    timesteps: list[dict] = pickle.load(f)
            except Exception as e:
                print(f"    Warning: failed to load {pkl_path.name}: {e}")
                continue

            # Build flat actions array: left_arm (7) + right_arm (7) = (T, 14)
            actions_list = []
            for step in timesteps:
                try:
                    la = step["qpos"]["left_arm"]
                    ra = step["qpos"]["right_arm"]
                    actions_list.append(la + ra)
                except (KeyError, TypeError):
                    actions_list.append([0.0] * 14)
            actions = np.array(actions_list, dtype=np.float32)

            # One rollout_id shared across all camera views of this pkl file
            rollout_id = generate_unique_id()

            # Build one trajectory per camera view (lazy frame loading)
            for camera in CAMERAS:
                trajectory = {
                    "id": generate_unique_id(),
                    "rollout_id": rollout_id,
                    "frames": ClothFrameLoader(str(pkl_path), camera),
                    "actions": actions,
                    "is_robot": True,
                    "task": TASK_DESCRIPTION,
                    "quality_label": quality_label,
                    "partial_success": 1.0 if quality_label == "successful" else 0.0,
                    "data_source": f"cloth_{split}_{camera}",
                }
                task_data.setdefault(TASK_DESCRIPTION, []).append(trajectory)
                total += 1

    total_trajs = sum(len(v) for v in task_data.values())
    print(f"Loaded {total_trajs} trajectories ({total_trajs // len(CAMERAS)} rollouts × {len(CAMERAS)} cameras)")
    return task_data
