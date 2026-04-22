# Server behavior notes

Operational notes for the Robometer eval server at `/evaluate_batch_npy`
as it relates to frames you send from this client. Verified against
the loaded checkpoint (`Qwen/Qwen3-VL-4B-Instruct` base, `Robometer-4B`
heads) and the server code on `main`.

Paths below are relative to the robometer repo root.

For how the training data is constructed (sampling strategies, per-clip
frame count, loss), see [`training_notes.md`](./training_notes.md).

---

## 1. Resolution

### Rule of thumb

**You do not need to reshape on the client side.** Send raw `(T, H, W, C)`
uint8 frames at whatever resolution your renderer produces. The server
downscales before the VLM if needed, and the HF processor upscales tiny
images to a floor.

### Pipeline

1. **Client upload.** Frames go out as-is; no resize.
2. **Server pre-clamp** (`robometer/data/collators/rbm_heads.py:22-42`):

   ```python
   MAX_IMAGE_SIDE   = 480        # bigger side
   MAX_IMAGE_PIXELS = 1024 * 1024  # 1.0 MP area cap

   def _resize_pil(pil, max_side=MAX_IMAGE_SIDE, max_pixels=MAX_IMAGE_PIXELS):
       ...
       if scale < 1.0:
           pil = pil.resize((nw, nh), resample=Image.BICUBIC)
   ```

   Every frame is BICUBIC-downscaled so max side ≤ 480 and total area ≤ 1 MP.
   Smaller frames pass through untouched.

3. **HF Qwen3-VL processor `smart_resize`.** Snaps each image to multiples
   of `patch_size × merge_size = 16 × 2 = 32 px` per side, with
   `shortest_edge ≈ 256 px` as the floor (anything smaller is upscaled).

### Effective working range

| Input resolution | Server behavior       | Image tokens/frame | Notes                            |
| ---------------- | --------------------- | ------------------ | -------------------------------- |
| 128×128          | upscaled to ~256×256  | ~64                | wasteful upscale, functional     |
| **256×256**      | passes through        | **~64**            | **sweet spot** — cheap, in-dist  |
| 384×384          | passes through        | ~144               |                                  |
| 480×480          | passes through        | ~225               | max detail; ~3.5× compute of 256 |
| 640×640+         | downscaled to 480/sid | ~225               | wasted upload bandwidth          |

Token cost per forward pass scales as `T × (H/32) × (W/32)`.

Aspect ratio is preserved — 320×240, 256×192, etc. all work.

### Practical recommendation

- Render sim frames at ~**256×256 uint8** RGB.
- Skip anything > 480 on either side; it gets downscaled away anyway.
- Subsample to `T = 8` frames per rollout (see §2.3) before sending.

---

## 2. Frame count and `use_frame_steps`

### 2.1 Two modes

`client.score(samples, use_frame_steps=...)` selects one of two server
behaviors. Default is `False`.

#### `use_frame_steps=False` — one-shot

```
[f_0, f_1, ..., f_{T-1}] ──► one VLM call ──► progress[T]
```

Send `T` frames, get back a length-`T` per-frame progress array. One
forward pass per sample. Fast. This is what you want for best-of-N.

#### `use_frame_steps=True` — cumulative sliding window

For each time step `i` in `[1..T]`, the server:

1. picks 4 linspace indices in `[0..i-1]`,
2. calls the VLM on those 4 frames,
3. keeps the last output as `progress[i-1]`.

`T` forward passes per sample. Smoother monotone curves, ~T× slower.

### 2.2 Expansion and aggregation, precisely

Expansion (`robometer/evals/eval_server.py:213, 226-231`):

```python
NUM_SUBSAMPLED_FRAMES = 4   # hardcoded

for i in range(1, num_frames + 1):
    indices = np.linspace(0, i - 1, NUM_SUBSAMPLED_FRAMES, dtype=int)
    sub_frames = frames[indices]
```

Aggregation (`robometer/evals/eval_server.py:109-115`):

```python
for i in range(num_frames):
    sub_pred = progress_pred[current_idx]
    if isinstance(sub_pred, list) and len(sub_pred) > 0:
        sample_predictions.append(sub_pred[-1])   # keep only last
```

The model is a per-frame head: feeding it 4 input frames returns 4
progress predictions (one aligned to each input frame). The aggregator
drops three of them and keeps the prediction aligned to the **last**
input frame — which is the "current" frame `i-1` we're labeling. The
other three get re-predicted in later sub-calls with fresh context.

Worked example for `T = 8`:

| labeling frame | indices into original video          | kept output         |
| -------------: | ------------------------------------ | ------------------- |
|              0 | `[0, 0, 0, 0]` — frame 0 ×4          | last → `progress[0]`|
|              1 | `[0, 0, 0, 1]`                       | last → `progress[1]`|
|              2 | `[0, 0, 1, 2]`                       | last → `progress[2]`|
|              3 | `[0, 1, 2, 3]`                       | last → `progress[3]`|
|              4 | `[0, 1, 2, 4]`                       | last → `progress[4]`|
|              5 | `[0, 1, 3, 5]`                       | last → `progress[5]`|
|              6 | `[0, 2, 4, 6]`                       | last → `progress[6]`|
|              7 | `[0, 2, 4, 7]`                       | last → `progress[7]`|

### 2.3 Training vs serving mismatch

The `4` in the eval server is **not** the frame count the model was
trained on. The two disagree:

| Path                                                 | Frames/clip    |
| ---------------------------------------------------- | -------------- |
| **Training**                                         | **8**          |
| **Training-time eval, `use_frame_steps=True`**       | **8**          |
| **Serving `/evaluate_batch_npy`, `False` (default)** | whatever you send |
| **Serving `/evaluate_batch_npy`, `True`**            | **4** hardcoded |

The training cap is `data.max_frames: 8` (from `/model_info`), enforced
at `robometer/data/samplers/base.py:698-701`:

```python
current_frame_count = len(subsampled) if hasattr(subsampled, "__len__") else subsampled.shape[0]
if current_frame_count > self.config.max_frames:
    subsampled, frame_indices_subsample = linspace_subsample_frames(subsampled, self.config.max_frames)
```

Training-time eval clips follow the same `max_frames` trim. Only the
serving `use_frame_steps` path is 4 — a batching-convenience choice so
all T sub-samples have a common tensor shape and memory doesn't blow up
T×.

### 2.4 When to use which

| Goal                                                         | Pick                    |
| ------------------------------------------------------------ | ----------------------- |
| Best-of-N ranking by `progress[-1]`                          | `use_frame_steps=False` |
| Smooth monotone progress curve for mid-rollout decisions     | `use_frame_steps=True`  |
| In-distribution fidelity (matches training most closely)     | `False` with `T = 8`    |
| Lowest latency / compute                                     | `False`                 |

For the physics-sim best-of-N case, default (`False`) with `T = 8`
linspace-picked frames is strictly better: 1 forward pass per sample
instead of 8, and matches training's clip length.

Edge case: `use_frame_steps=True` with `T = 1` is degenerate — the
single frame is replicated 4× and predicted on once. Only meaningful
for `T ≥ 2`.

---

## 3. Quick reference

```python
import numpy as np
from robometer_client import RobometerClient, VideoSample

def pick_8(frames: np.ndarray) -> np.ndarray:
    idx = np.linspace(0, frames.shape[0] - 1, 8, dtype=int)
    return frames[idx]

# Render your rollout at ~256x256 uint8, pick 8 frames, send with
# default use_frame_steps=False for best-of-N.
samples = [
    VideoSample(frames=pick_8(rollout), task=task, id=f"r{i}")
    for i, (rollout, task) in enumerate(zip(rollouts, tasks))
]

with RobometerClient("http://localhost:8000") as c:
    results = c.score(samples)

best = max(results, key=lambda r: float(r.progress[-1]))
```
