# Training data and sampling notes

Background on how the Robometer training data is assembled — useful for
matching the serving distribution (what `T`, what kind of motion, what
tasks) and for sanity-checking unexpected progress curves at inference
time. Verified against the loaded checkpoint (`Qwen/Qwen3-VL-4B-Instruct`
base, `Robometer-4B` heads) and the training code on `main`.

Paths below are relative to the robometer repo root.

For how these choices show up at serving time (resolution pipeline, the
`use_frame_steps` mode, the training/serving frame-count mismatch),
see [`server_behavior.md`](./server_behavior.md).

---

## 1. Per-clip frame count

**Every training clip is exactly `T = max_frames = 8` frames.** From
`/model_info`: `data.max_frames: 8`, `data.min_frames_per_trajectory: 5`.

Enforcement, in order:

1. **Drop-too-short filter** (`robometer/data/datasets/base.py:83`,
   `:98-104`): trajectories with fewer than 5 raw frames are excluded
   from the dataset before training.
2. **Trim if long** (`robometer/data/samplers/base.py:698-705`):
   segments with > 8 frames are uniform-linspace trimmed to 8.
3. **Pad if short** (`robometer/data/samplers/base.py:707-715`,
   `robometer/data/datasets/helpers.py:210-247`): segments with < 8
   frames are padded by repeating the last frame (and the last target
   progress value). A per-sample padding mask is built from the real
   length so padded positions don't contribute to the CE loss.

Net effect: the model always sees a `[B, 8, ...]` tensor during
training, but the *information content* can be < 8 frames of real
motion when padding kicks in.

---

## 2. Subsampling is segment-based, not globally uniform

There is no "always linspace-subsample the whole trajectory to 8" step
in training. Each clip is built in two random steps, then uniform-trimmed
only if it overflows.

**Step 1 — pick 3 random pivots.** `_get_subsample_indices`
(`robometer/data/samplers/base.py:476-574`) draws 3 distinct random
frame indices from the full trajectory:

```python
# robometer/data/samplers/base.py:529-530
# Sample three random distinct frames
frame_indices = sorted(random.sample(range(num_frames_total), 3))
```

These become `(start, middle, end)` — and the *ordering* assigned to
them is what defines the strategy (forward / rewind / reverse). This is
**random pivot selection, not uniform**.

**Step 2 — enumerate every frame along the path.**
`get_segment_indices_with_middle`
(`robometer/data/datasets/helpers.py:399-490`) walks `start → middle → end`
including every intermediate frame in order (forward or backward as
needed) and concatenates the two half-segments.

**Step 3 — uniform-trim to 8 if needed.**
`robometer/data/samplers/base.py:698-705`:

```python
current_frame_count = len(subsampled) if hasattr(subsampled, "__len__") else subsampled.shape[0]
if current_frame_count > self.config.max_frames:
    subsampled, frame_indices_subsample = linspace_subsample_frames(subsampled, self.config.max_frames)
```

`linspace_subsample_frames` (`robometer/data/datasets/helpers.py:293-355`)
picks evenly-spaced indices and **forces the first and last to be the
segment endpoints**. So within a segment the spacing is uniform, but
the segment itself is a random slice of the full trajectory.

Implication: the model was trained on clips with varied *temporal
coverage* (sometimes the whole video evenly sampled, sometimes a
zoomed-in slice of just the first third, sometimes reversed), not just
one canonical "uniform 8 over the whole rollout".

### Worked examples — how linspace interacts with the middle

Important: `linspace_subsample_frames` operates on the *flat enumerated
path*, not on the trajectory. "Uniform" is along path position. The
split of samples between the pre- and post-middle halves is
**proportional to segment lengths**, not fixed at 4-and-4.

All three runs below use an N=20 trajectory, and all compute `target_progress`
as `absolute_wrt_total_frames` (i.e. `(idx + 1) / 20`).

**FORWARD — `start=2, middle=9, end=17`**

```
path (16 elems): [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
linspace pos   : [0, 2, 4, 6,  9, 11, 13, 15]
final frames   : [2, 4, 6, 8, 11, 13, 15, 17]
target progress: [0.15, 0.25, 0.35, 0.45, 0.60, 0.70, 0.80, 0.90]  # monotone up
```

Path is just contiguous — middle is a point on a straight ramp; it has
no structural role.

**REWIND — `start=2, end=9, middle=17` (peak at middle)**

```
path (24 elems):
  ascent  2→17 : [2..17]                           (16 frames)
  descent 17→9 : [16..9]  (middle deduped)         (8 frames)
  combined     : [2,3,...,16,17,16,15,...,10,9]

linspace pos  : [0,  3,  7, 10, 13, 16, 20, 23]
final frames  : [2,  5,  9, 12, 15, 16, 12,  9]
target progress: [0.15, 0.30, 0.50, 0.65, 0.80, 0.85, 0.65, 0.50]  # peak then down
```

Two things to notice:
- The split is **5-to-3, not 4-and-4**: segment-1 (16 frames, ascent)
  gets 5 linspace samples; segment-2 (8 frames, descent) gets 3.
  Samples are allocated proportionally to segment length.
- **Frames can repeat**: frame 12 appears at positions 3 and 6 of the
  final 8 (once on the way up, once on the way down) and carries
  different targets (0.65 and 0.65 happen to match here, but in general
  targets come from absolute index and only coincide for rewind-symmetry
  frames).

**REVERSE — `start=17, middle=9, end=2`**

```
path (16 elems): [17,16,15,14,13,12,11,10, 9, 8, 7, 6, 5, 4, 3, 2]
linspace pos  : [0,  2,  4,  6,  9, 11, 13, 15]
final frames  : [17, 15, 13, 11, 8, 6, 4, 2]
target progress: [0.90, 0.80, 0.70, 0.60, 0.45, 0.35, 0.25, 0.15]  # monotone down
```

Same mechanics as forward, path is enumerated in descending order.

**General rule.** After enumeration, path length
`L = |start→middle| + |middle→end| − 1` (middle deduped). Samples on
the pre-middle half ≈ `round(8 × |start→middle| / L)`. The middle
frame itself may or may not land on a linspace position — it's
incidental.

---

## 3. Yes — rewind and reverse are used

For progress training, one of four strategies is picked per sample with
equal probability (`progress_strategy_ratio: [1.0, 1.0, 1.0, 1.0]`,
order at `robometer/data/samplers/progress.py:82-94`):

| Strategy                      | Probability | What it does                                              |
| ----------------------------- | ----------- | --------------------------------------------------------- |
| **DIFFERENT_TASK_INSTRUCTION** | 25%        | Frames from a different task; target progress = `[0.0]*T` |
| **FORWARD_PROGRESS**          | 25%         | `start < middle < end` — normal forward motion            |
| **REVERSE_PROGRESS**          | 25%         | `end < middle < start` — trajectory played backwards      |
| **REWIND**                    | 25%         | `start < end < middle` — forward, then backtrack          |

Preference training uses its own four-way split with REWIND and
REVERSE_PROGRESS as two of the strategies (see
`robometer/data/samplers/README.md` in the robometer repo).

**Progress labels match the pivot ordering** (computed from
`frame_indices` passed through `compute_progress_from_segment`,
`robometer/data/datasets/helpers.py:613-657`), so reverse clips get
decreasing targets and rewind clips get non-monotone targets — the
model is explicitly trained to predict progress going down.

---

## 4. Loss

Per-frame 10-way cross-entropy, averaged over the 8-frame clip.

`robometer/trainers/rbm_heads_trainer.py:2129-2141`:

```python
progress_pred_flat   = progress_pred.view(batch_size * seq_len, num_bins)  # [B*T, 10]
target_bins_flat     = target_bins.view(batch_size * seq_len)              # [B*T]
...
loss_per_sample_flat = F.cross_entropy(
    progress_pred_flat, target_bins_flat, reduction="none"
)   # [B*T]
```

Then masked/averaged (`:2186`):

```python
progress_loss = masked_loss.mean(dim=1).sum(dim=0) / (mask.sum() + 1e-8)
```

Target is continuous progress in `[0, 1]` bucketed into 10 bins via
`convert_continuous_to_discrete_bins` before loss. Padded positions
are excluded via the mask.

---

## 5. What this means for inference

- **Reverse and non-monotone progress are *expected* outputs.** If you
  feed a partial-failure rollout where the robot drops what it was
  holding or backs away, the model can legitimately return a decreasing
  progress curve. It saw that at training time.
- **Out-of-distribution frame counts.** Training was always `T = 8` (with
  padding). For one-shot serving (`use_frame_steps=False`) you control
  `T`; pick 8 for the closest match. Serving with `use_frame_steps=True`
  uses 4 per sub-call which is *not* what training saw — see
  [`server_behavior.md`](./server_behavior.md) §2.3.
- **Uniform-linspace subsampling from the client is a safe default.**
  It's what step 3 of training does when a segment overflows. Picking
  the first 8 frames, last 8 frames, or a random 8-frame slice also
  appears in training (as random pivot segments).
- **Short rollouts.** If your rollout has fewer than 8 frames, you can
  either pad by repeating the last frame (mirrors training) or just
  send fewer — the server's `/evaluate_batch_npy` accepts any `T ≥ 1`.
  Training's `min_frames_per_trajectory: 5` suggests ≥ 5 real frames
  for in-distribution behavior.
