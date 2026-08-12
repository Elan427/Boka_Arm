import shutil
from pathlib import Path

import numpy as np
import torch
from lerobot.datasets import LeRobotDataset

ROOT = Path("data/_roundtrip")
shutil.rmtree(ROOT, ignore_errors=True)

H, W, N = 480, 640, 10
rng = np.random.default_rng(0)

# distinctive per-frame pattern: solid color blocks, easy to compare
frames = np.zeros((N, H, W, 3), dtype=np.uint8)
for i in range(N):
    frames[i, :, :, 0] = i * 25
    frames[i, :, :, 1] = 255 - i * 25
    frames[i, 100:200, 100:200] = 255

states = rng.random((N, 6)).astype(np.float32)

ds = LeRobotDataset.create(
    repo_id="local/roundtrip",
    fps=30,
    root=ROOT,
    robot_type="so101_follower",
    features={
        "observation.state": {"dtype": "float32", "shape": (6,)},
        "action": {"dtype": "float32", "shape": (6,)},
        "observation.images.top": {"dtype": "video", "shape": (H, W, 3)},
    },
    video_backend="pyav",
)

for i in range(N):
    ds.add_frame(
        {
            "observation.state": states[i],
            "action": states[i],
            "observation.images.top": frames[i],
            "task": "roundtrip",
        }
    )

ds.save_episode()
ds.finalize()

print(ds._video_backend)   # or check meta / the dataset attrs in 0.6.1
# reload from disk
rl = LeRobotDataset("local/roundtrip", root=ROOT, video_backend="pyav")
print(f"frames: {len(rl)}")

max_state_err = 0.0
img_errors = []
for i in range(N):
    s = rl[i]
    img = (s["observation.images.top"].permute(1, 2, 0).numpy() * 255).astype(np.float32)
    err = np.abs(img - frames[i].astype(np.float32))
    img_errors.append(err.flatten())
    max_state_err = max(
        max_state_err,
        torch.abs(s["observation.state"] - torch.from_numpy(states[i])).max().item(),
    )

img_errors = np.concatenate(img_errors)
print(f"mean {img_errors.mean():.2f}  p99 {np.percentile(img_errors, 99):.1f}  max {img_errors.max():.1f}")
print(f"max state error : {max_state_err:.2e}  (expect ~0, must be exact)")
