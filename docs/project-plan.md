# SO-101 End-to-End Robot Manipulation Plan

## Deliverable definition

The goal is a defensible engineering artifact, not a single demo. The project must include:

- a real manipulation task with contact-rich execution and hardware success above 80%
- three policies trained on the same dataset under the same protocol: ACT, Diffusion Policy, and a VLA
- on-device deployment on Jetson with measured latency and control-loop frequency
- a simulation environment that mirrors the real setup closely enough to prototype in
- a writeup that clearly documents failures, not just the successful final run

## Recommended task

Start with a lid-seating task on a jar or container. It is a strong middle ground between simple pick-and-place and difficult insertion.

## Project phases

### Phase 0 — Environment

Requirements:

- Ubuntu 22.04 or 24.04
- NVIDIA driver 570+ / CUDA 12.8+
- `uv` for Python env management
- verify the local GPU is supported by the CUDA stack with `torch.cuda.get_device_capability()`
- for an RTX 3090, expect `(8, 6)`; for Blackwell, expect `(12, 0)`
- LeRobot import works and the CLI responds
- dataset round-trip works
- trivial ACT training over 100 steps completes on GPU
- `pyproject.toml` is fixed with explicit uv indexes and the project is locked with `uv lock`

Important caveats:

- install torch with the cu128+ index before LeRobot to avoid a mismatched GPU stack
- the local GPU is an RTX 3090 (`sm_86`), so there is no Blackwell-specific `sm_120` kernel problem or PTX JIT compatibility trap to solve here
- flash-attn is not the blocker on Ampere; the real constraint is model memory, not a kernel generation mismatch
- choose a single video backend (`torchcodec` or `av`), validate it with a 10-frame round-trip, and keep the choice pinned in the lockfile

#### Windows Phase 0 migration path

A Windows-based prototype path is viable and should be treated as a cost reduction move before the Linux hardware setup. The goal is to keep the migration cheap and avoid a full rewrite later.

Setup sequence:

```powershell
uv venv --python 3.12
.\.venv\Scripts\activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
python -c "import torch; print(torch.cuda.get_device_capability())"
```

On an RTX 3090, this should print `(8, 6)`. On Blackwell it should print `(12, 0)`. The requirement is a working, supported CUDA stack before LeRobot is installed; the exact capability depends on the GPU you have.

FFmpeg is the step that most often bites on Windows. TorchCodec on Windows requires the full-shared FFmpeg build with DLLs. The `essentials` build is not sufficient. Install `ffmpeg-master-latest-win64-gpl-shared` from BtbN's releases, unpack it somewhere permanent, add its `bin` directory to `PATH`, and confirm `ffmpeg -version` responds in a fresh shell. This build also includes `libsvtav1`, which is the encoder LeRobot prefers for dataset video.

Then continue with:

```powershell
uv pip install "lerobot[dataset]" mujoco mink
python -c "import torch; print(torch.__version__)"   # confirm still cu128
```

`lerobot` v0.6.0 keeps the heavy dataset tooling behind the `dataset` extra, so install that explicit extra rather than relying on the default lean install. Choose one video backend and keep it consistent:

```powershell
uv pip install torchcodec
# or
uv pip install av
```

Then validate the backend with a 10-frame round-trip before recording any episodes. The backend is a project choice, not a side effect of the environment. If both `torchcodec` and `av` are present, LeRobot may still pick one at runtime; the project should set the choice explicitly and document it in the README and lockfile.

Renderer lifecycle matters on Windows: MuJoCo renderers hold GL contexts and should be reused instead of created per episode. The safe pattern is to create two renderers once, reuse them across the dataset, and close them explicitly with a context manager:

```python
with mujoco.Renderer(model, 480, 640) as r:
    ...
```

If renderer contexts are created per episode and never closed, you will eventually leak GL state and crash at a time that looks unrelated to the real root cause.

#### Four Windows gotchas

- Enable Developer Mode in Settings → System → For developers. The Hugging Face Hub cache uses symlinks; without this, you get silent full copies and excessive disk churn. Also run `git config --global core.longpaths true`.
- Dataloader workers: Windows uses `spawn`, not `fork`. If `lerobot-train` throws odd multiprocessing errors, set `--num_workers=0`, confirm training works, then raise the value. Sim datasets are small enough that this is not a throughput bottleneck for the first pass.
- Line continuations: LeRobot docs often use `\`; in PowerShell the continuation character is a backtick (`` ` ``) or you can keep the command on a single line.
- Offscreen rendering works on Windows via WGL and `mujoco.Renderer` typically requires no special setup. This is the area that most often looks risky and is usually not the issue.

#### Keep the migration cheap

Three habits reduce future cost:

- use `pathlib` everywhere and avoid hardcoded separators or absolute paths in scripts
- keep all generated artifacts in git: scene XML, demo generator, recorder, and evaluation harness
- push the sim dataset to the Hub as soon as it exists so the Ubuntu machine can start from `LeRobotDataset("you/so101_sim_v1")` with no missing local state

This makes the Ubuntu installation a quick re-run of the same Phase 0 environment against a lockfile rather than a fresh-from-scratch setup.

#### Where the line is

Everything through Phase 1 can run on Windows: sim environment, scripted demos, dataset generation, ACT training, and the evaluation harness. It is also plausible to train Diffusion Policy and SmolVLA on sim data in this setup, which gets a large amount of the later comparison table working before hardware exists.

The switch to Linux should happen when hardware arrives. This is not because teleoperation is impossible on Windows, but because the USB topology debugging in §1.3 has no straightforward Windows story and JetPack flashing must happen on a Linux host anyway.

### Phase 1 — Simulation foundation (MuJoCo)

Goal: close the loop in sim before hardware arrives.

Key steps:

- obtain or build the SO-101 MuJoCo model
- build the scene and approximate camera geometry
- set up IK and scripted motion generation
- generate scripted demos and record them in LeRobot format
- train ACT on the simulation dataset
- run automated evaluation over randomized placements

Exit criteria:

- scripted policy >95% in sim
- 200 scripted episodes logged
- ACT >80% in sim
- eval script produces repeatable success-rate reports

### Phase 2 — Hardware build and calibration

- assemble the two SO-101 arms and label servo IDs
- test each servo individually before installation
- verify leader-follower mapping and calibration
- calibrate both camera intrinsics and extrinsics
- establish a canary routine for drift detection

Exit criteria:

- both arms assembled and stable
- teleop runs smoothly with zero drift
- both cameras stream at stable 30fps with timestamps
- canary routine is defined and passing

### Phase 4A — Instrumentation and automation rig

- load-cell success detector with rigid mounting and threshold calibration
- e-stop watchdog for hardware kill
- automated reset stage for repeatable object placement
- foot pedal annotation input
- common timestamp synchronization across logs

Exit criteria:

- load-cell and threshold validated against 20 hand-placed trials
- e-stop failure path verified
- reset stage repeatable with endstops and seeded random placement
- all logs share a common timebase

### Phase 3 — Teleoperation data collection

- train operator consistency before collection
- collect 100–200 episodes with a single grasp strategy
- inspect every episode visually before acceptance
- log session metadata and hold-out evaluation positions
- push dataset to HF Hub

### Phase 4 — Policy ladder

Train and compare:

1. ACT
2. Diffusion Policy
3. SmolVLA fine-tune

Evaluation protocol must be frozen before training begins:

- 20–30 trials per policy minimum
- same held-out object positions across all models
- binary success criterion defined before rollout starts
- score with load-cell detector and sampled video review
- record failure taxonomy

### Phase 5 — Sim2real (optional)

After the baseline pipeline works, consider Isaac Lab for domain randomization and co-training. Keep it as a separate research extension rather than a prerequisite.

### Phase 6 — Jetson deployment

- flash JetPack 6.2+
- enable MAXN Super mode with active cooling
- measure bare baseline latency
- export and benchmark TensorRT
- profile camera capture, preprocessing, inference, and servo write path
- use async inference + action chunking where feasible
- report actual control-loop frequency and jitter

### Phase 7 — Writeup and artifact packaging

The final writeup must include:

- a task definition and hardware stack
- dataset collection procedure and quality gates
- policy comparison table
- deployment numbers and latency budget
- failure analysis and what was learned
- code, data, and reproducibility notes

## Ordering

1. Buy the SO-101 hardware and cameras
2. Buy the Orin Nano and SSD/PSU
3. Set up workspace and automation hardware
4. Complete Phase 0 and Phase 1 while waiting on part delivery
5. Build hardware and calibrate when parts arrive
6. Collect data and train policies
7. Deploy to Jetson and write up the results

## Success definition

This project is successful only when the pipeline is instrumented, comparable, and real:

- same data for all policies
- same eval protocol for all policies
- measured hardware deployment
- recorded failure modes and drift problems
- reproducibility beyond a single lucky run
