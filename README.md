# SO-101 Manipulation Project

This workspace captures the end-to-end plan for a real robot manipulation project using a dual SO-101 leader/follower setup, with a full pipeline from teleoperation to imitation learning, VLA fine-tuning, and Jetson deployment.

## Project goal

Build a defensible manipulation system for a contact-rich task, with a real dataset, measured metrics, and honest failure analysis. The target is not simply "a robot can pick up something", but a repeatable research artifact with:

- a real task operating on hardware
- three policies trained on the same data: ACT, Diffusion Policy, and a VLA
- comparable evaluation under one protocol
- measured Jetson deployment performance
- a simulation foundation for prototyping and debugging

## Recommended task

Start with lid seating on a jar or container. It is a medium-difficulty task that is contact-rich but tolerates small errors and is realistic for the first full pipeline.

## Phase overview

1. Phase 0: environment setup and hardware-aware Python stack
2. Phase 1: MuJoCo sim foundation and scripted demonstrations
3. Phase 2: hardware build and calibration
4. Phase 4A: instrumentation and automation rig
5. Phase 3: teleoperation data collection
6. Phase 4: policy ladder (ACT, Diffusion Policy, SmolVLA)
7. Phase 5: optional sim2real deepening
8. Phase 6: Jetson deployment
9. Phase 7: beyond imitation and writeup

## Workspace structure

- docs/
  - project-plan.md: full implementation plan and scope
  - progress-tracker.md: phase checklist
  - evaluation-protocol.md: standardized policy evaluation procedure
- scripts/
  - setup-env.sh: environment bootstrap template
  - run_canary.sh: calibration canary runner template
- README.md
- pyproject.toml: pinned Python environment template

## Quick start

1. Review the implementation plan in docs/project-plan.md.
2. Configure the Python environment and verify the GPU stack.
3. Begin Phase 0 before hardware arrives.
4. Build the MuJoCo sim loop in parallel while hardware is shipping.
5. Keep all training and evaluation logs tied to a single reproducible dataset and protocol.

## Environment setup

### Video backend

Phase 0 decision (v0.6.1): **pinned to PyAV (`video_backend="pyav"`)**

- **Reason:** Implicit torchcodec import probing on every dataset open requires shared FFmpeg DLLs. Static FFmpeg builds (most common) fail silently without helpful error messages. PyAV was verified working on first attempt.
- **Trade-off:** torchcodec offers better decode throughput during training, but the bottleneck on a 10-frame sim dataset is not video decoding.
- **Next step:** If observation video quality becomes a limiter in real deployment (validate via scaling curves), we can lower CRF from LeRobot's default (CRF 30) once custom video encoding is stable (available in 0.6.0+). Revisit torchcodec's shared FFmpeg build only if training throughput becomes the limiting factor.

**Dataset and training code must explicitly specify the backend:**

```python
# in data loading
LeRobotDataset("local/roundtrip", root=ROOT, video_backend="pyav")

# in lerobot-train CLI: add --video_backend=pyav
```

## Core principles

- Sim-first, depth-first, no artificial deadline.
- Data quality beats model sophistication.
- One dataset, three policies, one eval protocol.
- Measure deployment latency honestly.
- Record what fails, not just what succeeds.

## Project status

See the phase trackers and evaluation documents for the live checklist.
