# Progress Tracker

## Phase 0 — Environment

- [ ] Ubuntu + driver 570+ / CUDA 12.8+
- [ ] torch with verified sm_120 support
- [ ] LeRobot installed without downgrading torch
- [ ] SDPA confirmed as attention backend
- [ ] Dataset round-trip verified
- [ ] Environment locked and reproducible

## Phase 1 — Simulation

- [ ] SO-101 MJCF loaded, joint limits verified
- [ ] Scene built with planned camera poses
- [ ] IK working via mink
- [ ] Scripted demo policy >95%
- [ ] 200 episodes in LeRobot format
- [ ] ACT trained, >80% in sim
- [ ] Automated eval harness

## Phase 2 — Hardware

- [ ] Arms assembled, servo IDs labeled
- [ ] Calibration complete, teleop smooth
- [ ] Cameras calibrated and bolted down
- [ ] Canary routine established

## Phase 4A — Instrumentation

- [ ] Load cell mounted, tared, calibrated, threshold validated
- [ ] E-stop watchdog built and failure path tested
- [ ] Reset stage repeatable, endstops fitted, RNG seeded and logged
- [ ] Annotation pedal + OLED status
- [ ] Common timebase across all logs, sync verified

## Phase 3 — Data

- [ ] 100–200 episodes, single strategy
- [ ] Every episode visually reviewed
- [ ] Pushed to HF Hub
- [ ] Held-out eval positions defined

## Phase 4 — Policies

- [ ] ACT trained and evaluated
- [ ] Diffusion Policy trained and evaluated
- [ ] SmolVLA fine-tuned and evaluated
- [ ] Comparison table complete
- [ ] Data scaling curve plotted

## Phase 5 — Sim2real (optional)

- [ ] Isaac Lab environment
- [ ] Domain randomization
- [ ] Co-training experiment

## Phase 6 — Deployment

- [ ] JetPack + Super mode
- [ ] TensorRT export
- [ ] Async inference + action chunking
- [ ] Measured Hz, jitter, and quantization delta

## Phase 7 — Writeup

- [ ] Results, failure taxonomy, and what didn't work
- [ ] Video of best and worst rollouts
- [ ] Code and dataset public
