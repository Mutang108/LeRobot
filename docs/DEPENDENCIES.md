# Dependency inventory

## Current LiftCube runtime

LiftCube uses the separately distributed `isaac-lab:4.5.0-v2.1.1-leisaac-v0.1.2-src` image: Isaac Sim 4.5.0, Isaac Lab v2.1.1 and LeIsaac v0.1.2. Obtain it through the [image guide](../course/liftcube/README.md#运行环境--runtime), which includes the SJTU share link and archive/image checksums. Preserve its bundled Torch/CUDA and source paths; the historical snapshots below are not a replacement for this runtime. PyArrow/PyAV are optional dataset-reading dependencies that may need a compatible install inside the application environment.

## Rule and planar-control development

The visual rule baseline supports two optional kinematics backends: NumPy/SciPy with a verified fixed SO101 joint-chain JSON, or LeRobot `RobotKinematics` with Placo and a matching URDF. The chain exporter additionally requires the image's USD Python API, but never starts SimulationApp. Both backends require independently validated calibration; see [the rule-baseline guide](SCRIPTED_LIFTCUBE.md).

The optional [planar EE layer](PLANAR_EE_DEVELOPMENT.md) uses NumPy/SciPy and a matching fixed joint chain, without Placo. Use the existing compatible application environment; neither controller requires demonstration data or trained weights. Robot/TCP calibration, camera/table transforms and workspace validation are external inputs, not supplied by an incomplete template. The simulator-pose camera provider is an explicitly opted-in development path, not the unreleased formal calibration SDK.

## Optional VLA backends

The four inference wrappers live under `course/liftcube/src/starter/models/`. Their CPU boundary tests require NumPy, CPU Torch and PyYAML only; they do not import native model libraries or fetch weights. See [VLA interfaces](VLA_INTERFACES.md#cpu检查--cpu-checks) for an isolated test environment and fixed backend revisions.

Full loading uses separate optional environments: pinned LeRobot v0.4.2 for SmolVLA, author-matched VLASH for SO101 PI05, GR00T N1.5, or the pinned SO101 OpenVLA-OFT implementation. These are not interchangeable installations and are not vendored together. Preserve the simulator's existing dependencies; use a validated inference-service boundary when they cannot coexist. Model, tokenizer, processor, statistics and trained head files must be provisioned locally under their own licenses before loading.

## Complete source snapshots

| Project | Fixed official source | Included |
| --- | --- | --- |
| LeRobot | https://github.com/huggingface/lerobot/tree/58f70b6bd370864139a3795ac3497a9eae8c42d5 | Entire tracked tree: policies, training, data, hardware, tests, docs, package metadata |
| LeIsaac | https://github.com/LightwheelAI/leisaac/tree/24d3bcd3f1e4585740fc79921782c41617237812 | Entire tracked tree, with expanded IsaacLab dependency |
| IsaacLab | https://github.com/isaac-sim/IsaacLab/tree/3c6e67bb5c7ada942a6d1884ab69338f57596f77 | Entire framework, controllers, tasks, tests, docs and build files |
| Integration utilities | course/scripts and scripts | Independent diagnostic tools, constraint examples, source audit and fixture recovery |

ACT, SmolVLA and SAC/HIL-SERL implementations are already in LeRobot; separate repositories are not needed for those included implementations. Optional backends/hardware extras remain declared in original package metadata; this is not a claim they are all installed.

## External components

| Component | Source | Treatment |
| --- | --- | --- |
| Isaac Sim / runtime | nvcr.io/nvidia/isaac-lab:2.3.0 | Obtain official image under its terms; verify actual runtime after pull. No image binaries in Git |
| Python dependencies | Upstream pyproject/extension metadata; course/constraints-*.txt | Resolve in compatible isolated application environments, retaining image-shared torch/CUDA |
| PickOrange kitchen / SO101 USD | manifests/external_assets.json and course/scripts/prepare_assets.py | Official URLs, sizes and SHA256; download to approved asset storage |
| LFS test fixtures | manifests/lfs_objects.json | All 45 binaries are included and verified; recovery tool supplied |
| Pretrained policies | Upstream model docs/cards | Select version and record digest before use; no arbitrary weights claimed as included |
| Real-robot calibration | LeRobot hardware modules and optional dependencies | Configure for the actual device; private calibrations not published |
| Formal task environment/scoring | External task provider | Supplied separately; no invented implementation of undisclosed protocols |

This is a complete snapshot of the three selected application projects, not an offline mirror of every Python/CUDA/system package. Additional repositories become dependencies when actually selected and pinned. RoboTwin/rl-garden are not runtime dependencies here.
