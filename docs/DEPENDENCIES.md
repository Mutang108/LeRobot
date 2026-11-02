# Dependency inventory

Choose the task using the [runtime/interface comparison](TASKS.md). LiftCube and PickOrange are current development tracks, not interchangeable installations.

## Current LiftCube runtime

LiftCube uses the separately distributed `isaac-lab:4.5.0-v2.1.1-leisaac-v0.1.2-src` image: Isaac Sim 4.5.0, Isaac Lab v2.1.1 and LeIsaac v0.1.2. Obtain it through the [image guide](../course/liftcube/README.md#运行环境--runtime), which includes the SJTU share link and archive/image checksums. Preserve its bundled Torch/CUDA and source paths; the historical snapshots below are not a replacement for this runtime. PyArrow/PyAV are optional dataset-reading dependencies that may need a compatible install inside the application environment.

## PickOrange runtime

Follow the [PickOrange guide](PICK_ORANGE.md). Its diagnostic launchers expect a separate Isaac Sim/IsaacLab runtime with Python3.11 `venv_sim`/`venv_bc` overlays. The public task source is pinned LeIsaac0.4.0 with IsaacLab2.3.0; the repository does not supply an equally locked matching image/installation. Verify the actual simulator version and imports instead of deriving them from a constraint filename or installing over the LiftCube image.

Kitchen/SO101 assets are external and checksum-listed below. `course/constraints-isaac51.txt` records a conflicting historical dependency experiment, not a supported unified installation recipe. Complete model backends need their own compatible environments and task adapters.

## Rule and planar-control development

The visual rule baseline supports two optional kinematics backends: NumPy/SciPy with a verified fixed SO101 joint-chain JSON, or LeRobot `RobotKinematics` with Placo and a matching URDF. The chain exporter additionally requires the image's USD Python API, but never starts SimulationApp. Both backends require independently validated calibration; see [the rule-baseline guide](SCRIPTED_LIFTCUBE.md).

The optional [planar EE layer](PLANAR_EE_DEVELOPMENT.md) uses NumPy/SciPy and a matching fixed joint chain, without Placo. Use the existing compatible application environment; neither controller requires demonstration data or trained weights. Robot/TCP calibration, camera/table transforms and workspace validation are external inputs, not supplied by an incomplete template. The simulator-pose camera provider is an explicitly opted-in development path, not the unreleased formal calibration SDK.

## Optional VLA backends

Training and policy evaluation use [SwanLab](SWANLAB.md), pinned to `swanlab==0.10.3` by the LiftCube `logging` extra and CPU CI. Install it in the actual training/evaluator environment without replacing bundled simulator Torch/CUDA. Offline mode needs no login; online mode requires the user's own private destination and safe authentication. No credentials or maintainer-specific project bindings are shipped.

The four inference wrappers live under `course/liftcube/src/starter/models/`. Their CPU boundary tests require NumPy, CPU Torch and PyYAML only; they do not import native model libraries or fetch weights. See [VLA interfaces](VLA_INTERFACES.md#cpu检查--cpu-checks) for an isolated test environment and fixed backend revisions.

Full loading uses separate optional environments: pinned LeRobot v0.4.2 for SmolVLA, author-matched VLASH for SO101 PI05, GR00T N1.5, or the pinned SO101 OpenVLA-OFT implementation. These are not interchangeable installations and are not vendored together. Preserve the simulator's existing dependencies; use a validated inference-service boundary when they cannot coexist. Model, tokenizer, processor, statistics and trained head files must be provisioned locally under their own licenses before loading.

Concrete checkpoint/dataset URLs and revisions are listed in [resources](RESOURCES.md). The PickOrange GR00T service and full-episode source port live in [course/pickorange](../course/pickorange/README.md); its audit requires NumPy and h5py, while the simulator and model processes retain separate dependencies. CPU CI installs h5py 3.12.1 and SciPy 1.14.1 for trajectory and controller fixtures.

Native imports can require more than the policy class: VLASH's package initialization imports its declared Reachy SDK dependencies, while OFT imports its RLDS utilities even for inference. OFT uses its own Transformers fork, TensorFlow 2.15/TFDS and `dlimp`; the plain OpenVLA environment is insufficient. A CPU-only dependency check may use TensorFlow CPU 2.15 with NumPy 1.26 in a separate environment. Resolve and pin those dependencies before real loading rather than editing shared dependencies or stubbing missing modules. The loader uses pinned processor classes instead of enabling arbitrary checkpoint Python code.

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
