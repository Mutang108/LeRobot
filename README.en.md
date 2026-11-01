<p align="center">
  <a href="assets/lerobot-tabletop-pushing-bilingual.svg">
    <img src="assets/lerobot-tabletop-pushing-bilingual.svg?v=20261003-pdf-aligned" width="960" alt="Bilingual LeRobot tabletop manipulation poster">
  </a>
</p>

<h1 align="center">LeRobot: Reinforcement Learning Project</h1>
<h3 align="center">Visual Tabletop Manipulation: From Simulation to SO-101 Deployment</h3>

<p align="center">
  Visual pick-and-place, imitation learning and reinforcement learning with LeRobot, LeIsaac and the SO-101.
</p>

<p align="center">
  <strong>English</strong> | <a href="README.md">简体中文</a>
</p>

<p align="center">
  <a href="#quick-start">Quick start</a> ·
  <a href="#repository-layout">Repository layout</a> ·
  <a href="#development-path">Development path</a> ·
  <a href="#documentation">Documentation</a> ·
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

---

Start with PickOrange to learn the data/control pipeline, establish training and evaluation on LiftCube, then align with the course multi-object task and real-robot deployment.

## Current scope

- **Development environment:** SO-101 LiftCube, with environment adapters, demonstration loading, BC, reference PPO and evaluation entry points. See the [LiftCube guide](course/liftcube/README.md) for the matching image and runtime.
- **VLA interfaces:** [Single-front wrappers for SmolVLA, PI05, GR00T and OpenVLA-OFT](docs/VLA_INTERFACES.md) have passed CPU interface tests; full-model and GPU/simulator acceptance remain pending. ACT and VLA training integration are in development.
- **Rule baseline:** [LiftCube visual grasping script](docs/SCRIPTED_LIFTCUBE.md), combining RGB localization and inverse kinematics as a development reference for later calibration and task integration. Simulated grasps have succeeded; cross-scene robustness still needs improvement and independent calibration is required.
- **Target task:** extend cube grasping/lifting to visual multi-object pick-and-place, evaluate using course-provided environments and scoring, then deploy on hardware.

Current LiftCube uses a single front RGB stream and six-joint control; see the [interface guide](course/liftcube/README.md#接口--interface) for inputs, actions and timing. Policies use images and permitted proprioception; object/target ground truth is reserved for training rewards or offline analysis.

An optional [planar EE development interface](docs/PLANAR_EE_DEVELOPMENT.md) provides XY control at a fixed reference height and approach axis; physical height tracking is not yet accepted. It is a self-developed execution layer, not the official course IK; existing joint policies are not directly compatible.

## Quick start

Clone and check the repository (Git and Python 3.11+ required):

```bash
git clone -c core.longpaths=true -c core.autocrlf=false https://github.com/Nole326/LeRobot.git
cd LeRobot
python scripts/check_project.py
```

This check starts neither simulation nor training. To run LiftCube, obtain the matching image from the [runtime guide](course/liftcube/README.md#运行环境--runtime) and execute task commands inside the container.

The rule baseline needs no demonstrations or trained weights, but does require matching robot/camera calibration. Start with the offline calibration checks in the [rule-runner guide](docs/SCRIPTED_LIFTCUBE.md). Prepare data and interfaces separately for BC/PPO using the [training guide](course/liftcube/TRAINING.md).

See [baseline notes](docs/BASELINE.md) and the [changelog](CHANGELOG.md) for versions. For development, branch from current `main` and follow the [contribution guide](CONTRIBUTING.md).

## Repository layout

Task code lives in `course/liftcube/src/starter/`, organized by interfaces, data, algorithms and evaluation:

```text
.
├── course/liftcube/
│   ├── src/starter/
│   │   ├── interfaces/        # Observation, action, policy and model contracts
│   │   ├── envs/              # Simulator, control and terminal handling
│   │   ├── data/              # Demonstration reading, conversion, validation and splits
│   │   ├── algorithms/        # BC, RL and reference PPO
│   │   ├── training/          # Configuration-driven training entry points
│   │   ├── evaluation/        # Shared evaluation and policy loading
│   │   ├── rewards/           # Training rewards
│   │   └── logging/           # Metric recording
│   ├── configs/              # Current task configurations
│   └── patches/              # Reversible changes to the course starter
├── course/scripts/           # Standalone diagnostics
├── configs/                  # Public experiment-plan templates
├── scripts/                  # Repository, provenance and plan checks
├── tests/                    # First-party regression tests
├── third_party/              # Pinned upstream sources
├── manifests/                # Source, version and hash inventories
├── docs/                     # Architecture, interfaces, runtime and reproducibility
└── assets/                   # Poster and project artwork
```

See [architecture](docs/ARCHITECTURE.md) for layer ownership and model integration. Obtain images, datasets and weights separately; their sources and versions are listed in [dependencies](docs/DEPENDENCIES.md).

## Development path

1. Validate control interfaces, demonstration data and replay.
2. Establish rule-based, ACT and compact-VLA baselines, then compare other VLAs on the task.
3. Train with RL, explore direct fine-tuning and residuals, and run ablations.
4. Evaluate seen, compositional and unseen conditions under a shared protocol, then align hardware deployment.

BC and pretrained VLAs provide initialization or comparisons; RL is a separate training stage. Report success rate, completion time, wrong/missed grasps and placement error. Task environments define the success criteria and horizons.

See the [RLinf assessment](docs/RLINF.md) for VLA reinforcement-learning references. RLinf is not yet integrated into this project's runtime.

### Upstream versions

These are historical source snapshots. Current LiftCube separately uses the course's Isaac Sim 4.5.0 / Isaac Lab v2.1.1 / LeIsaac v0.1.2 runtime.

| Project | Historical source version | Pinned commit |
| --- | --- | --- |
| LeRobot | v0.4.2 | [58f70b6](https://github.com/huggingface/lerobot/tree/58f70b6bd370864139a3795ac3497a9eae8c42d5) |
| LeIsaac | 0.4.0 | [24d3bcd](https://github.com/LightwheelAI/leisaac/tree/24d3bcd3f1e4585740fc79921782c41617237812) |
| IsaacLab | v2.3.0 | [3c6e67b](https://github.com/isaac-sim/IsaacLab/tree/3c6e67bb5c7ada942a6d1884ab69338f57596f77) |

See the [import inventory](docs/IMPORT_MANIFEST.md) for source provenance and integrity information.

## Documentation

- [Development and stage acceptance](docs/GETTING_STARTED.md)
- [LiftCube source, image and runtime](course/liftcube/README.md)
- [Visual rule reference, calibration and migration](docs/SCRIPTED_LIFTCUBE.md)
- [Planar EE development interface and limitations](docs/PLANAR_EE_DEVELOPMENT.md)
- [Architecture, extension points and collaboration](docs/ARCHITECTURE.md)
- [RLinf benefits and integration boundaries](docs/RLINF.md)
- [Observation, action, data and recovery contracts](docs/INTERFACE_CONTRACTS.md)
- [Pre-training validation and limitations](docs/TRAINING_VALIDATION.md)
- [Dependencies and external assets](docs/DEPENDENCIES.md)
- [Deployment and reproducibility](docs/REPRODUCIBILITY.md)
- [Source import inventory](docs/IMPORT_MANIFEST.md)
- [Contributing and version management](CONTRIBUTING.md)
- [Agent development guide](AGENTS.md)
- [Poster and image notes](assets/README.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

## References and acknowledgements

Thanks to [LeRobot](https://github.com/huggingface/lerobot), [LeIsaac](https://github.com/LightwheelAI/leisaac) and [IsaacLab](https://github.com/isaac-sim/IsaacLab). Engineering structure draws on [rl-garden](https://github.com/Nole326/rl-garden), and VLA reinforcement-learning research draws on [RLinf](https://github.com/RLinf/RLinf). See [third-party notices](THIRD_PARTY_NOTICES.md) for licenses.

The poster is a refined and adapted Chinese-English version of the course project poster. The robot image is sourced from the [official LeRobot SO-101 documentation](https://huggingface.co/docs/lerobot/so101).
