# SO-101 LiftCube

抓取桌面方块并抬升的开发起点，包含环境适配器、演示读取、BC、参考PPO和评测入口。完整多物体抓放使用后续发布的任务环境。

A starting point for picking up a tabletop cube, with an environment adapter, demonstration reader, BC, reference PPO and evaluation entry points. The complete multi-object task uses its separately released environment.

## 当前状态 / Status

六关节环境交互与BC/PPO参考训练基础已接入，规则脚本可独立于训练使用。候选演示尚未通过正式BC准入；[四种VLA的单front包装与共享chunk层](../../docs/VLA_INTERFACES.md)已有CPU接口检查，完整模型及GPU/闭环尚未验收，ACT接入仍在开发。训练和恢复用法见[训练指南](TRAINING.md)，任务迁移与采集方案见[开发路线](../../docs/LIFTCUBE_TRANSFER.md)。

Joint-environment interaction and BC/PPO reference foundations are connected; the rule runner is independent of training. Candidate demonstrations are not accepted for formal BC. [Four single-front VLA wrappers/shared chunks](../../docs/VLA_INTERFACES.md) have CPU interface checks, not full-model/GPU/rollout acceptance; ACT integration remains in development. See the [training guide](TRAINING.md) and [task-transfer plan](../../docs/LIFTCUBE_TRANSFER.md).

| 环节 / Component | 当前实现 / Implementation |
| --- | --- |
| 参考SmallCNN / Reference SmallCNN | RGB＋关节位置＋上一执行命令，屏蔽兼容补零字段 / RGB, joint positions and previous commands; compatibility padding is ignored |
| 规则基线 / Rule baseline | [视觉定位＋IK状态机](../../docs/SCRIPTED_LIFTCUBE.md)；实际标定辅助定位调度变体45/100，成功中位9.38仿真秒；旧几何原版13/30、固定标定1/10分组 / Assisted scheduling variant: 45/100, median successful time 9.38 simulation seconds; earlier 13/30 and fixed-calibration 1/10 protocols kept separate |
| 受限末端开发接口 / Planar EE development | 可选[绝对XY＋夹爪执行层](../../docs/PLANAR_EE_DEVELOPMENT.md)，固定高度/接近轴参考，物理定高未验收；不是正式IK或已有关节抓取策略的移植 / Opt-in absolute XY + jaw layer; physical height tracking unaccepted, not official IK or a port of the joint grasp policy |
| BC→RL | 30Hz策略在60Hz环境中保持两tick；严格匹配模型与数据规格 / 30Hz decisions hold for two 60Hz ticks; model/data contracts are checked |
| 存档恢复 / Checkpoint recovery | BC按epoch、PPO按完整回合/更新边界保存模型、Adam、进度与RNG；合成CPU恢复通过 / Model, Adam, progress and RNG at epoch/completed-rollout boundaries; synthetic CPU recovery checked |
| PPO超时 / PPO timeouts | 外部截断从reset前终态bootstrap，GAE在回合边界断开；有限时域可显式配置 / External timeouts bootstrap from pre-reset observations; GAE stops at episode boundaries; finite-horizon mode is explicit |

演示动作映射也需先完成核对，数据要求见下方[起步说明](#起步--getting-started)。ACT/VLA接入后共用环境和评测接口。

Verify demonstration action mapping before training; see [Getting started](#起步--getting-started) below. ACT/VLA integrations will share the environment and evaluation interfaces.

## 运行环境 / Runtime

使用配套镜像 `isaac-lab:4.5.0-v2.1.1-leisaac-v0.1.2-src`：Isaac Sim 4.5.0、Isaac Lab v2.1.1、LeIsaac v0.1.2、Python 3.10.15、Torch 2.5.1+cu118。包元数据可能显示Isaac Lab 0.41.3、LeIsaac 0.1.0。仓库原有 `third_party/` 是另一组固定版本，不要混入这个环境的Python路径。

Use the matching image `isaac-lab:4.5.0-v2.1.1-leisaac-v0.1.2-src`: Isaac Sim 4.5.0, Isaac Lab v2.1.1, LeIsaac v0.1.2, Python 3.10.15 and Torch 2.5.1+cu118. Package metadata may report Isaac Lab 0.41.3 and LeIsaac 0.1.0. The repository's existing `third_party/` snapshots use different versions; keep them off this runtime's Python path.

从[交大云盘分享页](https://pan.sjtu.edu.cn/web/share/74e1cfacf999d941ad66860a03104554)下载 `isaac-lab_4.5.0-v2.1.1-leisaac-v0.1.2-src.tar`，核对下方SHA256后在Docker宿主导入。该链接是分享页面，不是tar文件直链；访问按页面要求及课程授予的权限进行。仓库不存储模拟器镜像、场景资产或缓存。

Download `isaac-lab_4.5.0-v2.1.1-leisaac-v0.1.2-src.tar` from the [SJTU cloud-drive share](https://pan.sjtu.edu.cn/web/share/74e1cfacf999d941ad66860a03104554), verify the SHA256 below, then load it on the Docker host. This is a share page, not a direct tar URL; follow its access requirements and the permissions granted by the course. Simulator images, scene assets and caches remain outside Git.

```bash
sha256sum /path/to/isaac-lab_4.5.0-v2.1.1-leisaac-v0.1.2-src.tar
docker load -i /path/to/isaac-lab_4.5.0-v2.1.1-leisaac-v0.1.2-src.tar
docker image inspect --format '{{.Id}}' isaac-lab:4.5.0-v2.1.1-leisaac-v0.1.2-src
```

Windows下载后可用PowerShell核验 / Verify a Windows download with PowerShell:

```powershell
Get-FileHash -Algorithm SHA256 -LiteralPath '.\isaac-lab_4.5.0-v2.1.1-leisaac-v0.1.2-src.tar'
```

镜像归档SHA256 / Image archive SHA256：`83df9539e280d663681b876cda3aa4eff631164f743f0aad3d693e1e17d6b17f`。

导入后的image ID / Loaded image ID：`sha256:17d6d1a69fc4b361dffab3238ab170f72ec7f2f9b5cb6d58050e1c6ffd80d752`。

以上校验值对应当前配套镜像；若分发方更新文件，先核对发布说明和版本，再更新清单。

These checksums identify the current matching image. If the distributor replaces it, confirm the release/version before updating the inventory.

## 接口 / Interface

| 内容 / Field | 当前约定 / Contract |
| --- | --- |
| 图像 / Image | 单front RGB、uint8 HWC；参考模型默认64×64，规则入口可显式设128/256并匹配标定 / One front stream; 64×64 by default, optionally 128/256 square for the calibrated rule runner |
| 本体 / Proprioception | 6关节位置 / joint positions; 6速度补零 / zero velocity slots; 6上一执行命令 / previous applied commands; 可选3维末端位置补零 / optional 3 zero EE slots |
| 默认动作 / Default action | 6维绝对电机目标 / absolute motor targets; arm −100…100, gripper 0…100；三维末端开发模式单列 / planar development mode has a separate contract |
| 转换 / Conversion | 按关节限位映射到弧度 / per-joint mapping to radians |
| 频率 / Frequency | 环境60Hz，30Hz策略每两tick更新 / 60Hz environment, 30Hz policies hold for two ticks |
| 默认上限 / Default horizon | 640环境tick，约10.67秒 / 640 environment ticks, about 10.67 seconds |

默认评测为4回合、每回合最多640环境tick。这是起点配置，不是最终课程评测协议；正式任务的回合数、成功判据和时间上限以课程发布为准。

The default evaluation runs four episodes with at most 640 environment ticks each. These are starter defaults, not the final course protocol; use the course's published episode count, success criteria and time limit for formal evaluation.

参考BC/PPO配置在训练和推理中均默认使用64×64；这不是对所有encoder的强制尺寸。规则128/256输入须同步环境resize、`image_shape`和内参K，不能只改图像尺寸。其他模型更改输入需另验数据/权重兼容性。物体真值只供许可奖励或独立评分/诊断，不能进入策略；关节动作不能直接当成受限末端动作。

Reference BC/PPO uses 64×64 by default, not a universal encoder requirement. Rule inputs at 128/256 must match resizing, calibration shape and intrinsics. Other model/input changes require data/checkpoint compatibility checks. Ground truth stays in permitted rewards or independent scoring/diagnostics, never policy inputs; joint targets are not EE actions.

## 起步 / Getting started

先准备符合所在平台规范的GPU容器，将本仓库挂载到`/repo`；需要读取演示时再将数据单独挂载到`/data/lift_cube`。规则推理不需要演示或训练权重，但需要完整外部标定。保留镜像内的`/isaaclab`、`/leisaac`，缓存不要写入数据集目录。共享服务器在已批准的容器内执行以下命令，避免使用自动Docker启动路径。

Use a platform-approved GPU container with this repository mounted at `/repo`; mount `/data/lift_cube` separately only when reading demonstrations. Rule inference needs calibration, not demonstrations or trained weights. Preserve `/isaaclab` and `/leisaac`, keep caches outside datasets, and avoid automatic Docker startup on shared servers.

```bash
cd /repo/course/liftcube
export PYTHONPATH="$PWD/src:/leisaac/source/leisaac${PYTHONPATH:+:$PYTHONPATH}"
/isaac-sim/python.sh -m starter.scripts.smoke_test --config configs/lift_cube.yaml --steps 16
```

离线读取还需要PyArrow和PyAV；先检查镜像已有依赖，避免替换预装Torch/CUDA。数据可参考 [H2Ozone/lift_cube](https://huggingface.co/datasets/H2Ozone/lift_cube)，但须先核对单位、零位、限位和帧同步。v3加载器默认拒绝超限动作；`allow_action_clipping`仅为显式诊断选项。BC配置的数据路径应设为容器内实际挂载路径。

Offline reading requires PyArrow and PyAV; check existing packages and preserve bundled Torch/CUDA. [H2Ozone/lift_cube](https://huggingface.co/datasets/H2Ozone/lift_cube) is a candidate dataset, subject to unit, zero-point, limit and alignment checks. The v3 reader rejects out-of-range actions by default; `allow_action_clipping` is an explicit diagnostic option. Set the BC data path to the actual container mount.

`training/train_bc.py`、`training/train_rl.py`为参考训练入口，`evaluation/evaluate.py`通过`--policy-class package.module:ClassName`支持自定义策略加载；默认训练预算仅作示例。

`training/train_bc.py` and `training/train_rl.py` are reference training entry points. `evaluation/evaluate.py` supports custom loaders through `--policy-class package.module:ClassName`; default training budgets are examples.

规则策略使用独立`starter.scripts.evaluate_scripted`入口，先按[规则指南](../../docs/SCRIPTED_LIFTCUBE.md)完成`--check`；它会构造状态机，不通过需要模型权重加载器的通用`--policy-class`路径。[末端开发层](../../docs/PLANAR_EE_DEVELOPMENT.md)另需三维策略，不能直接复用这份六关节抓取策略。

The rule policy has its own `starter.scripts.evaluate_scripted` CLI and offline calibration check; it does not use the learned checkpoint loader. The separate planar layer needs a three-component policy, not this joint-output grasp runner.

## 来源与修改 / Provenance and changes

[来源清单](../../manifests/liftcube_source.json)记录原始归档及逐文件原始／当前SHA。[开发补丁](patches/team-interface.patch)记录环境接口与训练基础修改；评测配置保持原包内容。

The [source manifest](../../manifests/liftcube_source.json) records original/current file hashes. The [patch](patches/team-interface.patch) covers environment-interface and training-foundation changes. The evaluation configuration retains its original contents.

原包 `pyproject.toml` 声明MIT，未附独立LICENSE文件；保留原声明和作者字段，见[第三方说明](../../THIRD_PARTY_NOTICES.md)。镜像、数据和权重分别遵循其来源条款。

The original `pyproject.toml` declares MIT without a standalone LICENSE file. Its declaration and author field are retained; see [third-party notices](../../THIRD_PARTY_NOTICES.md). Images, datasets and weights follow their respective source terms.
