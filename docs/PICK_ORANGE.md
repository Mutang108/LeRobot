# SO-101 PickOrange

将三颗橙子放入盘中，再让机械臂返回休息姿态。这里说明环境与诊断入口；完整GR00T原生辅助开发评测和教师采集见[PickOrange运行指南](../course/pickorange/README.md)。任务方法和结果与[LiftCube](../course/liftcube/README.md)分开管理。两者差异见[任务对照](TASKS.md)。

Put three oranges on the plate and return the arm to rest. This guide covers the public environment and diagnostic tools currently in the repository. Keep its methods and results separate from LiftCube; see the [task comparison](TASKS.md).

## 代码与支持范围 / Code and scope

| 内容 / Component | 入口 / Location |
| --- | --- |
| 环境与描述 / Task definition | [pick_orange_env_cfg.py](../third_party/leisaac/source/leisaac/leisaac/tasks/pick_orange/pick_orange_env_cfg.py) |
| 原生成功 / Native success | [mdp/terminations.py](../third_party/leisaac/source/leisaac/leisaac/tasks/pick_orange/mdp/terminations.py) |
| 单臂观测 / Single-arm observations | [single_arm_env_cfg.py](../third_party/leisaac/source/leisaac/leisaac/tasks/template/single_arm_env_cfg.py) |
| 控制器转换 / Action mapping | [action_process.py](../third_party/leisaac/source/leisaac/leisaac/devices/action_process.py)、[joint_contract.py](../course/scripts/joint_contract.py) |
| 环境/RGB检查 / Environment smoke | [validate_sim.py](../course/scripts/validate_sim.py) |
| 轨迹/末帧/重放检查 / Trajectory diagnostics | [validate_rollout.py](../course/scripts/validate_rollout.py) |
| ACT数据桥 / ACT data bridge | [validate_bridge.py](../course/scripts/validate_bridge.py)、[训练诊断说明](TRAINING_VALIDATION.md) |

仓库另有`course/pickorange/`的GR00T模型服务、原生辅助完整回合评测及CPU轨迹审核，新移植仍待真实环境验收。尚无通用PickOrange训练器、已训练ACT策略或完整视觉规则baseline；其他VLA后端需继续接入。LiftCube的`starter.evaluation.evaluate`工厂只创建LiftCube。

The separate GR00T native-assisted source port includes a model service, full-episode runner and CPU audit; runtime validation remains pending. A general trainer, trained ACT policy, visual rule baseline and other task backends remain open. The LiftCube factory is not a PickOrange runner.

## 独立运行环境 / Separate runtime

公开源码固定为LeIsaac0.4.0与IsaacLab2.3.0，身份见[依赖清单](DEPENDENCIES.md)。这不是LiftCube配套的4.5.0/2.1.1/0.1.2镜像。PickOrange包装脚本依赖独立Isaac Sim安装和Python3.11应用覆盖层，当前没有与LiftCube镜像同等完整的可移植镜像/安装锁；部署时需核实实际模拟器、Torch/CUDA、导入路径及原生依赖。

The pinned sources are LeIsaac0.4.0 and IsaacLab2.3.0, not the LiftCube image. Launchers require a separate Isaac Sim installation and Python3.11 application overlays. A portable matching image/install lock is not supplied; verify actual simulator and dependency identities before running.

包装脚本的容器内布局 / Launcher layout inside the container:

```text
/workspace/course/
├── scripts/                 # 本仓库course/scripts中的工具 / tools
├── leisaac/source/leisaac/  # 匹配的任务源码 / matching task source
├── venv_sim/                # 仿真应用覆盖层 / simulator overlay
├── venv_bc/                 # LeRobot/ACT应用环境 / model environment
├── cache/                  # 不放数据集 / cache outside datasets
└── logs/
/data/course/assets/         # 独立外部场景资产 / external scene assets
/data/course/validation/     # 诊断产物 / diagnostic outputs
```

这是脚本接口路径，不是个人服务器绑定。资产来源与SHA见[外部资产清单](../manifests/external_assets.json)；`prepare_assets.py`会下载/解压并写文件，不能当只读检查。`constraints-isaac51.txt`明确记载历史安装冲突，不是可直接复现的环境锁；不要据其文件名推断已安装的Isaac Sim版本。

These are launcher paths, not personal infrastructure. Asset preparation downloads and writes files. The historical `constraints-isaac51.txt` is not an installation lock or evidence of the deployed simulator version.

## 观测、动作与评分 / Observation, action and scoring

- 原生`policy`组含front/wrist RGB（各480×640）、关节位置/速度及其他仿真字段；`subtask_terms`和物体真值只能用于许可评分/诊断，不送入视觉策略。仿真有速度不意味着真机接口会提供速度。 / Native simulator fields are not an automatic policy whitelist or hardware contract.
- `joint_pos`按实际关节顺序取弧度；模型使用motor时先按明确标定转换。执行端的`so101leader`动作管理器再处理scale/offset，不能直接将motor或绝对q送给`env.step`。 / Convert model units to absolute radians, then to native raw actions using actual scale/offset.
- 原生成功要求三颗橙子同时满足盘区XYZ条件，且机械臂回位；一颗入盘或曾经命中过盘区是进度，不是整任务成功。 / Native success requires all three objects and the rest pose.
- 运行前记录实际`physics_dt`、`step_dt`、动作保持和时限；20帧10Hz、128×128桥接条件仅属于该诊断，不是统一策略规格。 / Record real timing; bridge settings are not a task-wide specification.
- 环境提供双相机与课程允许策略使用双相机是不同问题；按实际考核要求固定front-only或双视角协议，报告真实输入，不把双图结果当单图结果。 / Freeze the permitted view protocol and report it accurately.
- 保持原生物理、执行器、成功条件及reset前终态；键盘/状态机等上游路径可能启用重力或夹爪辅助，教师采集与视觉baseline必须分组。 / Separate assisted teachers from visual baselines.

## 已配置环境中的检查 / Checks in a configured runtime

仅在已获准运行仿真、资源和依赖均满足时，于上述容器内执行。`NEW_TAG`替换为未使用的名称；脚本会写日志与轨迹，不启动训练。

Run only with simulation approval and a configured runtime. Use a fresh tag; this writes diagnostic outputs, not training results.

```bash
cd /workspace/course
bash scripts/course_python.sh scripts/validate_rollout.py \
  --run-tag NEW_TAG --headless --enable_cameras
```

来源manifest及SHA通过后，才做ACT格式/前向桥接，使用另一个新tag / Validate source identity before the separate ACT bridge:

```bash
bash scripts/course_bc_python.sh scripts/validate_bridge.py \
  --source-tag SOURCE_TAG --output-tag NEW_BRIDGE_TAG
```

不要批量执行`validate_*.py`，也不要import探测其可用性；部分脚本在顶层创建应用/写文件。`validate_sim.py`使用固定输出路径，已有结果时不得覆盖重跑。详细工具副作用见[目录AGENTS](../course/scripts/AGENTS.md)。

Do not batch-run or import diagnostics as probes: some create applications or files at module scope. Preserve existing fixed-path outputs.

## 补齐任务方法 / Completing the task pipeline

先验收[已整理的GR00T原生辅助入口](../course/pickorange/README.md)，再接其他后端与完整训练流程。模型后端与模拟器依赖分离；数据、任务提示、统计、视角和评分按PickOrange冻结。使用[公共单front包装](VLA_INTERFACES.md)需要显式任务适配，不能把LiftCube配置换成PickOrange名称即运行。权重/演示见[资源清单](RESOURCES.md)，训练见[使用指南](TRAIN_AND_EVALUATE.md)。

Build the task observation/action/timing boundary and terminal capture first, then reuse model/chunk components in a full-episode runner. Keep model and simulator dependencies separate, with PickOrange-specific data, prompts, normalization and evaluation identities.
