# 从准备到训练 / From preparation to training

先完成一条最小闭环，再扩大模型和训练量：读取观测 → 输出动作 → 安全执行 → 记录结果。能导入代码、能计算一次 loss、能完成任务，是三个不同的验收层级。

Build one minimal loop before increasing model size or training budget: read observations → produce actions → execute safely → record outcomes. Importing code, computing a loss and completing a task are different acceptance levels.

## 1. 先看什么 / Where to start

阅读顺序：本页 → [接口约定](INTERFACE_CONTRACTS.md) → [依赖](DEPENDENCIES.md) → [训练验证](TRAINING_VALIDATION.md)。正式任务定义优先于海报和上游示例；尚未发布的参数保持未确认，不用“常见默认值”补齐。

Read this page, then the [interface contracts](INTERFACE_CONTRACTS.md), [dependencies](DEPENDENCIES.md) and [training validation](TRAINING_VALIDATION.md). The formal task definition takes precedence over artwork and upstream examples. Leave unpublished parameters unresolved instead of substituting familiar defaults.

当前先按[任务对照](TASKS.md)选择LiftCube或PickOrange，再按[资源清单](RESOURCES.md)获取模型/演示、按[训练与评测](TRAIN_AND_EVALUATE.md)选择执行入口。LiftCube已有配套镜像、视觉脚本与参考训练；PickOrange使用独立环境，另有[GR00T原生辅助服务/评测源码](../course/pickorange/README.md)，移植待实际验收。两任务不混用依赖、统计或评分，不等待尚未发布的正式多物体环境才开发。

Choose a task in the comparison, obtain its resources and follow the execution guide. LiftCube has a matching image and reference implementations; PickOrange has a separate runtime and a GR00T assisted source port awaiting runtime acceptance. Keep task dependencies and results separate.

按目标选择入口，避免把所有训练前提都套到规则推理上：

| 目标 / Goal | 入口与前提 / Entry and prerequisites |
| --- | --- |
| PickOrange接口与重放 / PickOrange interfaces and replay | [任务指南](PICK_ORANGE.md)；独立仿真/模型环境、真实动作offset与终态，诊断不作正式评测 / Separate runtime, native action mapping and terminal capture; diagnostics are not benchmarks |
| 视觉规则抓取 / Rule grasping | [规则指南](SCRIPTED_LIFTCUBE.md)；匹配标定与运动学，先`--check`再获准闭环，无需演示或权重 / Calibrated kinematics, offline check then authorized rollout; no dataset/weights |
| 三维末端开发 / Planar EE development | [开发契约](PLANAR_EE_DEVELOPMENT.md)；独立标定、有效初始姿态及匹配策略，物理精度未验收 / Separate calibration, valid starting pose and compatible policy; tracking not accepted |
| BC/PPO / Learning | [训练指南](../course/liftcube/TRAINING.md)；当前为六关节SmallCNN参考路径，BC先完成演示准入 / Six-joint SmallCNN reference path; BC requires accepted demonstrations |
| VLA推理接口 / VLA inference interfaces | [四种包装与CPU检查](VLA_INTERFACES.md)；先验证单front、六关节与chunk边界，完整模型验收需要匹配后端、权重与统计 / Check boundaries first; full loading requires matching backends, weights and statistics |

先在本地运行以下检查。第一条应成功；第二条对未填写的模板应返回退出码 3，并逐项列出缺失字段。

Run these local checks first. The first should pass; the second should return exit code 3 for the unfilled template, listing the unresolved fields.

```bash
python scripts/check_project.py
python scripts/inspect_plan.py configs/experiment-plan.toml --print-config
```

配置检查只需 Python 3.11+，不导入 Torch、LeRobot 或模拟器。退出码 2 表示错误，3 表示未填齐，0 只表示字段齐全且通过静态检查，**不表示已经允许或可以安全开训**。`--print-config` 输出配置和文件 SHA，便于对照；没有启动或安装副作用。

Plan inspection needs only Python 3.11+, with no Torch, LeRobot or simulator imports. Exit code 2 means invalid, 3 means incomplete, and 0 means specified and statically consistent—**not authorized or safe to train**. `--print-config` prints the configuration and file SHA for comparison; it launches and installs nothing.

## 2. 按阶段交付 / Deliver stage by stage

| 阶段 / Stage | 交付 / Deliverable | 通过条件 / Acceptance |
| --- | --- | --- |
| 接口 / Interfaces | 观测与动作 schema、控制周期 / Schemas and timing | 实测坐标、单位、正负方向、限速、相机时序 / Measured frames, units, directions, limits and camera timing |
| 采集 / Collection | 少量完整轨迹及质量报告 / Small complete dataset and quality report | 动作可重放，成功可靠，训练/验证隔离 / Replay, trustworthy outcomes, separated splits |
| BC / Imitation | ACT 与 SmolVLA 适配、模型和归一化 / Adapters, checkpoints and normalization | 小批拟合、padding、保存恢复和闭环 / Small-batch fitting, padding, recovery and rollout |
| RL / Reinforcement learning | 奖励、普通RL、再加残差 / Reward, plain RL, then residuals | 末帧正确、真实更新、匹配存档 / Correct terminal observations, real updates, matching checkpoints |
| 评测 / Evaluation | 固定清单、逐回合结果、失败记录 / Fixed manifest, per-episode results and failures | 全部场景按原规则计分 / All scenes scored using the original rules |
| 真机 / Hardware | 部署配置及安全检查 / Deployment configuration and safety checks | 限速、急停、视觉丢失暂停及人工确认 / Limits, emergency stop, tracking-loss pause and human acceptance |

先让 ACT 和 SmolVLA 使用同一批演示和同一评测协议。模型内部预处理可以不同，但最终动作必须回到同一个执行接口。先建立 BC 与普通 RL 对照，再接残差，便于判断改进来自哪里。演示预训练与在线交互成本分别记录。

Use the same demonstrations and evaluation protocol for ACT and SmolVLA. Their internal preprocessing may differ, but outputs must reach the same execution interface. Establish BC and plain-RL baselines before adding residuals so improvements can be attributed. Record demonstration pretraining and online interaction costs separately.

## 3. 代码改在哪里 / Where changes belong

各层职责与ACT/VLA/RL的扩展位置统一见[架构说明](ARCHITECTURE.md)；RLinf是否适合当前任务见[接入评估](RLINF.md)。根说明提供[中文](../README.md)和[英文](../README.en.md)两个入口，其他技术文档暂保留双语。

See [architecture](ARCHITECTURE.md) for layer ownership and ACT/VLA/RL extension points, and the [RLinf assessment](RLINF.md) for applicability. The root guide has [Chinese](../README.md) and [English](../README.en.md) editions; other technical guides remain bilingual.

`third_party/` 是冻结的上游参考，不是日常实验草稿区。当前 `course/scripts/` 是独立诊断工具，`scripts/` 是仓库和计划检查，`tests/` 放回归测试，`configs/` 放可公开的计划模板。具体适配器通过验收后再接入训练入口，不先创建一堆空训练命令。私有机器配置、数据、权重和结果留在仓库外或忽略目录。

`third_party/` is a frozen upstream reference, not a scratch area for experiments. `course/scripts/` holds standalone diagnostics, `scripts/` repository and plan checks, `tests/` regressions, and `configs/` public planning templates. Integrate adapters into a trainer after validation rather than creating empty training commands. Keep machine configuration, data, weights and results outside the repository or in ignored directories.

LiftCube代码位于`course/liftcube/`：模型、数据和环境适配在`src/starter/`，任务配置在该目录的`configs/`。新增模型沿用共同动作和评测接口；公共接口修正与模型专属实现分开评审。

LiftCube code lives in `course/liftcube/`: models, data and environment adapters under `src/starter/`, with task configurations in its `configs/`. New models share action and evaluation contracts; review shared-interface fixes separately from model-specific implementations.

现有诊断脚本中有固定挂载路径、20帧数据和小模型等专用条件，运行前看脚本说明，不把它们当成通用采集器或完整训练器。[训练验证](TRAINING_VALIDATION.md)说明了已覆盖与未覆盖的内容。

Some diagnostics assume specific mount paths, 20-frame data or small models. Read their descriptions before running them; they are not generic collectors or complete trainers. [Training validation](TRAINING_VALIDATION.md) documents coverage and limitations.

## 4. 团队怎么协作 / Working as a team

按接口而不是按文件分工：环境/控制、数据、BC/VLA、RL、评测/复现，各自可以有负责者，但改变共同 schema 要一起评审。一个小改动一个功能分支和 PR；PR 写清目的、接口变化、测试、剩余风险。README 只改受影响段落，重要变化追加 `Unreleased`，不复制一份新的 README。

Assign ownership by interface rather than by file: environment/control, data, BC/VLA, RL, and evaluation/reproducibility. Shared schema changes need joint review. Use a focused branch and PR for each change, describing its purpose, interface impact, tests and remaining risks. Edit only affected README sections and append important changes to `Unreleased`, rather than duplicating the README.

`main` 保持可复现；功能分支可直接 PR 到 `main`。只有确实需要多个模块一起联调时，再使用团队约定的 `develop` 集成分支。本说明没有创建分支、开启保护规则或设置审核权限，具体流程见[贡献说明](../CONTRIBUTING.md)。

Keep `main` reproducible; feature branches may open PRs directly into it. Use an agreed `develop` integration branch only when several modules need joint testing. This guide does not create branches, enable protection or assign review permissions. See [Contributing](../CONTRIBUTING.md).

## 5. 借鉴 rl-garden 的部分 / Lessons from rl-garden

借鉴 [rl-garden](https://github.com/Nole326/rl-garden) 的做法：薄命令入口、先查看完整配置再运行、环境与算法分层、离线和在线预算分开、恢复时显式处理优化器与 replay。这里先采用轻量配置检查，不立即复制完整算法注册系统。后端可选依赖不应阻塞文档检查或配置读取。

We follow [rl-garden](https://github.com/Nole326/rl-garden) in keeping entry points thin, inspecting resolved configurations before execution, separating environments from algorithms, accounting for offline and online budgets separately, and handling optimizer/replay restoration explicitly. We start with lightweight plan inspection rather than copying the entire algorithm registry. Optional backend dependencies should not prevent documentation checks or configuration inspection.

不继承其他任务的权重、奖励、动作比例、步数上限或测试 seed；这些都要在当前任务重新确认。保存 RNG 也不等于保存完整仿真现场，不能仅凭上游的 checkpoint 选项承诺无缝续训。

Do not inherit another task's weights, rewards, action scales, horizons or evaluation seeds; validate them for this task. Saving RNG does not save the complete simulator state, and upstream checkpoint options alone do not establish seamless resumption.
