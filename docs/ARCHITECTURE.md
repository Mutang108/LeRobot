# 架构与扩展 / Architecture and extension

[中文入口](../README.md) · [English entry](../README.en.md) · [开发起步 / Getting started](GETTING_STARTED.md)

## 1. 组织原则 / Organization

参考 [rl-garden](https://github.com/Nole326/rl-garden) 的算法、策略、环境、数据和训练分层。当前直接复用 `course/liftcube/src/starter/`，保留课程包的导入路径、补丁和来源清单；不为了目录对称搬动已验证代码，也不复制完整 rl-garden 作为第二套运行框架。

Follow [rl-garden](https://github.com/Nole326/rl-garden) in separating algorithms, policies, environments, data and training. Reuse `course/liftcube/src/starter/`, preserving its import paths, patches and provenance. Do not move validated code merely for symmetry or copy an entire second framework.

策略链路为：图像与许可本体 → 模型预处理 → 策略预测 → 公共动作转换/安全执行 → 环境反馈 → 数据和指标。训练奖励可读取规则允许的特权信息，策略输入不能读取；正式评分与训练奖励分开。

The policy path is: images/permitted proprioception → model preprocessing → prediction → shared action conversion/safe execution → environment feedback → data/metrics. Permitted privileged information may enter training rewards, not policy observations. Formal scoring remains separate from training rewards.

## 2. 当前代码与职责 / Existing responsibilities

下表路径相对 `course/liftcube/src/starter/`。它描述已存在的代码，不代表所有训练或模型已验收。

Paths below are relative to `course/liftcube/src/starter/`. These are existing modules, not claims that all trainers or models have passed acceptance.

| 层 / Layer | 当前位置 / Location | 负责 / Responsibility |
| --- | --- | --- |
| 契约 / Contracts | `interfaces/observation.py`, `action.py`, `policy.py` | 图像、本体、动作语义与策略协议 / Images, proprioception, action semantics and policy protocol |
| 模型 / Model | `interfaces/model.py`, `utils.py::build_policy` | 当前小型CNN及显式构造 / Current small CNN and explicit construction |
| 环境 / Environment | `envs/factory.py`, `lift_cube.py`, `render_ready.py` | 配套环境、限位转换、时序、reset前终态与初始渲染 / Runtime, mapping, timing, terminal capture and initial rendering |
| 视觉规则 / Visual rule policy | `scripted_liftcube.py`, `scripted_geometry.py` | RGB定位、抓取状态机及可选几何预设，不经过学习模型构造器 / RGB localization, grasp stages and optional geometric presets; bypasses the learned-model builder |
| 标定与运动学 / Calibration, kinematics | `camera_calibration.py`, `joint_chain.py`, `scripts/export_joint_chain.py` | 外参来源边界、固定机器人FK/IK与来源核验 / Extrinsic provenance, fixed-robot FK/IK and source validation |
| 末端开发层 / Planar EE development | `envs/planar_ee.py` | 可选三维末端请求到关节执行，独立于旧抓取脚本；物理定高待验 / Opt-in EE-to-joint execution, separate from the grasp runner; physical height tracking remains unaccepted |
| 数据 / Data | `data/hf_adapter.py`, `schema.py`, `validator.py`, `loader.py` | 数据格式、单位、校验及按回合划分 / Formats, units, validation and episode-level splits |
| 算法 / Algorithms | `algorithms/bc.py`, `rl.py`, `reference/ppo.py` | 损失、更新与参考算法 / Losses, updates and reference algorithms |
| 训练 / Training | `training/train_bc.py`, `train_rl.py` | 解析配置并组合上述模块 / Parse configuration and compose modules |
| 评测 / Evaluation | `evaluation/evaluate.py` | 策略加载、共享环境与逐回合结果 / Policy loading, shared environment and episode results |
| 奖励、日志 / Rewards, logging | `rewards/`, `logging/` | 训练信号与指标，不替换官方成功条件 / Training signals and metrics, not replacement scoring |

`PolicyModel`定义训练与推理协议，`build_policy()`当前仍固定构造`SmallCNNPolicy`。VLA推理包装通过公共评测的`--policy-class`加载，未实现训练协议中的`forward/update/save`。接入训练时再扩展构造器和算法检查，不能仅换配置或类名。

`PolicyModel` covers training and inference; `build_policy()` still constructs SmallCNN. VLA wrappers load through evaluation's `--policy-class` and do not implement training's `forward/update/save`. Training integration requires corresponding builder and algorithm changes, not just another class name.

规则入口`starter.scripts.evaluate_scripted`自行构造规则策略，随后复用共同`run_episode`和六关节环境；它不经`build_policy()`，也没有训练步骤。自研`planar_ee_dev`则从工厂显式选择，要求匹配的三维策略，旧抓取入口会拒绝误接。接入位置和限制分别见[规则指南](SCRIPTED_LIFTCUBE.md)与[末端开发契约](PLANAR_EE_DEVELOPMENT.md)。

The rule CLI constructs a rule policy directly, then reuses `run_episode` and the joint environment without training. The factory's explicit `planar_ee_dev` mode instead requires a compatible three-component policy; the old grasp runner rejects it. Follow the separate rule and EE guides rather than treating them as interchangeable entry points.

## 3. 新功能放在哪里 / Extension points

已实现的SmolVLA、π₀.₅、GR00T和OFT单front包装及共享关节chunk层见[VLA推理接口](VLA_INTERFACES.md)。公共评测在回合两端及异常退出清空可选策略缓存，不改变seed；各完整模型后端仍需实际环境验收。当前仅CPU接口/mask/加载辅助检查，不声称完整checkpoint、GPU或抓取验收。

See [VLA inference interfaces](VLA_INTERFACES.md) for four single-front wrappers and shared joint chunks. Evaluation resets optional caches at both boundaries and on exceptions without reseeding. Full native backends still require runtime acceptance; CPU interface/mask/loading-helper checks are not model/GPU/grasp acceptance.

| 改动 / Change | 约定 / Convention |
| --- | --- |
| ACT / SmolVLA / OpenVLA-OFT / π₀.₅ / GR00T | 模型专属代码在`starter/models/`，四种VLA已有推理包装，ACT待接入；复用公共观测/动作与评测 / Model-specific code belongs under `starter/models/`; four VLA inference wrappers exist, ACT is pending |
| SFT/BC 与 RL / SFT/BC and RL | 模型前向与优化算法分开，复用或新增实际需要的训练入口 / Separate forward inference from optimization; extend entry points only as needed |
| residual | 基座预测、残差和实际执行分别记录；同一执行层限位 / Record base, residual and applied actions separately; use the same execution safety layer |
| 新任务 / New task | 扩展 `envs/` 和任务配置；不复制每种模型一份环境 / Extend environments and task configurations, not an environment copy per model |
| 配置 / Configuration | 当前任务运行参数在 `course/liftcube/configs/`；根 `configs/` 保留无依赖计划模板 / Task runtime configuration stays under the package; root configs remain dependency-free planning templates |
| 外部框架 / External framework | 独立固定版本及环境，通过经验证的适配边界接入；不混装配套模拟器依赖 / Pin separately and integrate through validated adapters; do not mix simulator dependencies |

保留训练目标、预测动作和实际执行动作的区别。更换相机、resize、归一化、关节顺序、单位、chunk长度或控制周期时，更新对应契约和数据/权重身份；不能通过截掉一维或用真值补输入来冒充模型适配。

Distinguish training targets, predictions and applied commands. Changes to cameras, resizing, normalization, joint order, units, chunks or control periods require matching contracts and data/checkpoint identities. Dropping an action dimension or supplying ground truth is not a valid substitute for adaptation.

## 4. 分支与版本 / Branches and versions

共同起点已合入 `main`。后续从最新 `main` 各开短期功能分支，例如 `feature/liftcube-act`、`feature/liftcube-pi05`、`feature/liftcube-rl`。开发完成后通过PR合入并删除已合并分支；所有方法的源码最终在main共存，运行时由已实现的入口与配置选择。

The shared foundation is merged into `main`. Branch subsequent work from current main, for example `feature/liftcube-act`, `feature/liftcube-pi05` or `feature/liftcube-rl`. Merge completed work through PRs and delete merged branches; implementations coexist on main and are selected through implemented entry points/configurations.

多个未合入模块确实需要一起联调时，才加临时 `dev/liftcube`。不按成员、seed或checkpoint永久维护分支。共同接口修复单独提交，其他分支同步；README、CHANGELOG与代码在同一PR更新。阶段发布使用CHANGELOG版本条目和不可覆盖的tag，而不是复制整套代码目录。

Use temporary `dev/liftcube` only for genuine integration of pending modules. Avoid permanent branches per person, seed or checkpoint. Merge shared fixes separately and synchronize dependent branches; include README/changelog updates in the same PR. Publish milestones through changelog entries and immutable tags, not copied source trees.

## 5. 验证与外部文件 / Validation and external files

只修改文档和目录导航时，复用未变代码/数据的有效验收，检查受影响链接与展示。修改模型、采样、动作或恢复时，补对应最小正确性和集成检查。完整存档需要模型、优化器、计数、数据位置及全部独立RNG同点；上游有resume选项不等于本项目已经验证恢复。

For documentation/navigation changes, reuse valid unchanged code/data evidence and check affected links/rendering. Model, sampling, action or recovery changes need targeted correctness and integration checks. Complete checkpoints require model, optimizer, counters, data position and all independent RNG streams at one point; an upstream resume option does not establish acceptance here.

Git只保存源码、可公开配置、来源与校验清单。镜像、数据、权重、缓存、日志、机器配置和私有实验记录存于获准的外部位置；遵循[贡献规范](../CONTRIBUTING.md)、[依赖说明](DEPENDENCIES.md)及[部署边界](REPRODUCIBILITY.md)。

Git stores source, public configuration, provenance and verification inventories. Keep images, datasets, weights, caches, logs, machine configuration and private experiment records in approved external storage; follow [Contributing](../CONTRIBUTING.md), [dependencies](DEPENDENCIES.md) and [deployment boundaries](REPRODUCIBILITY.md).
