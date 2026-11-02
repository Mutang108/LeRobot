# AGENTS.md

本文件供开发智能体使用，适用于整个仓库。按任务读取下表中的子级指令；跨目录修改同时读取双方。项目介绍与安装说明见 [README](README.md)，协作流程见 [CONTRIBUTING](CONTRIBUTING.md)。

## 从这里开始

1. 检查 `git status --short`、`git branch --show-current` 和相关 diff，保留已有改动。
2. 选择下表中的入口，读实际调用代码及相关测试，再修改。只读取当前任务需要的文档。
3. 按用户任务完成实现与针对性验证；缺少环境、资产或设备时，继续能独立完成的工作，并报告具体缺口。
4. 提交、推送、合并和真机运行按用户授权执行。已有明确授权无需重复询问；示例命令不额外授予运行权限。

这是课程集成仓库，不是可在根目录安装的 Hugging Face LeRobot 包；不要在根目录执行 `pip install -e .`。克隆后的工作不依赖维护者的聊天记忆、相邻项目或私有文档。

首次接手时先读[开发起步](docs/GETTING_STARTED.md)和[任务对照](docs/TASKS.md)。当前分别推进LiftCube和PickOrange，正式多物体任务后续再接。仅修改代码或文档可直接使用根目录检查环境，无需先下载镜像、数据或权重；运行任务时按对应指南和子级指令准备环境。新功能从最新 `main` 建短期分支；用户已经指定工作分支时沿用该分支，不自动切换。

接手一个具体任务时，先从仓库确认四件事：目标任务/模型、当前入口与配置、输入输出契约、通过条件。能从文件确定的内容自行核对；只有会改变任务含义的缺口才询问用户。例如“修复单位换算”应验证现有映射及调用者，“接入某个VLA”还必须确定权重版本、动作语义和运行后端，不能用一个模型名称代替这些信息。

资料发生冲突时，对照实际源码、固定版本清单与适用的任务定义定位差异，并在同次修改中修正文档。README中的规划不等于已实现功能，上游最新说明不一定适用于本仓库冻结版本；不要为消除差异擅自升级依赖或改变任务。

## 按任务找入口

| 任务 | 修改位置与必读入口 |
| --- | --- |
| 仓库检查、文档、来源清单 | `scripts/`、`tests/`、`manifests/`；[来源说明](docs/IMPORT_MANIFEST.md) |
| 固定版本与基准复现 | [版本说明](docs/BASELINE.md)区分初始源码tag与规则策略基准；新开发从最新main，复现固定commit与外部标定身份，不混用旧tag功能 |
| SO101 接口、LiftCube 数据/训练/评测 | `course/liftcube/src/starter/`；[LiftCube AGENTS](course/liftcube/AGENTS.md) |
| LiftCube视觉规则基线 | `starter/scripted_liftcube.py`、`scripted_geometry.py`、`joint_chain.py`与`starter.scripts.evaluate_scripted`；[标定及验收](docs/SCRIPTED_LIFTCUBE.md)，实际相机标定辅助须显式启用，不得以合成标定启动仿真 |
| 自研受限末端开发接口 | `starter/envs/planar_ee.py`、`envs/factory.py`、`tests/test_planar_ee.py`及[开发契约](docs/PLANAR_EE_DEVELOPMENT.md)；三维绝对XY/夹爪，非课程正式IK；旧关节策略不得自动接入 |
| PickOrange任务与诊断 | [任务指南](docs/PICK_ORANGE.md)、`course/scripts/`及冻结上游任务源码；[目录AGENTS](course/scripts/AGENTS.md)、[部署边界](docs/REPRODUCIBILITY.md) |
| PickOrange GR00T原生辅助评测 | `course/pickorange/`；[目录AGENTS](course/pickorange/AGENTS.md)、[运行指南](course/pickorange/README.md)；固定N1.5双视角、60Hz及显式物体真值夹爪辅助，不当无特权或课程正式评分 |
| 教师、数据/权重获取、离线训练 | [资源清单](docs/RESOURCES.md)、[训练与评测](docs/TRAIN_AND_EVALUATE.md)；LiftCube显式`--privileged-teacher`，PickOrange已有上游状态机，其直接写状态不自动满足BC准入 |
| 训练与评测曲线 | [SwanLab指南](docs/SWANLAB.md)、`starter/logging/swanlab.py`与`starter.scripts.train_lerobot`；共享数值记录，保留本地证据，不另造每模型日志器 |
| ACT、SmolVLA、OpenVLA-OFT、π₀.₅、GR00T 接入 | 对应任务的策略包装；[架构](docs/ARCHITECTURE.md)、[接口契约](docs/INTERFACE_CONTRACTS.md)、该任务的子级 AGENTS |
| 单front VLA与chunk生命周期 | `starter/models/`的SmolVLA/PI05/GR00T/OFT包装、共享chunk/backend I/O及[推理接口](docs/VLA_INTERFACES.md)；GR00T可显式在模型侧接收不同输入尺寸，环境保持不变；CPU替身通过不等于真实模型或GPU验收 |
| 正式多物体任务或真机 | 先确认实际提供的环境、SDK、标定与评分包；[接口契约](docs/INTERFACE_CONTRACTS.md) |

LiftCube 已有 SmallCNN BC/PPO 参考训练与评测，`starter.utils.build_policy()` 仍固定 SmallCNN。PickOrange另有GR00T原生辅助评测源码，移植尚待真实环境验收，其他VLA任务后端及通用训练接入仍有缺口。禁止只换task ID、沿用单front包装或将`starter`工厂冒称通用PickOrange入口。两任务独立记录视角、动作映射、统计、预算和评分；正式多物体与真机后续按实际接口开发，不提前虚构或自动恢复其适配工作。

## 环境边界

| 工作范围 | 运行前提 |
| --- | --- |
| 根目录源码检查 | Git + Python 3.11+；不需要模拟器 |
| LiftCube | 配套镜像：Isaac Sim 4.5.0 / Isaac Lab 2.1.1 / LeIsaac 0.1.2；Python 3.10.15 |
| PickOrange | 独立仿真/BC环境；现有诊断包装含固定容器路径及Python3.11路径，无LiftCube同款可移植镜像锁 |
| `third_party/` | 冻结参考：LeRobot v0.4.2 / LeIsaac 0.4.0 / IsaacLab v2.3.0；不等于 LiftCube 镜像内版本 |

依赖、镜像、资产与权重获取入口见 [DEPENDENCIES](docs/DEPENDENCIES.md)。运行前核对解释器、实际导入路径、设备和数据/输出位置。不要把全部上游路径加入 `PYTHONPATH`，或为运行仓库检查升级模拟器的 Python/Torch/CUDA。可选模拟器与模型依赖应延迟到对应入口加载，保持纯配置与软件检查可独立运行。

## 修改约定

- 当前任务共用代码位于 `course/liftcube/src/starter/`：`interfaces/` 定义契约，`envs/` 处理环境，`data/` 处理演示，`training/` 负责训练与存档，`evaluation/` 负责评测。模型包装复用这些边界，不复制环境和评分逻辑。
- `course/liftcube/configs/` 是运行配置；根 `configs/experiment-plan.toml` 只是计划模板。`course/scripts/` 保持为独立诊断工具。
- `course/pickorange/`保存该任务模型服务/原生辅助评测及其协议，不与LiftCube或教师生成器混用。新增资源回填`docs/RESOURCES.md`及`docs/TASKS.md`，写明完整revision和实际验收级别，不提交模型参数或演示。
- `third_party/` 为冻结来源，优先在外部适配。未经明确任务要求，不改第三方源码、依赖版本或生成文件来消除检查失败。
- 沿用邻近代码与测试风格，复用现有函数、标准库和测试。修复实际调用链，只增加当前需求所需的实现；不预建空注册器、无关抽象或按模型复制整套训练器。
- 代码标识符应清晰，注释解释不明显的原因。公开说明、提交消息和 PR 面向未参与对话的读者，不记录聊天过程或个人操作流水。

## 不得改变的语义

完整定义以 [INTERFACE_CONTRACTS](docs/INTERFACE_CONTRACTS.md) 与任务配置为准：

- SO101 六维为五个机械臂关节加夹爪。明确关节顺序、单位、标定、绝对/增量和夹爪范围；不能截掉模型输出的一维冒充适配。
- 请求动作、限幅后发送动作、实测位置分开；发送不等于到位。不得用下一帧实测位置替换请求动作标签，或静默裁剪演示标签。
- 兼容补零不是实际速度/末端反馈。必要关节状态缺失应报错；派生状态注明来源与时序。
- 相机布局、图像处理、环境周期、策略频率、chunk 与执行长度显式对应；训练和评测共用兼容执行层。
- 物体/目标真值、分割和成功标记不得进入普通视觉策略或其 metadata；教师、监督、奖励、特权 critic 与评分分别隔离。
- 自动 reset 前捕获终态；区分 `terminated`、`truncated`、bootstrap 与成功。接口修正不改变物理、限位、执行器或评分条件。
- 推理导出、权重初始化和完整续训分别处理。续训需同点模型、优化器、计数、数据位置/replay 与相关 RNG；重新 seed 不等于恢复。
- 成功率仅统计同协议完整回合。保留失败与 partial，不换 seed、跳回合或覆盖旧证据；加载和进程退出不能证明测试完成。
- 新训练/策略评测曲线使用SwanLab。默认offline，online须实际使用者提供私有项目；只传数值与白名单配置，不传完整cfg、媒体/权重、终端或主机探针。恢复创建新日志分段并保留全局计数，日志run不代替模型恢复。云端异常不授权打断健康worker；不自动重跑或迁移旧事件。

## 验证

先运行受影响的最小检查，复用身份一致的有效证据。目标测试和仿真 smoke 见对应子级 AGENTS。文档调整检查链接、命令和清单，不触发 GPU 或训练。

提交前，从仓库根目录运行统一检查（Python 3.11+）：

```bash
python scripts/check_project.py
git diff --check
git diff --cached --stat
```

统一检查覆盖来源 SHA、项目 Python 语法、Markdown 本地链接、海报和软件单测；不启动模拟器。依赖缺失导致的 skip 必须披露，不能当相关功能通过。交付时写明实际命令、结果和未验证部分，不把合成测试写成真实仿真/真机成绩。

修改VLA包装时，另在具备NumPy、CPU Torch和PyYAML的隔离环境运行`python -m unittest discover -s tests -p 'test_vla*.py' -v`；环境说明见[推理接口](docs/VLA_INTERFACES.md)。CI的`vla-cpu`专门执行这些测试，避免纯标准库检查将其全部skip；feature分支推送本身不触发此工作流，PR、main推送或手动触发才运行。

修改日志接线时，另运行`tests/test_swanlab.py`和`tests/test_liftcube_training.py`；隔离环境需SwanLab 0.10.3、NumPy、CPU Torch、PyYAML。保持`--check`无SDK运行副作用，验证数值/配置白名单、RNG、计数/分段、原生logger挂钩及故障本地保全。真实SDK测试必须offline，不能把CPU QA写入团队正式云端项目。

`python scripts/inspect_plan.py configs/experiment-plan.toml --print-config` 只检查计划；退出码 0 为静态通过、2 为无效、3 为模板未填齐，均不表示训练已运行。

## 改动后同步什么

| 变化 | 同步内容 |
| --- | --- |
| 输入、动作、时序、数据或恢复语义 | 对应配置、契约、模型/数据身份与针对性测试 |
| `course/` 中新增或修改公开文件 | `manifests/course_files.json` 中受影响文件的大小与 SHA |
| 修改课程原始受管源码 | 同时更新 `manifests/liftcube_source.json` 当前 SHA 与 `course/liftcube/patches/team-interface.patch`，保留原始 SHA |
| 入口、命令、依赖或执行规则改变 | 根目录及受影响子级 AGENTS，同 PR 更新；其他变更无需机械改日期 |
| 用户可见功能或用法改变 | 对应文档；首页信息同步中英文 README，重要变化写 CHANGELOG |

`manifests/projects.json` 和 `source_files.json` 固定上游身份。不要用 `audit_repository.py --write-manifests` 消除普通检查差异。来源检查按 Git 索引核对 `course/` 文件集合；获准提交新增文件时，文件与清单一起暂存再检查。新增团队说明不伪造课程原归档条目。

## 完成与交接标准

- 修改应有清晰入口：说明调用位置、所需配置、工作目录和实际依赖；命令中不得引用未提交的临时脚本。无法随仓库分发的文件给出来源、固定版本及获取条件。
- 修复应有针对原问题的验证，新增接口应覆盖有效输入与关键错误输入。测试复用现有框架与替身环境，不把真实GPU或硬件变成文档/纯函数检查的隐式依赖。
- 结果区分“已实现”“通过软件检查”“通过实际环境验证”。只有静态或合成测试时，明确缺少哪个实际验证；不要写笼统的“全部可用”。
- 交接提供改动摘要、实际检查及skip、剩余限制和下一步可执行入口。涉及实验时记录配置/模型/数据版本、seed、预算单位与输出位置，私有绝对路径不进入公共说明。
- 推送前检查暂存文件及相对目标分支的完整差异，确保源码、配置、文档、测试和来源清单一致。使用现有PR模板；用户自行创建PR时只推送指定分支，不代建或合并PR。

## 执行与公开边界

- 按实际使用者的设备、预算和平台规则执行。平台配置留在仓库外；缺少部署信息只阻塞相关运行。不要停止或重启无关/健康任务、覆盖产物或执行全局清理。
- 真机运行需实际 SDK、校准、限速、急停与用户确认。仿真通过不能替代真机安全条件。
- 不提交凭据、完整环境变量、私有配置、个人机构/服务器/账号/设备信息、本机路径、预约、个人运维规则或未获准材料。公开示例使用通用占位符或源码必需的容器路径；数据、权重和运行日志保留在外部存储。
- 发布前复核文件、提交正文及 Git 作者/提交者身份；保持用户同意的公开身份，不擅自改全局配置或重写历史。保留许可、版权与来源，访问许可不等于再分发许可，详见 [THIRD_PARTY_NOTICES](THIRD_PARTY_NOTICES.md)。
- AGENTS 仅保留可执行的项目约定，不要求特定 agent 插件，也不存实验成绩、PID、故障流水或机器配置。详细用户教程放既有文档；[PR 模板](.github/pull_request_template.md)用于交付复核。
