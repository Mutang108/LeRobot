<p align="center">
  <a href="assets/lerobot-tabletop-pushing-bilingual.svg">
    <img src="assets/lerobot-tabletop-pushing-bilingual.svg?v=20261003-pdf-aligned" width="960" alt="LeRobot 桌面操作双语海报">
  </a>
</p>

<h1 align="center">LeRobot：强化学习项目</h1>
<h3 align="center">视觉桌面操作：从仿真学习到 SO-101 真机部署</h3>

<p align="center">
  基于 LeRobot、LeIsaac 与 SO-101，探索视觉抓放、模仿学习和强化学习。
</p>

<p align="center">
  <a href="README.en.md">English</a> | <strong>简体中文</strong>
</p>

<p align="center">
  <a href="#快速开始">快速开始</a> ·
  <a href="#仓库结构">仓库结构</a> ·
  <a href="#开发路线">开发路线</a> ·
  <a href="#文档">文档</a> ·
  <a href="CONTRIBUTING.md">参与开发</a>
</p>

---

当前阶段推进 LiftCube 和 PickOrange 两条任务路线，分别建立策略、数据与评测流程。课程正式多物体环境与真机迁移留待后续。

## 当前范围

- **LiftCube**：提供环境适配、演示读取、BC、参考 PPO、视觉规则策略和公共评测入口。配套镜像与运行方法见 [LiftCube 指南](course/liftcube/README.md)。
- **PickOrange**：提供冻结环境源码、独立GR00T双视角服务与原生辅助评测源码、轨迹审核及ACT数据桥接诊断；新评测移植待真实环境验收。见 [PickOrange 指南](course/pickorange/README.md)。
- **VLA接口**：[SmolVLA、π₀.₅、GR00T N1.5和OpenVLA-OFT单front推理包装](docs/VLA_INTERFACES.md)位于LiftCube包；已有CPU检查，完整任务验收按后端推进。ACT及VLA训练接入仍在开发，PickOrange需要独立任务适配。
- **规则基线**：[LiftCube视觉抓取脚本](docs/SCRIPTED_LIFTCUBE.md)，使用RGB定位与逆运动学，作为后续标定和任务适配的开发基准；已有仿真抓取成功，跨场景稳定性仍待改进，需要独立标定。
- **后续任务**：正式多物体抓放、跨物体泛化与真机部署；保留迁移设计，当前不以未发布环境的适配为前提。

两任务的环境版本、专家脚本、演示数据及四种VLA的权重与验收进度见[任务与适配状态表](docs/TASKS.md)。同为SO-101不等于接口相同，配置、权重、统计与成绩分别管理。

获取模型与演示从[资源清单](docs/RESOURCES.md)开始；教师、训练及评测按[操作指南](docs/TRAIN_AND_EVALUATE.md)选择对应任务入口。LiftCube特权教师为新开发候选；PickOrange原生辅助使用物体真值夹爪控制，不作为无特权视觉baseline。

当前 LiftCube 使用单路 front RGB 与六关节控制，具体输入、动作和时序见[接口说明](course/liftcube/README.md#接口--interface)。策略仅使用图像与许可本体状态，物体和目标真值用于训练奖励或离线分析。

另提供可选的[受限末端开发接口](docs/PLANAR_EE_DEVELOPMENT.md)，用于固定参考高度和接近轴下的XY控制，物理定高精度尚未验收；它是自研执行层，不是课程正式IK，旧关节策略不能直接套用。

## 快速开始

获取代码并运行仓库检查（需要 Git 和 Python 3.11+）：

```bash
git clone -c core.longpaths=true -c core.autocrlf=false https://github.com/Nole326/LeRobot.git
cd LeRobot
python scripts/check_project.py
```

该检查不启动仿真或训练。运行 LiftCube 请按[镜像与环境说明](course/liftcube/README.md#运行环境--runtime)获取配套镜像；PickOrange使用[独立部署入口](docs/PICK_ORANGE.md)，不要混用两套依赖或启动脚本。

只运行规则基线无需演示或训练权重，但需要匹配的机器人与相机标定；从[规则脚本指南](docs/SCRIPTED_LIFTCUBE.md)的离线标定检查开始。BC/PPO训练另按[训练指南](course/liftcube/TRAINING.md)准备数据和接口。

项目版本见[基准说明](docs/BASELINE.md)与 [CHANGELOG](CHANGELOG.md)。参与开发请从最新 `main` 创建功能分支，流程见[贡献指南](CONTRIBUTING.md)。

## 仓库结构

LiftCube集成代码位于 `course/liftcube/src/starter/`；PickOrange运行入口位于`course/pickorange/`，诊断工具和上游任务源码分别保留：

```text
.
├── course/liftcube/
│   ├── src/starter/
│   │   ├── interfaces/        # 观测、动作、策略与模型契约
│   │   ├── envs/              # 仿真接入、控制与终态处理
│   │   ├── data/              # 演示读取、转换、校验与划分
│   │   ├── algorithms/        # BC、RL与参考PPO
│   │   ├── training/          # 配置驱动的训练入口
│   │   ├── evaluation/        # 共用评测与策略加载
│   │   ├── rewards/           # 训练奖励
│   │   └── logging/           # 指标记录
│   ├── configs/              # 当前任务配置
│   └── patches/              # 相对课程原包的可逆修改
├── course/scripts/           # PickOrange环境、重放、ACT桥接等诊断
├── course/pickorange/        # 独立GR00T原生辅助评测、模型服务与审核
├── configs/                  # 公共实验计划模板
├── scripts/                  # 仓库、来源与计划检查
├── tests/                    # 项目回归测试
├── third_party/              # 固定版本的上游源码
├── manifests/                # 源码、版本与哈希清单
├── docs/                     # 架构、接口、运行与复现说明
└── assets/                   # 海报与项目素材
```

模型接入与各层职责见[架构说明](docs/ARCHITECTURE.md)。镜像、数据与权重单独获取，来源和版本见[依赖说明](docs/DEPENDENCIES.md)。

## 开发路线

1. 分别冻结LiftCube与PickOrange的环境、输入、动作和成功判据，补齐各自运行入口。
2. 优先完成LiftCube规则、演示准入及ACT/小规模VLA对照；并行补齐PickOrange的任务适配与完整回合评测。
3. 根据已验收的数据和模型推进微调、RL及必要消融，记录完整回合和失败原因。
4. 正式多物体环境发布后，再恢复跨物体迁移、泛化和真机对齐。

BC与预训练VLA作为初始化或对照，RL为独立训练阶段。评测记录成功率、完成时间、误抓/漏抓和放置误差，具体成功判据及时间上限随任务环境确定。

VLA 强化学习实现参考 [RLinf](docs/RLINF.md)，目前尚未接入本项目运行环境。

### 上游版本

以下是历史源码快照；当前 LiftCube 使用课程配套的 Isaac Sim 4.5.0 / Isaac Lab v2.1.1 / LeIsaac v0.1.2，二者分开管理。

| 项目 | 历史源码版本 | 固定提交 |
| --- | --- | --- |
| LeRobot | v0.4.2 | [58f70b6](https://github.com/huggingface/lerobot/tree/58f70b6bd370864139a3795ac3497a9eae8c42d5) |
| LeIsaac | 0.4.0 | [24d3bcd](https://github.com/LightwheelAI/leisaac/tree/24d3bcd3f1e4585740fc79921782c41617237812) |
| IsaacLab | v2.3.0 | [3c6e67b](https://github.com/isaac-sim/IsaacLab/tree/3c6e67bb5c7ada942a6d1884ab69338f57596f77) |

源码来源与完整性信息见[导入清单](docs/IMPORT_MANIFEST.md)。

## 文档

- [开发起步与阶段验收](docs/GETTING_STARTED.md)
- [LiftCube与PickOrange：资源与适配状态表](docs/TASKS.md)
- [模型权重与专家演示获取](docs/RESOURCES.md)
- [按任务进行教师采集、训练和评测](docs/TRAIN_AND_EVALUATE.md)
- [SwanLab训练与评测曲线](docs/SWANLAB.md)：仅数值与脱敏配置，默认离线，在线使用私有项目
- [LiftCube 源码、镜像与运行方法](course/liftcube/README.md)
- [PickOrange 源码、独立环境与诊断入口](docs/PICK_ORANGE.md)
- [视觉规则基准、标定与后续适配](docs/SCRIPTED_LIFTCUBE.md)
- [受限末端开发接口与验证限制](docs/PLANAR_EE_DEVELOPMENT.md)
- [架构、扩展位置与团队协作](docs/ARCHITECTURE.md)
- [RLinf 对本项目的帮助与接入边界](docs/RLINF.md)
- [观测、动作、数据与恢复契约](docs/INTERFACE_CONTRACTS.md)
- [训练前验证与限制](docs/TRAINING_VALIDATION.md)
- [依赖与外部资产](docs/DEPENDENCIES.md)
- [部署与复现边界](docs/REPRODUCIBILITY.md)
- [源码导入清单](docs/IMPORT_MANIFEST.md)
- [修改与版本管理](CONTRIBUTING.md)
- [智能体开发规范](AGENTS.md)
- [海报与图像说明](assets/README.md)
- [第三方许可与版权](THIRD_PARTY_NOTICES.md)

## 参考与致谢

感谢 [LeRobot](https://github.com/huggingface/lerobot)、[LeIsaac](https://github.com/LightwheelAI/leisaac) 和 [IsaacLab](https://github.com/isaac-sim/IsaacLab)。工程分层参考 [rl-garden](https://github.com/Nole326/rl-garden)，VLA 强化学习方案参考 [RLinf](https://github.com/RLinf/RLinf)。相关许可见[第三方说明](THIRD_PARTY_NOTICES.md)。

海报为课程项目海报优化调整后的中英双语版。机器人图像来源为 [LeRobot SO-101 官方文档](https://huggingface.co/docs/lerobot/so101)。
