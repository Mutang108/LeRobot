# 任务、资源与适配状态 / Tasks, resources and adaptation

当前阶段分别推进 **LiftCube** 和 **PickOrange**。正式指定物体的多物体环境与真机迁移留待后续，不作为这两个开发任务的前置条件。PickOrange本身包含三颗橙子，但不等于尚未发布的课程多物体任务。

The current scope covers **LiftCube** and **PickOrange** as separate tasks. The later course multi-object environment and hardware transfer are deferred, not prerequisites. PickOrange contains three oranges; it is not the unreleased course multi-object task.

任务范围调整不自动取消报告中的方法对照、RL或消融要求；这些仍按课程当前要求确认。 / This scope change does not by itself waive method comparisons, RL or ablation requirements.

## 环境与接口 / Environments and interfaces

| 内容 / Item | LiftCube | PickOrange |
| --- | --- | --- |
| 指南 / Guide | [任务README](../course/liftcube/README.md) | [任务与部署指南](PICK_ORANGE.md) |
| 目标 / Objective | 抓取并抬升方块 / Grasp and lift a cube | 三颗橙子全部入盘并回位 / Put all three oranges on the plate and return to rest |
| 环境ID / Environment ID | `LeIsaac-SO101-LiftCube-v0` | `LeIsaac-SO101-PickOrange-v0` |
| 运行栈 / Runtime | 课程配套Isaac Sim4.5.0 / IsaacLab2.1.1 / LeIsaac0.1.2镜像，Python3.10.15 | 独立Isaac Sim/IsaacLab运行环境；公开冻结LeIsaac0.4.0 / IsaacLab2.3.0源码，诊断入口要求Python3.11覆盖层；无同等配套镜像锁 / Separate runtime and Python3.11 overlays; pinned sources, not a turnkey image |
| 代码 / Code | `course/liftcube/src/starter/` | `course/pickorange/`模型评测、`course/scripts/`诊断与`third_party/leisaac/`任务源码 / Separate evaluation, diagnostics and pinned task source |
| RGB | 包装后单front，默认64×64；规则可显式用匹配标定的128/256 / One resized front stream | 公开环境front+wrist，各480×640；桥接诊断另用128×128 / Two native views; separate bridge resizing |
| q与其他状态 / State | 实测q6；速度/末端兼容槽补零，上次发送目标另列 / Measured q6, declared padding and previous commands | 原生关节弧度及其他仿真诊断字段；策略输入须单独选许可字段 / Native joint radians and diagnostic fields; an explicit policy whitelist is required |
| 动作 / Actions | 六维绝对motor，适配器映射弧度 / Absolute motor targets mapped to radians | `so101leader`原生action manager有scale/offset；motor→绝对弧度→raw action须明确转换 / Explicit conversion through native scale/offset |
| 时序与时限 / Timing and horizon | 默认30Hz策略、60Hz环境两tick保持；参考640环境tick | 从实际`physics_dt`、`step_dt`和运行配置读取；20帧10Hz桥接只是诊断 / Read actual clocks and run limits; bridge timing is diagnostic only |
| 评分 / Scoring | 当前任务原生抬升判据；训练奖励另列 | 原生三颗XYZ盘区条件且机械臂回位；局部命中不算整任务成功 |

两套环境不能通过只更换task ID来互换。共享SO101名称不意味着相机、零位、动作接口、控制频率或成功条件相同。四回合、640tick、20帧或历史长时限都不是自动继承的期中评分协议。

Changing a task ID is not an environment port. Shared robot names do not establish matching cameras, calibration, actions, clocks or scoring. Starter and historical diagnostic budgets do not define the midterm evaluation protocol.

## 状态表的读法 / Reading the status tables

**整理日期：2026-11-02。** 以下区分公开代码、外部资源和项目已有验证记录；不是本次重新运行的结果。CPU验证沿用截至该日的记录，物理闭环结果来自此前的独立实验。模型与数据不随Git仓库分发，已准备的资产也不代表克隆仓库后即可运行。

**Snapshot: 2026-11-02.** Public code, external assets and prior project validation are listed separately. No new simulator or GPU run was performed for this inventory. Prepared assets are not bundled with the repository.

- **接口已接入 / Interface:** 软件字段、单位、视角与chunk管理已对接，不代表模型会完成任务。
- **CPU验证 / CPU validation:** 按行注明是预处理、权重加载还是完整前向；三者不等价。
- **实际闭环 / Task rollout:** 模型或脚本已经控制仿真物理执行；一次成功不等于稳定成功率。
- **任务权重 / Task checkpoint:** 外部作者针对该任务训练的权重；本表不声称本项目已完成VLA微调。

### 脚本与演示 / Scripts and demonstrations

| 资源 / Resource | LiftCube | PickOrange |
| --- | --- | --- |
| 视觉脚本 / Visual script | **已有公开实现与实际评测。** [规则策略与评测入口](../course/liftcube/README.md)。既有100回合结果为45/100，条件为front 256×256、q6及显式相机标定辅助；不能称为不使用标定的纯视觉结果，也不是高可靠专家。 / Implemented and evaluated with explicit calibration assistance. | **尚无完整公开、已验收的自研视觉策略。** 现有任务诊断入口不能代替视觉baseline。 / No complete accepted first-party visual baseline. |
| 特权state脚本 / Privileged teacher | **已有SO101开发教师源码与CPU测试，真实成功率待验。** [PrivilegedLiftCube](../course/liftcube/src/starter/privileged_liftcube.py)显式读取cube真值，复用原控制保护；[执行与采集准入](TRAIN_AND_EVALUATE.md)。上游Franka状态机不直接适用。 / Implemented development candidate; physical grasp acceptance pending. | **已有官方采集状态机源码。** [任务状态机](../third_party/leisaac/source/leisaac/leisaac/datagen/state_machine/pick_orange.py)与[生成入口](../third_party/leisaac/scripts/datagen/state_machine/generate.py)已收录；其直接写关节状态等限制见[采集指南](../course/pickorange/README.md)。另有历史单场景教师成功记录，不能作为本入口批量BC准入。 / Included upstream generator, with physics/state-write caveats. |
| 网络演示 / Online demonstrations | **有候选资源。** [azazdeaz/teleop_so101_leisaac_lift_cube](https://huggingface.co/datasets/azazdeaz/teleop_so101_leisaac_lift_cube)：17回合、front+wrist、30fps。已核对元数据、统计与任务文本；未完成全量下载、逐回合成功和当前环境重放验收。 / Metadata checked; full replay and training admission remain open. | **有官方资源。** [LightwheelAI/leisaac-pick-orange](https://huggingface.co/datasets/LightwheelAI/leisaac-pick-orange)：60回合、front+wrist、30fps。已有样例和接口检查，不代表60回合均已独立复核成功或可直接用于BC。 / Official dataset exists; sample checks are not whole-dataset acceptance. |

特权教师使用物体真值，只能作为另行声明的采集工具，不能冒充视觉规则baseline。官方PickOrange生成流程会改变重力、夹爪控制及终止方式，并在setup/回位/成功检查中写关节状态；使用前须与冻结的评测环境逐项对齐，不能仅依据状态机结束就认定原生任务成功。

Privileged teachers are data-generation tools, not visual baselines. The official PickOrange generation setup can change gravity, gripper control and termination; audit those differences against the evaluation contract.

历史候选`H2Ozone/lift_cube`另有动作范围不匹配记录：约62.64%的帧至少一维请求超出当前声明范围。这不是直接裁剪原标签的理由；其映射、标定和原生请求/执行时限幅对照重放尚未完成，不能与上表17回合数据集混为同一份数据。

The older `H2Ozone/lift_cube` candidate has a separate action-range mismatch. Its pending mapping and replay checks must not be attributed to the 17-episode dataset above, or resolved by silently clipping labels.

### 四种VLA / Four VLA families

| 任务 / Task | 模型与候选权重 / Model and checkpoint | 已有资产与验证 / Assets and validation | 公开仓库接入与缺口 / Public integration and remaining work |
| --- | --- | --- | --- |
| LiftCube | **GR00T N1.6任务权重**：[azazdeaz/groot_n1_6_so101_leisaac_lift_cube](https://huggingface.co/azazdeaz/groot_n1_6_so101_leisaac_lift_cube) | 完整任务导出已准备；原生state/action变换CPU往返通过。**尚未完成完整模型加载、前向或物理闭环。** / Export prepared; CPU transform round-trip only. | 仓库包装器面向**N1.5**，不能直接加载N1.6。须补匹配后端，并处理原双相机训练与当前单front输入差异；权重再分发许可待确认。 / N1.6 backend and view compatibility remain open. |
| LiftCube | **SmolVLA任务权重**：[azazdeaz/smolvla_so101_leisaac_lift_cube](https://huggingface.co/azazdeaz/smolvla_so101_leisaac_lift_cube) | 完整权重及原统计已准备，统计与源数据核对；严格CPU加载及front 64×64＋q6的**50×6完整前向通过**。 / Strict CPU load and full forward passed. | 已有单front包装与公共chunk管理。模型原用front+wrist；缺失视角的性能影响、GPU及LiftCube闭环**未验收**。 / Single-view performance and task rollout remain unverified. |
| LiftCube | **π₀.₅ SO101域适配基座**：[dongyoonkim/so101-pi05-base](https://huggingface.co/dongyoonkim/so101-pi05-base) | 完整权重、内嵌统计已准备；作者模型图严格CPU加载及front 64×64＋q6的**50×6前向通过**。未确认可直接采用的同任务专用权重。 / Embodiment base, not an accepted LiftCube specialist. | 已有作者模型图专用包装，不等于通用LeRobot π₀.₅加载器。仍需GPU与LiftCube闭环；会输出六关节动作不代表会抓方块。 / GPU and task capability remain unverified. |
| LiftCube | **OpenVLA-OFT**：未确认匹配的完整任务权重 | 通用OpenVLA核心及处理器CPU加载通过；六维state/action、chunk30的接口合同与测试已有。**缺少匹配且受训的OFT动作头、状态投影及统计组合。** / Core load and contract tests, not a trained OFT policy. | 已有OFT包装入口；未完成匹配策略前向或任务闭环。不能将通用OpenVLA核心、随机初始化头或其他架构当作可用SO101-OFT策略。 / A compatible trained full package is still required. |
| PickOrange | **GR00T N1.5官方任务权重**：[LightwheelAI/leisaac-pick-orange-v0](https://huggingface.co/LightwheelAI/leisaac-pick-orange-v0) | 完整权重、GPU及真实物理闭环已有历史验收。固定60秒、100回合为**25/100**；选用的原生辅助协议为**52/100**，两组不可合并。 / Historical task validation under two distinct protocols. | [双视角服务、完整评测与审核源码](../course/pickorange/README.md)已整理；新移植有CPU合同检查，尚未重新GPU/仿真验收，不自动继承历史52%。 / Public source port and CPU checks; new runtime validation pending. |
| PickOrange | **SmolVLA任务权重**：[edge-inference/smolvla-so101-pick-orange](https://huggingface.co/edge-inference/smolvla-so101-pick-orange)、[wsagi/SmolVLA-PickOrange](https://huggingface.co/wsagi/SmolVLA-PickOrange) | 两份完整权重及原统计已准备，GPU与双RGB闭环已有历史记录。Edge有一次整任务成功；Wsagi所测长回合超时。**没有这两份权重的完整100回合成绩。** / Task rollouts exist, not an accepted success-rate estimate. | LiftCube单front包装不等于PickOrange双视角runner；需整理专用服务、完整回合及终态审计入口。 / Dual-view task integration remains separate. |
| PickOrange | **π₀.₅任务权重候选**：[wsagi/Pi0.5-PickOrange](https://huggingface.co/wsagi/Pi0.5-PickOrange)；另有上述SO101域基座 | 任务专用模型已发现，**未下载或完成CPU/GPU适配**；作者标注负面结果，不能假定可用。SO101域基座另有历史短时闭环，但条件延长100回合组仍为**0个完整回合**，不是0%成功率。 / Task checkpoint discovered only; separate base-model partial runs are not completed evaluation. | 任务候选需独立核对模型图、处理器和统计，不能复用域基座加载器即宣称通过。完整公开runner也待整理。 / Checkpoint-specific integration and task validation remain open. |
| PickOrange | **OpenVLA-OFT**：未确认匹配的完整任务权重 | 可复用核心CPU检查与SO101软件接口准备；没有匹配OFT策略的GPU或完整任务验收。历史普通OpenVLA短测**不属于OFT验证**。 / Software preparation only; vanilla OpenVLA evidence does not validate OFT. | 同样缺匹配受训动作头、状态投影、统计及核心权重组合，之后再接PickOrange双视角/动作合同。 / Trained package and task-specific runner remain open. |

单相机GR00T N1.5候选的局部投影/预处理检查，不等于上表LiftCube N1.6任务模型完整验收。通用SO101-OFT候选[Atabaku/so101-openvla-oft-5000](https://huggingface.co/Atabaku/so101-openvla-oft-5000)需要访问批准，尚无可验收的完整权重，也未确认与任一任务匹配。“未确认找到”不表示网络上绝对不存在。

Partial N1.5 component checks do not validate the N1.6 task checkpoint. The gated SO101-OFT candidate is neither an accepted full package nor a confirmed task match. “Not confirmed” is not a claim of nonexistence.

模型代码、外部权重和经过验收的任务策略是三件事。外部模型即使同为SO101，也要核对训练视角、关节单位、归一化统计、参考动作语义和模型图。后端细节见[VLA接口指南](VLA_INTERFACES.md)；PickOrange入口边界见[任务指南](PICK_ORANGE.md)。

Code, external checkpoints and accepted task policies are distinct. Validate views, units, normalization, action references and architecture. See the [VLA interface guide](VLA_INTERFACES.md) and [PickOrange guide](PICK_ORANGE.md) for integration boundaries.

### 其他方法代码 / Other method implementations

| 方法 / Method | LiftCube | PickOrange |
| --- | --- | --- |
| SmallCNN BC/PPO | 已有参考训练、存档与评测代码，真实学习能力另验 / Reference trainers and evaluation | 未接入这些训练器 / Not integrated |
| ACT | 冻结LeRobot实现可复用，当前任务训练包装待接 / Upstream implementation, task integration pending | 有20帧格式/CPU前向及优化恢复诊断；非受训任务策略 / Diagnostic bridge, not a trained task policy |

## 当前开发顺序 / Development order

1. 分别冻结任务、观测、动作、物理与评分配置；沿各自入口检查真实RGB/q及终态。 / Freeze each task's own contracts and verify observations and terminal capture.
2. LiftCube优先形成可复现的规则结果、合格演示以及ACT/SmolVLA对照；其他VLA按匹配程度接入。 / Establish rule and ACT/SmolVLA comparisons with accepted demonstrations.
3. PickOrange先验收已整理的GR00T原生辅助服务、完整回合与终态，再接其他模型；教师采集与固定物理BC准入另审，不把诊断或状态写入成绩当正式评测。 / Validate the source port before additional models and teacher-data admission.
4. 两任务独立配置、数据划分、统计、checkpoint和结果目录；比较时保留任务、视角、控制与预算信息。 / Keep data, normalization, checkpoints and evaluation records separate.

多物体迁移方案保存在[后续路线](LIFTCUBE_TRANSFER.md)。它不触发额外训练，也不要求先完成未知正式环境的适配。

See the [deferred transfer plan](LIFTCUBE_TRANSFER.md) for later work; it does not launch additional training.

资源下载、固定revision见[资源清单](RESOURCES.md)，教师与训练/评测步骤见[使用指南](TRAIN_AND_EVALUATE.md)。 / See [resources](RESOURCES.md) and [training/evaluation](TRAIN_AND_EVALUATE.md) for actionable next steps.
