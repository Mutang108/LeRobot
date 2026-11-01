# LiftCube智能体指令

适用于本目录及其子目录，与[根AGENTS](../../AGENTS.md)同时遵守。表中的源码路径相对 `src/starter/`，配置与文档路径相对本目录；验证命令另行注明工作目录。这是SO101方块抬升起点。

## 1. 按任务读取

| 任务 | 必读文档与代码 |
| --- | --- |
| 环境、镜像、首次运行 | [README](README.md)；`envs/lift_cube.py`、`envs/factory.py`、`envs/render_ready.py`、`envs/timing.py`，`configs/lift_cube.yaml` |
| 数据与BC | [TRAINING](TRAINING.md)；`data/hf_adapter.py`、`data/schema.py`、`data/validator.py`、`data/loader.py`，`training/train_bc.py`，`configs/bc_lift_cube.yaml` |
| PPO、存档恢复 | [TRAINING](TRAINING.md)；`training/train_rl.py`、`training/checkpoint.py`，`algorithms/reference/ppo.py`、`rewards/lift_cube.py`，`configs/rl_lift_cube.yaml` |
| 策略与评测 | `interfaces/policy.py`、`interfaces/model.py`，`utils.py::build_policy`，`evaluation/evaluate.py`，`configs/eval_lift_cube.yaml` |
| 视觉规则脚本 | `scripted_liftcube.py`、`joint_chain.py`、`scripts/export_joint_chain.py`、`scripts/evaluate_scripted.py`、[标定与验收](../../docs/SCRIPTED_LIFTCUBE.md)；每回合新建状态机，沿用共同`run_episode`，不接训练协议 |
| 受限末端开发接口 | `envs/planar_ee.py`、`envs/factory.py`、[契约](../../docs/PLANAR_EE_DEVELOPMENT.md)；`control_mode: planar_ee_dev`仅自研开发，不当课程正式控制器 |
| ACT/VLA或任务迁移 | [架构](../../docs/ARCHITECTURE.md)、[迁移路线](../../docs/LIFTCUBE_TRANSFER.md)、[契约](../../docs/INTERFACE_CONTRACTS.md)及上面的策略/环境边界 |

## 2. 常见开发任务怎么推进

开发时按下面的顺序推进；每一步只运行其所需检查：

- **修环境或动作接口**：沿 `envs/factory.py → LiftCubeEnvAdapter.reset/step → Observation/ActionSpec` 追踪数据；先用 `tests/test_so101_adapter.py` 的替身环境覆盖单位、形状与边界，再补真实环境短测。
- **运行规则基准**：走`starter.scripts.evaluate_scripted`，先按规则指南检查外部标定；不需要演示、训练权重或`build_policy()`。不要为此运行BC/ACT优化验证。正式标定迁移只修改读取/坐标边界，控制SDK改变时再处理执行层，复用未变验证。
- **接入演示数据**：沿 `data/hf_adapter.py → schema/validator → loader → training/train_bc.py` 检查字段、episode划分和标签单位；先验证一小批真实样本，再做获准的小批拟合。合成单测不替代真实数据映射验收。
- **新增 ACT/VLA**：在实际接入时新增 `starter/models/` 包装，核对 `interfaces/policy.py` 中 `act/forward/update/save/load` 及调用者。推理先接 `evaluation.evaluate.load_policy` 的 `--policy-class`；训练再接 `utils.build_policy`、数据预处理和对应算法。现有BC/PPO依赖SmallCNN的具体实现，不能仅满足协议或替换模型名就宣称可训练；后端依赖保持可选。
- **处理动作chunk**：`act()`向执行层返回当前一个六维目标，预测chunk可在包装内缓存；规格中的策略频率须与 `envs/timing.py` 一致。公共`run_episode`在两端和异常退出调用可选`policy.reset()`清缓存、不重seed；旧无reset策略兼容。复用`models/joint_chunk.py`；任务文本变化走`set_task()`，接管后显式reset，不能跨回合沿用旧动作。
- **SmolVLA单front推理**：见[推理接口](../../docs/VLA_INTERFACES.md)和`tests/test_vla_interfaces.py`。本地适配JSON通过`--policy-class starter.models.smolvla:SmolVLAFrontPolicy`加载；相机首槽、缺槽mask、六维motor和显式统计严格核对。CPU替身/mask检查不等于完整权重或GPU/仿真通过；不将新包装接入SmallCNN训练器。
- **其他VLA包装**：`models/pi05.py`保留SO101 VLASH内嵌统计/state_cond，`models/gr00t.py`使用单图模态和受检形态加载，`models/openvla_oft.py`要求六维SO101常量及已训练L1头/本体投影。加载描述和固定后端见上述接口文档；不得以假头、三图改名或HF不同模型图代替。先运行`test_vla_native_backends.py`，大型backend导入/完整模型另验。
- **修改VLA加载器**：先校验描述中的身份、单front观测与执行参数，再导入大型backend或分配模型；保留严格加载错误的`loading_info`。张量输出转FP32后才能交NumPy（BF16不受支持），不借转换裁剪或截维度。OFT CLI需显式`--robot_platform SO101`，其后端在首次导入时读取真实进程参数；程序调用者须自行保证常量已正确初始化，不在加载器中改全局常量或`sys.argv`。
- **新增完整抓放任务**：扩展 `envs/`、`envs/factory.py` 和独立任务配置，复用公共数据/策略边界；接入实际任务的观测与评分，保留LiftCube配置和回归检查。旧权重仅在规格兼容时初始化新实验，不能当作同任务resume。

新模型的最小交付包含：可加载的策略、真实输入/动作映射、独立配置、相关测试、保存加载说明及依赖/权重获取方式。先验证前向与保存加载，再验证获准的少步闭环；只有完成优化和恢复检查后才标为支持训练。

### ACT/VLA接入的具体检查

1. 固定模型仓库/revision、tokenizer或processor、归一化统计及许可。确认权重是否针对SO101训练、训练时的相机和动作定义；“下载成功”不代表当前任务可用。
2. 包装只接收公共 `Observation` 及明确允许的任务条件。记录图像通道/尺寸/数值范围、q顺序与单位；需要额外相机或真实速度而当前接口不提供时，列明不兼容，不能复制图像或补造状态冒充满足要求。
3. 在模型边界完成预处理和反归一化，在公共执行层完成动作范围处理。检查最终目标为六维有限数值，分别记录模型输出、发送目标与反馈；不得重复归一化或让两个控制层各保持一次动作。
4. 先用固定输入检查前向形状/有限值、加载缺失键、保存后重载及chunk消费顺序。若涉及微调，列明训练参数、冻结模块、损失和优化器归属；单步有梯度、有参数更新后，再验证完整恢复。
5. 使用公共评测入口进行获准闭环，保持环境、场景与评分协议一致。新模型确需不同执行频率时，显式记录并测试，不把不同预算的结果合并为同一比较。

### 数据接入的完成条件

数据清单需说明版本、帧率、图像键、关节顺序/标定、动作含义、episode边界及成功标签来源。检查真实样本的缺帧、非有限值、动作范围和观测→动作时间对应；划分按episode或任务要求的物体/场景组进行，不随机拆相邻帧造成泄漏。归一化统计只从训练部分拟合，验证/测试复用冻结统计。动作超限与请求未到位是不同问题，先定位标定/执行原因，不用裁剪标签提高通过率。

接入后至少能读取一小批真实样本，得到与策略契约一致的输入和标签，并给出每个划分的episode数量及排除原因。数据身份或处理逻辑未变时复用已有检查；只有新增、变更或完整性存疑部分需要重新验证。原始数据保留，转换产物独立输出。

## 3. 运行前提

- 镜像 `isaac-lab:4.5.0-v2.1.1-leisaac-v0.1.2-src`，归档SHA与image ID见README。内部Python 3.10.15、Torch 2.5.1+cu118；包元数据与源码标签可能不同，核对真实来源，不为标签升级。
- 使用配套运行环境，示例约定仓库挂载 `/repo`，演示 `/data/lift_cube`，输出到当前使用者可写位置。保留镜像内 `/isaaclab`、`/leisaac`；缓存与数据分开。按使用者的平台核对设备映射、RAM、磁盘和GPU资源。
- `envs/container.py`可在宿主自动启动Docker，硬编码GPU0/设备/默认容器名并传入许可同意变量。先核对是否适合当前平台；已有兼容容器时在其中执行，不自行启动额外容器或代用户同意许可。
- 使用镜像已有解释器，先核对NumPy/Torch/PyYAML、TensorBoard；数据读取另需PyArrow/PyAV。缺包不授权重装模拟器或Torch/CUDA。
- 配置 `device: cpu`可能只控制策略，`env.sim_device: cuda:0`仍消耗仿真GPU；不得标成纯CPU操作。

## 4. 最小闭环入口

仿真/GPU任务已获准且前提齐全后，在容器内运行；不需要为已授权的这一检查重复询问：

```bash
cd /repo/course/liftcube
export PYTHONPATH="$PWD/src:/leisaac/source/leisaac${PYTHONPATH:+:$PYTHONPATH}"
/isaac-sim/python.sh -m starter.scripts.smoke_test \
  --config configs/lift_cube.yaml --steps 16 --log-dir /approved/new-smoke-output
```

`/approved/new-smoke-output`是占位，先换为获准、未使用的输出目录。此命令执行16个随机仿真动作并写 `smoke.json`，不训练、不证明任务成功。通过需 `status: ok`、实际完成请求步数、观测规格/有限值与稀疏奖励断言通过。`skipped_isaac_unavailable`是未执行；异常保留输出，不循环重启或改依赖。

先从源码核对CLI再执行。部分入口在解析 `--help` 前调用自动容器逻辑，不能在宿主用它探测。训练/恢复命令见 [TRAINING](TRAINING.md)，默认小预算也不构成开训授权。评测入口是 `starter.evaluation.evaluate`，默认配置 `configs/eval_lift_cube.yaml`；`--checkpoint`读推理导出，`--policy-class package.module:ClassName`要求实际可导入且实现约定的加载器，不提供自动模型适配。

## 5. 接口修正时守住的语义

- 环境ID `LeIsaac-SO101-LiftCube-v0`。单front、uint8 HWC，参考配置默认64×64×3；规则可选128/256需按下文同步标定。相机原分辨率与模型resize分开。新模型更换输入须显式更新数据/模型规格，不直接改正式评分口径。
- 本体q[6]、dq补零[6]、上一发送命令[6]、可选EE补零[3]，总维18或21。SmallCNN只融合图像、q和上一命令；补零不是实际可测状态。
- 关节顺序在 `interfaces/action.py::JOINT_NAMES`；六维绝对电机目标，arm[-100,100]、gripper[0,100]，逐关节映射弧度。不是增量、物理角度或末端动作。
- 外部raw环境的 `observation.state`别名用 `raw_joint_units`声明单位；必要q/图像无效就拒绝，不补造反馈或重复换算。
- `info.command`的 `requested_motor`、`applied_motor`、`applied_radians`、`clipped`、`measured_motor`分开核对；applied是发送值，不是到位值。
- `planar_ee_dev`是独立三维绝对桌面XY米＋夹爪motor接口，固定参考高度与接近轴、允许绕轴转动；不是完整姿态锁定或课程正式动作。30Hz求解/原60Hz两tick保持只执行一次，不改物理/评分；标定缺失、不可达或超范围拒绝，IK/跟踪故障锁定保持。旧六维策略/演示不能改名接入；上一关节命令观测保持原语义。改此路径运行`test_planar_ee.py`，涉及工厂默认值时补旧接口回归。
- 此开发接口的`tracking_compensation`须显式启用；只在静止名义参考上积分q误差、偏置限±3motor，不补偿夹爪。名义参考、偏置后发送目标和实际反馈分开；平面约束不能由发送目标或未触发关节保护来冒充已物理满足。真实精度未验收前不标记正式可用。
- 实际环境周期读 `step_dt`。当前60Hz环境、30Hz策略保持2tick；适配器step、决策步与chunk分别计数，终止立即结束保持。
- 当前默认评测4回合/640环境tick只是起点。训练奖励、原生success和正式评分分开；不改USD、限位、执行器、reset或判据来制造成功。
- 渲染准备不额外推进物理/策略动作。真实终态在自动reset前捕获，物体真值不流入策略观测或metadata。
- 适配器关闭上游默认演示记录器，避免每步在GPU上拼接完整RGB历史；评测轨迹与图像由显式外部记录器保存。需要原生演示导出时使用独立采集入口，不在普通评测中隐式开启，也不靠周期清缓存掩盖增长。
- 规则策略可单独标定`lift_rotation`，并显式启用`gravity_compensation`。实测反馈不按待发送动作裁剪；名义参考、积分偏置与实际发送值分开，原发送范围/限速/跟踪保护不变。不同IK后端须验证实际路径，不能把独立USD诊断后端的成绩当作Placo入口已通过。
- 可选`contact_close`仅用实测夹爪位置限制闭合领先量；停滞代理不是力传感器或抓取成功信号，抬升后仍按原生评分判定。复用已验证几何，独立冻结测试seed/预算；不得混合调参回合、重选失败或更改物理来达到目标比例。
- 规则入口默认front64，也可显式配置同一路front128/256：环境resize、标定`image_shape/K`必须一致，像素中心随resize换算，不能只放大旧图伪称新视觉信息。其他模型默认输入不变；跨分辨率成绩分协议记录。实际策略类与基类源码分别留SHA，诊断子类成绩不冒称默认入口成绩。
- 非空`grasp_presets`显式选择`starter.scripted_geometry.GeometricLiftCube`，要求joint_chain与有界补偿；固定CAD姿态选择不读取物体真值。离桌子阶段只按实测q的FK高度/朝向连续三次切入运输，≤0.25motor/步；它不是抓取成功，原发送范围、8motor保护、最后评分不变。预设示例只适用于其资产SHA与3cm方块假设，不能作为完整标定直接启动。
- `plan_while_opening`默认false且仅用于上述几何模式；启用后在原开爪期间积累三次稳定RGB观察，完整开爪命令和实测到位等待保持，不提前移动手臂。视觉未就绪则走原定位流程；开关写入manifest，成绩分组。修改此路径至少检查开爪前缀、视觉缺失回退及关闭时行为不变，不能把调度提速当抓取精度改善。
- `--runtime-camera-calibration`仅显式启用配套模拟器的相机/基座标定接口，每次reset记录实际变换；相机随机化后的`camera.data`缓存可能仍是名义值。此模式与固定外参分组，真机接口支持待确认，不把额外标定输入藏进普通RGB/q成绩。`--check`仍不创建环境。
- 固定关节链后端用NumPy/SciPy和来源可核验的五轴SO101 JSON，原生轴向与零偏置，不读实时物体状态；导出只用USD Python API，不启动SimulationApp。URDF/Placo仍为可选后端。标定IK回算使用每个参考点自身的朝向，各阶段预检显式传入阶段朝向，不按高度猜测。

## 6. 数据、模型、恢复

- 仓库无演示/训练权重。`H2Ozone/lift_cube`只是候选数据；先核对来源版本、关节顺序/标定/零位、动作范围、同步与成功证据，不因同名通过验收。
- `data/hf_adapter.py`读取本地LeRobot v2.1/v3，不下载。BC需真实 `meta/info.json`、兼容图像/状态/动作与episode级划分。文件末尾的终止标记不等于任务成功标签。
- 超限先留原值诊断；`allow_action_clipping`仅诊断，正式BC拒绝启用。禁止放宽范围、裁剪标签或用下一帧q替代请求动作来通过检查。
- `build_policy()`固定SmallCNN。ACT/VLA包装须真实处理图像、tokenizer/统计、本体、动作语义与执行长度，不能只加模型名或随机权重就称完成基线。
- 当前BC/PPO验收是默认六关节路径；三维`planar_ee_dev`不自动继承其数据、权重或恢复验收，不能只更改模式就宣称支持末端训练。
- `checkpoint.pt`是推理导出；`checkpoint.pt.training.pt`及SHA是完整续训。BC→PPO只复制规格匹配权重，新建优化器；同任务resume恢复原模型/Adam/计数/数据身份/RNG。
- BC完整epoch保存、当前 `num_workers=0`；PPO在完整采集回合/更新边界保存。不承诺任意中途仿真接续。加入私有RNG、异步worker、外部VLA缓存前扩展恢复合同。
- epochs/updates是总目标，不是追加量。新任务、结构或数据身份另开实验，权重初始化不伪装resume。只加载可信本地存档，SHA不等于来源认证。
- 新输出位置与旧实验隔离，保留日志/存档。TensorBoard用真实事件，优化更新、决策步与物理tick分别记录。

## 7. 改动后的最小验证

命令在仓库根目录、具备相应依赖的环境执行；这些软件单测不要求模拟器或特定服务器，缺少可选依赖的跳过须说明：

```bash
python -m unittest discover -s tests -p test_so101_adapter.py -v
python -m unittest discover -s tests -p test_dataset_validation.py -v
python -m unittest discover -s tests -p test_liftcube_training.py -v
python -m unittest discover -s tests -p test_liftcube_import.py -v
python -m unittest discover -s tests -p test_scripted_liftcube.py -v
python -m unittest discover -s tests -p test_joint_chain.py -v
python -m unittest discover -s tests -p test_planar_ee.py -v
python -m unittest discover -s tests -p test_vla_interfaces.py -v
python -m unittest discover -s tests -p test_vla_native_backends.py -v
```

- 观测/单位/命令映射优先SO101测试，使用替身环境，不等于Isaac验证。
- 数据字段/逐帧元数据变更运行dataset validation测试；未知success保留未知，执行限幅诊断不修改训练标签。配对重放边界见[验收说明](../../docs/SO101_ACCEPTANCE.md)。
- 模型/BC/PPO/恢复选训练测试，覆盖合成数据、动作保持、GAE、下一更新恢复；说明依赖缺失导致的跳过。
- 导入/补丁/配置选导入测试；仅改说明不触发模型GPU回归。
- 实际环境路径变化再补授权下reset/step/RGB/终态最小集成检查，复用未受影响证据。
- 规则脚本的`--check`需匹配URDF与独立标定；空模板必须拒绝。CPU合成通过不表示真实IK/抓取通过。运行必须显式`--run-simulation`，禁止使用特权物体坐标补齐视觉失败，不擅改默认评测预算或固定物理。

故障按层定位：导入失败先检查解释器和模块来源；观测/动作失败先检查形状、单位和标定；恢复失败先核对存档类型、SHA及配置身份；加载成功但无任务进展则分析实际图像、模型输出与执行反馈。保留首个有效报错，不用切换模型、放宽评分、重装环境或反复重跑来掩盖未知原因。

修改受管源码同步来源清单、可逆补丁、契约和测试，具体分工见根AGENTS。命令、前提或接口改变时，同步本文件及根任务分流。
