# PickOrange与独立诊断工具：智能体指令

适用于本目录，先读[根AGENTS](../../AGENTS.md)、[PickOrange指南](../../docs/PICK_ORANGE.md)及[任务对照](../../docs/TASKS.md)。PickOrange与LiftCube均为当前开发任务；本目录仍只含PickOrange仿真诊断及可复用软件工具，不是完整训练/评估套件。LiftCube走[专属AGENTS](../liftcube/AGENTS.md)，不要混装两套环境或只换task ID迁移策略。正式多物体环境的接入留待后续。

PickOrange完整GR00T原生辅助开发评测、服务及轨迹审核走[独立目录](../pickorange/AGENTS.md)，不用本目录20帧诊断拼正式runner。数据/权重与训练流程见[资源清单](../../docs/RESOURCES.md)、[使用指南](../../docs/TRAIN_AND_EVALUATE.md)。

## 1. 工具选择与副作用

| 文件 | 实际用途 |
| --- | --- |
| `inspect_dataset.py` | 只读指定LeRobot `meta/info.json`并打印；不解码全数据、不判断标定或成功 |
| `environment_inventory.py` | 输出解释器/包身份；共享前复核路径，不能输出秘密 |
| `prepare_assets.py` | 下载/复制固定官方厨房与SO101 USD，核验、解压并写 `/data/course/assets`；不是只读探针 |
| `course_python.sh` / `course_bc_python.sh` | 分别包装历史仿真/BC解释器与覆盖层，设置固定资产/缓存路径 |
| `validate_cuda.py` | 实际CUDA检查，写固定日志，消耗GPU |
| `validate_sim.py` | PickOrange reset/65步/双RGB，写固定JSON和图片；不是成功率测试 |
| `validate_rollout.py` | 回合边界、终态、20帧10Hz受控轨迹、重放及IK数学检查；写新tag日志/轨迹，不是专家演示 |
| `validate_bridge.py` | 上述20帧转v3、CPU ACT前向和动作单位检查；不优化、不部署未训练ACT |
| `validate_act.py` | 合成CUDA ACT/视频检查，写固定输出；无任务训练权重或闭环成绩 |
| `validate_dataset.py` | 向固定目录写合成v3数据并检查读取/padding；不是采集 |
| `training_contracts.py` / `validate_training_readiness.py` | 独立终态/RNG工具及20帧样本的少量ACT优化/跨进程恢复，见[专项说明](../../docs/TRAINING_VALIDATION.md) |
| `joint_contract.py` / `validate_joint_contract.py` | 历史动作换算与CPU检查；固定标定不自动适用新任务 |
| `validate_ee_control.py` / `analyze_ee_motion.py` | 开发用基座XY控制/分析，不是课程桌面控制器或现成专家教师 |
| `audit_install_plan.py` | 读既有安装计划、写约束；不是自动安装器或通用依赖锁 |

不要批量运行所有 `validate_*.py`。多份脚本在模块顶层就解析参数、写文件或创建应用，不能通过import来无副作用探测。CPU授权不覆盖CUDA/仿真，读取授权不覆盖下载/解压/覆盖。

修改这些工具时，先从CLI参数、输入文件、固定输出路径和主流程确定影响范围。读取元数据只选 `inspect_dataset.py`；动作换算先选 `joint_contract.py` 及对应检查；轨迹到数据集的转换按rollout→bridge顺序；少量优化/恢复再进入training-readiness。前一阶段已有身份一致的合格产物就直接复用，不为运行后一步重做整条链。

## 2. 历史部署前提

读[部署约定](../../docs/REPRODUCIBILITY.md)和[依赖](../../docs/DEPENDENCIES.md)，核对以下实际路径/版本：

- 工具 `/workspace/course/scripts`；模拟器源码 `/workspace/course/leisaac/source/leisaac`；覆盖层 `/workspace/course/venv_sim`、`venv_bc`。
- 包装脚本含Python 3.11路径，调用 `/isaac-sim/setup_python_env.sh`、`python.sh`。不是Python 3.10 LiftCube镜像的可复用配方。
- 数据/资产 `/data/course`，日志 `/workspace/course/logs`，缓存 `/workspace/course/cache`；缓存不写数据集，预先核对权限与已有文件依赖。
- 实际导入须为预期固定版本，不能被同名namespace遮蔽。PickOrange注册ID `LeIsaac-SO101-PickOrange-v0`，源码在 `third_party/leisaac/source/leisaac/leisaac/tasks/pick_orange/`。
- `course/constraints-isaac51.txt`注明有冲突的历史诊断约束，不是当前安装配方；其中未随仓库分发的参考文档不能作为必需步骤。按公开依赖与实际环境检查缺口，不据此重装。
- 场景资产不在Git，来源见 `manifests/external_assets.json`和 `prepare_assets.py`。已验SHA且未变的资产复用，下载/解压和运行授权分开确认。

这些路径是脚本当前的容器内接口，不是某个人的宿主目录。不同平台可用挂载或独立适配满足接口；版本/路径不符先列明差异，不依赖未发布部署记录，不强行合装LiftCube。

## 3. 已配置环境的执行入口

最小只读起点，在仓库根目录运行（替换数据路径）：

```bash
python course/scripts/inspect_dataset.py /approved/dataset
```

GPU仿真仅在前提齐全且获准后，于容器内执行；`NEW_TAG`换成未使用的名称：

```bash
cd /workspace/course
bash scripts/course_python.sh scripts/validate_rollout.py \
  --run-tag NEW_TAG --headless --enable_cameras
```

创建 `/workspace/course/logs/NEW_TAG`、`/data/course/validation/NEW_TAG`。旧证据保留；固定输出脚本若已有结果，先考虑独立入口/输出参数，不覆盖重跑。

只有来源rollout完整manifest与SHA匹配后才做桥接，输出tag不得与来源重复：

```bash
bash scripts/course_bc_python.sh scripts/validate_bridge.py \
  --source-tag SOURCE_TAG --output-tag NEW_BRIDGE_TAG
```

部署缺项按用户任务范围处理；以上命令不扩展设备或运行预算，已有授权无需重复确认。`validate_training_readiness.py`参数与通过标准见[专项文档](../../docs/TRAINING_VALIDATION.md)，任意公开演示不能替代所需20帧样本。

## 4. 语义与验收

- 诊断主要是front/wrist双RGB；桥接为128×128、20帧10Hz。不能当LiftCube单front 64×64、30Hz策略合同。
- `so101leader`动作管理器有offset/scale；核对真实顺序、绝对弧度、电机归一化与raw管理器动作，不能直接发绝对q冒充偏置后动作。
- 受控轨迹/scene-state重放只证明对应记录、转换与连续性，不是专家成功或完整物理/RNG/RTX存档。桥接ACT未训练，不报正式成绩。
- 通过以实际断言、完整结果、所需manifest/轨迹一致为准，不以PID、零退出或单个JSON替代验收。保留partial/错误，没跑的写明。
- reset前捕获末帧；元数据不能证明成功、控制可达或适合训练。规则/IK诊断不改官方物理/执行器/评分。
- 文件名、固定步数和目录只是诊断条件，不推广为新任务预算或协议。

纯函数/合同修改选根目录对应单测，仿真路径变化才补授权下最小集成检查；复用其他证据。正式模型/任务使用明确训练与评测入口，不从诊断脚本拼隐式生产流程。改本目录文件同步 `manifests/course_files.json`实际条目；改命令/前提/输出同步本文件与根分流，不重写冻结上游清单。

新增诊断脚本应明确输入、输出、是否使用GPU/仿真和通过条件；可复用逻辑放函数，运行副作用放 `main` 保护内。新输出支持调用者指定位置，遇到已有结果应明确处理而非静默覆盖。报告只保留排查所需的非敏感字段，区分成功、失败与依赖不足未执行；沿用现有工具能完成的检查不要再造一份。
