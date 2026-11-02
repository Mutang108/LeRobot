# 按任务训练与评测 / Train and evaluate by task

先读[任务与适配状态](TASKS.md)，再从[资源清单](RESOURCES.md)获取环境资产、演示与权重。以下是使用者在完成前提和取得计算资源后运行的命令说明；仓库检查或下载不会代为启动训练。

Start with the [status matrix](TASKS.md) and [resources](RESOURCES.md). Commands below are execution instructions, not automatic launches or claims that task-specific training has already passed acceptance.

## 1. 选择正确入口 / Choose an entry

| 目标 / Goal | LiftCube | PickOrange |
| --- | --- | --- |
| 视觉脚本 / Visual rule | [已有规则基准](SCRIPTED_LIFTCUBE.md) | 暂未公开实现，可后续开发 / Not yet implemented |
| 特权教师 / Privileged teacher | 本页下一节，显式`--privileged-teacher` | [已收录官方状态机](../course/pickorange/README.md#特权教师采集--privileged-teacher-collection)，含物理/直接写状态限制 |
| 预训练VLA推理 / Pretrained inference | [单front包装](VLA_INTERFACES.md)，匹配本地适配描述；完整GPU闭环按模型验收 | [GR00T双视角原生辅助](../course/pickorange/README.md)，其他后端按状态表接入 |
| SmallCNN BC/PPO | [已有参考训练](../course/liftcube/TRAINING.md) | 当前不使用LiftCube训练器 / Not integrated |
| ACT/SmolVLA训练 | 数据验收后复用下述上游离线训练；任务闭环单独接入 | 同理，保持双视角与本任务动作/统计 / Separate task adapter required |
| π₀.₅/GR00T/OFT微调 | 先冻结形态、输入/动作与数据合同，再按匹配版本原生训练器实现 / Backend-specific training integration pending | 同左；不得混用不同模型图、形态头和处理器 / Same boundary |

推理包装只负责加载与执行，不能直接接入SmallCNN优化器。现有VLA CPU前向、上游训练脚本和本项目任务训练验收分别记录；没有一键完成四种模型训练的统一入口。

Inference wrappers are not trainers. Use each model's own optimizer/loss and checkpoint format; no unified four-model task trainer is supplied.

新训练/评测曲线统一使用[SwanLab](SWANLAB.md)，默认离线，在线须指定私有项目。只记录曲线与白名单配置，原始结果和模型保留本地。New training/evaluation curves use scalar-only SwanLab tracking; configure the private destination before online runs.

## 2. LiftCube特权教师 / LiftCube privileged teacher

教师在`starter.privileged_liftcube.PrivilegedLiftCube`，读取cube相对机器人base的真实位置，复用现有开爪→定位→下降→闭合→抬升控制及限位/限速/跟踪保护；不改物理或成功条件。它是**尚未真实仿真验收的开发候选**，不能作为视觉baseline。

The teacher explicitly reads object truth outside `Observation`, retaining controller safety and native scoring. It is a development candidate, not a validated expert or a visual baseline.

在LiftCube配套环境内，工作目录`course/liftcube`，设置`PYTHONPATH=$PWD/src:/leisaac/source/leisaac`。使用与实际机器人/TCP对应的独立标定；沿[标定指南](SCRIPTED_LIFTCUBE.md)准备至少三个FK检查。教师使用固定抓取/抬升姿态和高度：标定不得启用`grasp_presets`或`plan_while_opening`。相机矩阵沿用标定格式，但定位不使用RGB或相机外参；物体尺寸/高度变化需重新检查路径。

```bash
python -m starter.scripts.evaluate_scripted --privileged-teacher \
  --calibration /calibration/liftcube-teacher.json --check

python -m starter.scripts.evaluate_scripted --privileged-teacher \
  --calibration /calibration/liftcube-teacher.json \
  --config configs/eval_lift_cube.yaml \
  --output /outputs/liftcube/teacher-new --run-simulation
```

第一条只作CPU标定/IK检查；第二条真实运行仿真。记录中显式标注特权输入，保留原始请求、发送/反馈、终态及RGB；输出目录必须新建。它仍是评测轨迹格式，不是可直接喂给LeRobot的专家数据。先验证实际抓取和冷重放，再转换、筛选及重新分训练/验证集。

The runner records evaluation trajectories, not an automatically accepted LeRobot dataset. Validate task success and replay before conversion and training.

## 3. 数据准入 / Data admission

两任务独立维护源数据revision、原始文件、转换清单、episode划分和训练统计：

1. 核对六关节顺序、motor/radian、绝对/增量、夹爪范围、真实相机、帧率及观测→请求动作对齐。上一发送目标、实测q和动作标签分别保留。
2. 明确教师是否使用物体真值、动态夹爪辅助、重力补偿或直接写状态。辅助协议与固定物理分组；禁止静默裁剪原标签、改USD限位或用下一帧q替换标签。
3. 验证完整终态、原生成功和reset前RGB；少量独立冷重放检查连续性和抓取。模型生成的成功轨迹也须审核，不自动变成BC数据。
4. 按episode/场景/物体分组划分，不能随机拆相邻帧。仅训练划分拟合统计，测试集不用于统计或挑选成功片段。
5. 将通过的公共图像、q、请求动作及任务文本转换为所选后端支持的LeRobot版本；物体真值等另存诊断，不放入普通策略输入。

Keep source data immutable, splits episode-based and normalization training-only. Record assistance explicitly. The existing 20-frame PickOrange bridge is a diagnostic fixture, not a whole-dataset converter.

LiftCube本地数据检查入口见[数据指南](../course/liftcube/TRAINING.md)和`starter.scripts.inspect_dataset`。PickOrange模型轨迹先用`course/pickorange/audit.py`；它不验证相机新鲜度或完成独立冷重放，也不直接批准BC。

## 4. ACT与SmolVLA离线训练 / Offline training

在**独立LeRobot模型环境**安装本仓库冻结的`third_party/lerobot`及所选模型extra；不要在仓库根或模拟器依赖上覆盖安装。上游具体选项见[ACT](../third_party/lerobot/docs/source/act.mdx)、[SmolVLA](../third_party/lerobot/docs/source/smolvla.mdx)与[训练脚本](../third_party/lerobot/src/lerobot/scripts/lerobot_train.py)。

以下为经过数据准入之后的小预算开发示例，不是本项目已验证的训练配方。`team/liftcube-accepted`是需替换的逻辑ID；`--dataset.root`必须指向已转换且支持的本地数据目录。正式预算另行配置。

These are small development examples after data acceptance, not validated task recipes. Substitute the logical dataset ID, accepted local data and fresh output directory.

```bash
python -m starter.scripts.train_lerobot --policy.type=act \
  --dataset.repo_id=team/liftcube-accepted --dataset.root=/datasets/liftcube/accepted \
  --output_dir=/outputs/liftcube/act-dev --job_name=liftcube-act-dev \
  --policy.device=cuda --batch_size=8 --steps=1000 \
  --policy.push_to_hub=false --wandb.enable=false

python -m starter.scripts.train_lerobot --policy.path=/models/liftcube/smolvla-a124aa1 \
  --dataset.repo_id=team/liftcube-accepted --dataset.root=/datasets/liftcube/accepted \
  --output_dir=/outputs/liftcube/smolvla-dev --job_name=liftcube-smolvla-dev \
  --policy.device=cuda --batch_size=8 --steps=1000 \
  --policy.push_to_hub=false --wandb.enable=false
```

以上在仓库根目录设置`PYTHONPATH=$PWD/course/liftcube/src`后运行，使用独立模型环境。SwanLab包装复用原生训练器，将其W&B logger路由至只记录数值的共享层；示例中的`--wandb.enable=false`不关闭SwanLab。记录模式与项目按[SwanLab指南](SWANLAB.md)设置，不把本地项目/凭据加入共享配置。

Run from the repository root with the LiftCube `src` on `PYTHONPATH`, in the separate model environment. The wrapper replaces the native logger hook only; `--wandb.enable=false` does not disable SwanLab. This bridge has CPU fixture checks, not full native model-training acceptance.

使用旧模型配置前先核对当前后端支持字段，处理不生效的legacy字段须留下明确兼容记录；严格权重和统计不能放宽。SmolVLA源权重为双视角，若训练采用单front，须按[包装契约](VLA_INTERFACES.md)明确缺槽mask/预处理与导出，不能假定改数据路径就完成适配。

PickOrange沿相同离线入口，但更换成其**独立已准入**数据、匹配视角/统计/任务权重和输出目录。不能只换任务名称沿用LiftCube统计，也不能把上述GR00T服务当ACT/SmolVLA推理器。

先验证一批样本、损失与参数实际更新、保存重载、完整优化器/RNG恢复，再扩大训练。日志区分训练loss、训练内成功率与独立策略评测；评测必须用完整冻结场景和实际执行结果。

## 5. 评测与发布 / Evaluation and release

- LiftCube用`starter.evaluation.evaluate`，VLA通过`--policy-class`与本地适配JSON加载，规则/教师走专属入口。相同输入、动作、频率、预算和评分才合算。
- PickOrange用[独立原生辅助入口](../course/pickorange/README.md)。新后端须先完成真实双相机/动作转换和回合缓存验证；未经接入不能把训练loss当闭环能力。
- 当前标准VLA包装均未提供任意训练任务resume承诺。不同任务初始化、LoRA微调、完整续训及纯推理导出分别处理。
- GitHub发布代码、公开配置、来源及结果摘要；模型/LoRA/动作头和统计可发布到Hugging Face模型仓库，演示发布到数据集仓库，均先确认许可、隐私及完整性。下载URL和固定revision回填[资源清单](RESOURCES.md)，运行产物不进入Git。

Publish reproducible code/configuration in Git, model exports on an appropriate model hub and demonstrations in a dataset repository. Keep artifact URLs and revisions synchronized with the status matrix; report only independently audited complete episodes under one protocol.
