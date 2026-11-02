# LiftCube训练基础

BC/PPO入口用于开发训练管道；ACT和VLA接入后沿用同一观测、动作与评测约定。正式训练前仍需验收数据动作映射及实际设备路径。

本指南及现有训练验收针对默认六关节`joint`路径。视觉规则基准使用[独立推理入口](../../docs/SCRIPTED_LIFTCUBE.md)，无需先训练或读取演示；它的抓取成绩不代表训练管道已经取得相同性能。`planar_ee_dev`是另一个三维动作契约，尚不是已验收的BC/PPO训练路径，不能只改`control_mode`后复用六关节标签或权重。

This guide and current training evidence cover the default six-joint path. The visual rule runner is independent of training and demonstration loading. Its scores do not establish learned-policy performance. The experimental three-component planar mode requires separate data/model/control validation before training.

## 输入与控制周期

SmallCNN融合front RGB、6维关节位置和6维上一执行命令；兼容布局中的速度、末端位置不进入网络。关节与上一命令按既有电机范围缩放，不用物体真值。旧image-only权重加载时保留原前向路径，不能静默当作融合模型初始化。

64×64输入经卷积得到8×8特征，4×4均值池化使用等价的普通2×2池化，避免配套PyTorch的adaptive pooling反向算子在严格CUDA确定性模式下报错；不增加权重或改变存档键。其他可整除网格同理，非整除尺寸保留自适应实现，须单独核对设备确定性支持。保存CUDA RNG不等于启用了确定性算子；严格恢复对照还需固定相关CUDA/cuDNN设置与环境版本。

PPO默认`ppo.policy_hz: 30`，在60Hz环境中保持每个目标两tick。物理dt、USD、执行器、成功条件及评测YAML不变。奖励在实际执行的各tick计算后求和，gamma/lambda按**决策步**定义；终止时立即结束保持，不多执行一次动作。`rollout_steps`是最少决策数，采集到该数量后继续到当前回合结束再更新，另记录真实物理tick数。

`ppo.bootstrap_timeouts: true`将外部超时从reset前终态bootstrap，GAE在所有回合边界断开；真正terminated不bootstrap，二者同时出现时terminated优先。若采用明确的有限时域终止目标，可在独立实验配置中设为false；这不改变环境的终止判据。不能把超时当作成功。

## 开发命令

在配套容器中，进入`course/liftcube`并设置`PYTHONPATH=$PWD/src:/leisaac/source/leisaac`。先检查已有NumPy/Torch/PyYAML及SwanLab 0.10.3，不升级配套Torch/CUDA；项目的`logging` extra声明SwanLab依赖。默认离线写入`logging.dir/swanlab/segment-*`，在线私有项目配置见[SwanLab指南](../../docs/SWANLAB.md)。BC横轴为累计优化更新，PPO横轴为累计决策数，同时记录物理tick；不再新建TensorBoard曲线，旧事件保留。

以下是训练入口示例，不是已验收数据集的发布或正式训练预算。BC要求可用LeRobot数据、正确单位/标定和episode级划分。`data.allow_action_clipping`仅供独立数据诊断，训练入口会拒绝启用它。

```bash
python -m starter.training.train_bc --config configs/bc_lift_cube.yaml \
  --data-path /data/lift_cube --checkpoint outputs/bc/checkpoint.pt

python -m starter.training.train_rl --config configs/rl_lift_cube.yaml \
  --bc-checkpoint outputs/bc/checkpoint.pt --checkpoint outputs/rl/checkpoint.pt
```

30Hz BC导出只有在相机、图像尺寸、本体布局、动作语义/顺序/范围、决策频率和模型结构均一致时才能初始化PPO。初始化只复制模型，PPO使用新优化器。

## 保存与恢复

- `checkpoint.pt`是推理导出，供评测和新任务初始化。
- `checkpoint.pt.training.pt`是训练恢复文件，附`.sha256`。包含模型、Adam、epoch/update、累计优化/交互计数、历史、全局Python/NumPy/Torch CPU及已初始化CUDA的全部随机流。
- BC只在完整epoch后保存，当前明确要求`num_workers=0`，无独立sampler generator；epoch内故障从上个完整epoch恢复，不宣称中途batch游标已保存。
- PPO只在完整回合收集及优化更新结束后保存。每个update从`seed + update`的独立reset开始；既有更新内回合使用环境自己的随机流。恢复不拼接任意中途仿真现场，不保证RTX或新进程物理位级等价。
- 恢复核对代码、有效配置、策略规格和BC已加载数组的身份；除总预算/输出位置等明确排除项外，不允许悄悄改配置。当前不序列化自定义私有RNG、异步worker或外部模型缓存；添加此类组件前必须扩展状态契约。
- 文件通过临时文件原子替换、SHA侧文件最后发布；文件或SHA不完整时拒绝恢复。只读取可信的本地训练文件，SHA用于完整性检查而非来源认证。

```bash
# epochs/num-updates都是总目标，不是再追加的数量。
python -m starter.training.train_bc --config configs/bc_lift_cube.yaml \
  --data-path /data/lift_cube --epochs 6 \
  --resume outputs/bc/checkpoint.pt.training.pt

python -m starter.training.train_rl --config configs/rl_lift_cube.yaml \
  --num-updates 4 --resume outputs/rl/checkpoint.pt.training.pt
```

恢复时保留此前日志；不要将`.training.pt`交给推理加载器。新增模型/环境属于新实验，使用导出权重初始化，不用`--resume`伪装同任务续训。新输出目录应与旧实验分开。

## 验证范围

`tests/test_liftcube_training.py`提供合成CPU检查：本体融合/补零屏蔽、旧权重前向、动作保持/终帧、GAE边界、完整BC/PPO入口和下一次更新恢复、身份及损坏文件拒绝。它不依赖Isaac或公开演示。

```bash
python -m unittest discover -s tests -p test_liftcube_training.py -v
```

该命令在仓库根目录执行。SwanLab另运行`tests/test_swanlab.py`的离线CPU检查。真实GPU恢复、Isaac边界接续、在线曲线显示及闭环成功率分别验收；软件测试通过不代表这些结果已经通过。
