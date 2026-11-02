# SwanLab曲线记录 / Experiment tracking

新训练和策略评测使用SwanLab。现有JSON、轨迹、模型存档和历史TensorBoard事件保留，不因接入日志平台重跑实验。本页说明记录方式，不授予训练或仿真运行权限。

New training and policy evaluations use SwanLab. Local results, trajectories, checkpoints and historical TensorBoard events remain intact. Tracking does not replace data admission or independent evaluation audits.

## 安装与登录 / Setup

在对应训练或评测环境中检查`SwanLab 0.10.3`。LiftCube可安装`course/liftcube`的`logging` extra；独立模型环境可单独安装SDK。先检查依赖兼容，不覆盖模拟器自带Torch/CUDA。

```bash
python -m pip install swanlab==0.10.3
swanlab login
```

在交互提示输入密钥，不使用带密钥的命令参数，不写入源码、YAML或Git。在线记录前，在网页创建或确认**私有项目**，并核对工作空间。代码传入`public=False`，但不能把已存在的公开项目自动变为私有。登录和项目选择由实际使用者完成；公共仓库不提供维护者账号配置。

Use interactive login, never inline API keys. Verify the destination workspace and **private** project in the web UI first. `public=False` requests private creation; it does not change an existing project's visibility.

## 配置 / Configuration

默认`offline`，写本地SwanLab记录，不登录或上传。显式选择`online`才同步云端，且必须提供项目名。以下变量适用于两任务及LeRobot训练包装；进程环境优先于LiftCube的`logging.swanlab`配置。

```bash
export LEROBOT_SWANLAB_MODE=online
export LEROBOT_SWANLAB_PROJECT=your-private-project
export LEROBOT_SWANLAB_WORKSPACE=your-workspace
export LEROBOT_SWANLAB_NAME='训练-ACT-LiftCube-seed17'
```

`WORKSPACE`省略时使用当前登录者的默认工作空间。显示名首个分隔符用`-`；历史名称不会自动修改。Windows终端若使用GBK，先设置`PYTHONIOENCODING=utf-8`，避免SDK终端提示的Unicode编码错误。

仅LiftCube配置也可以这样写；账号、项目私有绑定留在本地未跟踪的配置中：

```yaml
logging:
  dir: /outputs/liftcube/new-run
  swanlab:
    mode: offline  # online需额外指定project；disabled仅用于明确的无记录检查
    name: 评估-SmolVLA-LiftCube-seed17
```

Modes: `offline` stores locally, `online` explicitly uploads, and `disabled` is for checks that intentionally produce no experiment record. Never use disabled tracking for a formal run. Online mode requires an explicit project; project/workspace/name are runtime inputs, not repository defaults.

## 已接入的入口 / Connected entries

| 入口 / Entry | 记录 / Curves and step |
| --- | --- |
| `starter.training.train_bc` | `bc/*`，累计优化更新 / Optimizer updates |
| `starter.training.train_rl` | `ppo/*`，累计决策步；另记物理tick / Decision steps, separate physics counters |
| `starter.scripts.train_lerobot` | ACT/SmolVLA上游训练的`train/*`与`eval/*`，累计优化更新 / Native optimizer steps |
| `starter.evaluation.evaluate` | `eval/*`，完整回合数；长度、return、sim秒、终止及完成前缀成功率 / Completed episodes |
| `starter.scripts.evaluate_scripted` | 同上；标注视觉规则或特权教师 / Rule/teacher identity kept separate |
| `course/pickorange/evaluate.py` | `eval/*`完整回合，`inference/*`查询序号；原生辅助标记 / Episodes and query counters |

LiftCube评测入口同样覆盖已接入的VLA包装，不需要每个模型再建一个日志run。PickOrange仅evaluator写SwanLab，模型service不重复写。曲线中的完成前缀仍需本地逐回合审核，不能视为已发布的最终成绩。`--check`、数据检查、模型下载和单元测试不是策略实验，不隐式创建云端run。

ACT/SmolVLA使用[训练指南](TRAIN_AND_EVALUATE.md)中的`python -m starter.scripts.train_lerobot`，而不是直接`lerobot-train`。包装仅在当前进程替换冻结LeRobot v0.4.2的logger挂钩，保留原训练循环、主进程选择、优化器和恢复逻辑；内部W&B开关只是路由条件，不会调用W&B SDK。模型、视频上传请求被忽略，原文件仍保留本地。其他版本须重新核对挂钩。

Use the wrapper for the pinned ACT/SmolVLA offline trainer. It does not implement or validate new PI05/GR00T/OFT fine-tuning pipelines; those future trainers should call the same scalar writer after their own training integration is accepted. Upstream teacher generators and historical diagnostic tools are not silently rewritten.

## 隐私、异常与恢复 / Privacy and recovery

- 仅发送有限数值曲线与白名单配置：seed、算法/任务、输入尺寸、频率、chunk及训练预算等。不传完整配置、数据/模型路径、图像、视频、模型参数、终端输出或主机探针。SDK硬件、运行时、依赖、Git、系统监控和webhook采集全部关闭。
- SDK异常/中断钩子会默认采集调用栈，包装恢复原处理器；结束状态只附通用说明，不上传原异常文本。不要在同一进程额外注册自动采集或第三方上传回调。
- 每次进程启动创建新的`logging.dir/swanlab/segment-0001`等目录；已有目录不覆盖。恢复创建`-segment2`等新显示分段，保留原全局优化/交互计数，不复用或冒充恢复旧云端run。ACT/SmolVLA的首个真实计数来自原生恢复后的训练回调，初始化收据中`start_step=null`表示尚未观测，不表示0步重训。
- `tracking.json`记录模式、run ID、分段起点和最后计数；`scalars.jsonl`先落盘数值，再交SDK。SDK记录调用抛错时保留本地journal并报警，不因云端失败停止健康worker。网络不可靠时选offline，避免在线认证或结束同步等待影响任务。
- 日志层保护已加载的Python/NumPy/Torch CPU及已初始化CUDA RNG，不初始化CUDA；它不是训练恢复点。模型/优化器/RNG等完整存档仍由训练器负责，严格代码身份检查不放宽，旧版本存档不能因日志升级就跳过兼容检查。

Only scalar metrics and an allowlisted configuration are sent. Every new process owns a new segment/run, with original global counters and no overwrite of historical steps. SDK log exceptions degrade tracking, not the worker. Prefer offline mode for unreliable networks. Model recovery remains the trainer's responsibility; a tracking run ID is not a checkpoint.

离线SDK原始记录在各分段的`sdk/run-*`目录。需要补传时，先复核其内容及目标项目隐私，再用SDK命令指定目标；不要对任意历史日志目录整目录盲传，也不要手工改run ID重复上传：

```bash
swanlab sync /outputs/liftcube/new-run/swanlab/segment-0001/sdk/run-REPLACE \
  --project your-private-project --workspace your-workspace
```

## 软件验收 / CPU checks

仓库根目录、独立CPU环境：

```bash
python -B -m unittest discover -s tests -p test_swanlab.py -v
python -B -m unittest discover -s tests -p test_liftcube_training.py -v
```

覆盖数值白名单、隐私开关、RNG不变、分段/计数、故障journal、上游logger挂钩及合成LiftCube评测。真实SDK测试只作离线CPU写入；没有启动云端实验、模型训练或模拟器。完整在线显示、实际模型训练及PickOrange闭环仍须在获准的运行环境验收。

Tests exercise synthetic CPU fixtures and a real **offline** SDK run, not cloud or robot performance. CI installs the pinned SDK so the roundtrip cannot silently skip. API details: [initialization](https://docs.swanlab.cn/api/py-init.html), [settings](https://docs.swanlab.cn/api/py-settings.html).
