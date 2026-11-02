# VLA推理接口 / VLA inference interfaces

本页的四个包装位于LiftCube包，读取单front/q6并由LiftCube执行层运行；不是PickOrange通用加载器。PickOrange原生双RGB、动作offset和任务终态需按[任务指南](PICK_ORANGE.md)另接，跨任务支持见[对照表](TASKS.md)。共同机器人形态不替代环境适配。GR00T包装仅对应N1.5，N1.6/N1.7检查点需匹配版本的独立后端。

These wrappers target the LiftCube single-front/q6 contract. PickOrange requires its own observation/action/terminal adapter; they are not universal task loaders. The GR00T loader is N1.5, not N1.6/N1.7.

权重、演示与完整revision见[资源清单](RESOURCES.md)，成员训练/评测步骤见[指南](TRAIN_AND_EVALUATE.md)。PickOrange GR00T双视角服务与原生辅助评测在[独立目录](../course/pickorange/README.md)，不使用这里的单front类。 / See the resource and execution guides; the dual-view PickOrange runner is separate.

当前提供SmolVLA、SO101版π₀.₅、GR00T N1.5和OpenVLA-OFT单front推理包装及共享六关节chunk执行层。边界测试使用CPU替身；SmolVLA/π₀.₅另通过真实权重CPU加载及合成输入前向，OpenVLA原基座通过CPU加载和处理器检查。GR00T完整构造受下述CUDA依赖限制；任务统计/受训形态头、GPU及LiftCube闭环仍需验收。本页不提供训练入口，原SmallCNN/BC/PPO默认不变。

Four single-front wrappers share joint-chunk execution. Boundary tests use CPU fixtures; separate real-weight checks cover SmolVLA/PI05 CPU loading and synthetic forwards, plus original OpenVLA core loading and preprocessing. GR00T construction has the CUDA limitation below. Task statistics/trained embodiment heads, GPU inference and LiftCube rollouts still require acceptance. No VLA trainer is supplied; SmallCNN/BC/PPO defaults are unchanged.

## 公共执行层 / Shared execution

`starter.models.joint_chunk.JointChunkPolicy`接收一个`predict(observation, task)`可调用对象，返回已反归一化的`[time, 6]`目标。输出只允许SO101绝对motor，或显式`arm_degrees_gripper_motor`（前五维角度、夹爪仍为motor）；末端动作不适用。逐模型反归一化由模型处理器完成，公共层不猜统计、不裁剪、不截维度。

The predictor returns unnormalized `[time, 6]` targets: absolute SO101 motor values or explicitly `arm_degrees_gripper_motor`. EEF commands are not supported. Model processors unnormalize outputs; the shared layer never guesses statistics, clips or drops dimensions.

- `execute_steps`指定每次预测实际使用的前缀，耗尽后用最新观测重规划；短于此前缀、空块、非有限或越界块会拒绝并清空。`last_raw_chunk`与`last_motor_chunk`保留当前块供诊断，实际发送/反馈仍由环境记录。
- `act()`一次返回一个目标，30Hz策略在60Hz物理环境的两tick保持仍由公共评测层执行，不重复保持。
- 公共`run_episode`在reset前及成功、失败、超时、异常退出时调用可选`policy.reset()`；旧无reset策略继续兼容。该钩子只能清缓存，不能重seed。任务文本变化通过`set_task()`清空旧块；外部接管后须显式reset。
- `front_inputs()`只读取实际RGB和许可q，忽略metadata、速度/EE补零及上一命令。图像从当前观测转为CHW float[0,1]；不增加腕部相机或读取物体真值。模型内部resize不能恢复被环境缩小的信息。

`execute_steps` limits the consumed prefix; replanning uses the next observation. Invalid chunks clear caches and are rejected. `act()` returns one target; only the runner applies the physical hold. The runner calls optional `policy.reset()` at both boundaries and on errors, without changing RNG. Use `set_task()` for language changes and reset after interventions. `front_inputs()` forwards only real RGB, measured q and caller-supplied task—not metadata, padding or object truth.

四个加载器在导入大型backend前检查任务文本、正整数执行长度、有限正频率及单front观测规格，观测/动作时钟必须一致。SmolVLA和PI05的输出horizon须与已加载配置相等。BF16输出显式转FP32后交NumPy，此转换不裁剪目标。

All four loaders validate task text, executed prefix, frequency and single-front specifications before importing large backends. Observation/action clocks must match. SmolVLA/PI05 output horizons must match their loaded configurations. BF16 tensors convert to FP32 before NumPy, without clipping targets.

## SmolVLA单front / Single-front SmolVLA

入口：`starter.models.smolvla:SmolVLAFrontPolicy`，对应仓库冻结LeRobot v0.4.2 API。模拟器保持配套版本，模型后端独立安装；不要把`third_party/`所有目录加入模拟器路径。加载器只读本地目录，不自动下载；tokenizer/VLM配置也须事先可用。

Entry: `starter.models.smolvla:SmolVLAFrontPolicy`, using the pinned LeRobot v0.4.2 API. Keep model dependencies separate from the simulator; the loader uses local files only, including tokenizer/VLM assets.

front必须映射到配置的第一个训练相机槽。包装复制配置，只保留该真实图像特征，按原相机槽数请求`empty_cameras`；SmolVLA正式`validate_features/prepare_images`生成带false mask的空槽。原权重和配置文件不修改，不复制front补其他视角。仅支持单时刻观测和绝对SO101关节动作；拒绝ALOHA转换、delta/RTC配置。

The first trained camera slot receives front RGB. A copied configuration retains that real feature and requests backend-generated masked empty slots. Original files are unchanged. No duplicate camera images are fabricated. Single-step observations and absolute SO101 joints only; ALOHA, delta and RTC modes are rejected.

### 本地适配描述文件 / Local adapter descriptor

公共评测的`--checkpoint`在此指可信本地JSON，不是SmallCNN `.pt`。相对路径以该JSON目录为基准。描述文件由实际checkpoint/训练数据生成；仓库不提供可直接执行的假统计。

For this loader, `--checkpoint` names a trusted local JSON rather than a SmallCNN `.pt`. Paths resolve relative to it. Populate values from the actual checkpoint/training data; no runnable identity-statistics template is supplied.

| 字段 / Field | 必填语义 / Required meaning |
| --- | --- |
| `model_path`, `model_revision`, `vlm_path` | 本地模型、固定权重revision、本地匹配VLM/tokenizer / Local model, pinned revision, matching VLM/tokenizer |
| `joint_names`, `joint_units` | 与`interfaces/action.py::JOINT_NAMES`完全同序；`motor` / Exact canonical order; `motor` |
| `calibration_source` | 动作与当前SO101 motor映射的来源记录；字符串不代替实际标定验收 / Provenance, not proof of calibration |
| `image_key` | 模型第一个训练视角，例如`observation.images.camera1` / First trained camera slot |
| `task` | 当前任务文本，不能沿用PickOrange指令 / Current task text |
| `observation_spec` | `ObservationSpec.to_dict()`，匹配实际front尺寸、18/21维本体与30Hz时钟 / Actual source image/proprio specification |
| `frequency_hz`, `execute_steps` | 实际策略频率与每块执行长度 / Decision rate and executed prefix |
| `normalization.source` | checkpoint或合格训练集统计来源，不使用测试集拟合 / Checkpoint or accepted training-statistics identity |
| `normalization.stats` | `observation.state`与`action`各自六维`mean/std`，std有限且为正 / Six-channel mean/std for both keys |

加载器要求STATE/ACTION采用MEAN_STD、VISUAL采用IDENTITY，使用严格权重加载、显式统计及一次输出反归一化。没有有效统计时拒绝，不静默切换单位统计或PickOrange统计。基座特征可能已满足六维形状，但这不证明动作标定、单相机技能或任务能力。旧checkpoint若含当前类不匹配的legacy normalizer键，严格加载失败须保留证据，不改为strict=False。

STATE/ACTION must use MEAN_STD and VISUAL IDENTITY. Strict loading and explicit statistics are required; no silent identity or cross-task fallback. Six-dimensional features do not prove calibration or single-camera task competence. Preserve strict-loading failures for legacy checkpoints rather than suppressing missing/unexpected keys.

当前固定SmolVLA基座的处理器存档未提供此包装要求的显式六维state统计。原生模型的严格加载与归一化张量前向可单独检查，但真实motor/q包装仍须合格训练统计；不要把action统计直接当state统计，也不补造单位统计。

上述缺口针对通用基座，不适用于已找到的LiftCube任务权重`azazdeaz/smolvla_so101_leisaac_lift_cube@a124aa1`：该任务导出含自身state/action统计，已与源数据核对并完成单front缺槽mask的CPU完整前向。使用时从该任务导出读取真实统计填入描述；当前环境标定、缺失wrist的技能影响和GPU闭环仍待验收。 / The task checkpoint supplies its own verified statistics; generic-base limitations must not be attributed to it.

The pinned SmolVLA base processor files lack the explicit six-channel state statistics required here. Native loading/normalized-tensor forwards can be checked separately; physical motor/q adaptation still needs accepted training statistics. Do not substitute action statistics or fabricated identity values.

资源与运行获准后，在兼容的已配置环境使用公共评测入口（本命令会启动仿真，不属于CPU检查）：

After resource and execution approval, use the common runner in a compatible configured runtime. This starts simulation, not a CPU check:

```bash
python -m starter.evaluation.evaluate --config configs/eval_lift_cube.yaml \
  --policy-class starter.models.smolvla:SmolVLAFrontPolicy \
  --checkpoint /path/to/adapter.json
```

配套模拟器若不能导入模型后端，此同进程命令不可用；应沿用独立推理服务架构，后续接经过验证的传输层，不在模拟器中强行升级依赖。本次没有新增服务/网络协议。CPU测试使用注入backend验证实际包装的前/后处理与队列，不声称完整模型加载通过。

If simulator and model dependencies cannot coexist, use a separately validated inference service instead of upgrading the simulator. No new transport is provided here. Injected CPU backends validate wrapper preprocessing/postprocessing and queues, not full checkpoint loading.

## 其他三个后端 / Other native backends

| 路线 / Backend | 可复用 / Reusable | 仍须解决 / Remaining |
| --- | --- | --- |
| π₀.₅ | `models.pi05:Pi05FrontPolicy`；原生批次、内嵌统计、首视角及两空槽、严格权重名/存储alias恢复；真实CPU加载/合成前向通过 / Native batches, embedded stats, masked slots, strict alias loading and CPU synthetic forward checked | GPU、真实观测与闭环待验；不是HF原版PI05模型图 / GPU and real-observation rollouts pending, not stock HF PI05 |
| GR00T | `models.gr00t:Gr00tFrontPolicy`；官方单图模态、5＋1输出、原生变换统计及受检加载 / Single-camera modality, 5+1 outputs and native transforms | 需要匹配单图配置及SO101形态元数据的checkpoint；通用基座不自带已训练SO101头 / Requires matching trained embodiment metadata and one-view recipe |
| OpenVLA-OFT | `models.openvla_oft:OpenVLAOFTFrontPolicy`；单图原生函数、六维BOUNDS、本体投影和连续头 / Native single-image OFT, six-channel BOUNDS, projector and continuous head | 单图受训头/投影及六维后端常量须匹配；不接受7维末端原版 / Requires trained one-view components and SO101 backend constants, not vanilla EEF output |

三个包装均接受注入原生backend，CPU检查使用替身；真实`load()`沿各自原生库调用，不能把替身通过称为完整依赖安装/权重加载通过。它们共用队列、任务切换、观测白名单、动作拒绝及公共评测，不共用错误的模型归一化。没有随机头执行、隐藏的维度截取或新训练器。

Each wrapper accepts an injected native backend for CPU checks and supplies a native local loader. Fixture success is not full dependency/checkpoint integration. Queues, task switches, input whitelists, action rejection and evaluation are shared—not model normalization. No random-head execution, hidden dimension trimming or trainer is added.

冻结GR00T/Eagle代码的`radio_model.py`在导入时调用`torch.cuda.get_device_capability()`；无GPU进程可读取单图模态与校验权重文件，但完整原生构造会失败。不要伪造CUDA能力来声称CPU加载通过。OpenVLA的CPU严格加载仅验证原始通用核心与单图处理器，不补齐受训SO101 L1头/投影，也不把原版7维动作当六关节输出。

The pinned GR00T/Eagle `radio_model.py` queries CUDA capability at import. CPU-only processes can inspect modalities and verify files but cannot complete native construction; do not fake CUDA capability. OpenVLA CPU loading covers the original generic core and processor, not missing trained SO101 components or six-joint decoding.

### 本地描述文件与后端版本 / Local descriptors and backend versions

三个新加载器的公共字段为`backend`、`joint_names`、`joint_units`、`model_path`、`model_revision`、`backend_revision`、`calibration_source`、`task`、`observation_spec`、`execute_steps`及`frequency_hz`。模型目录必须已存在；版本和来源字段为明确声明，部署前仍须核对实际源码/文件指纹，不能仅凭描述字符串认定身份。每个后端在隔离环境中安装，不自动安装或下载权重。

Common fields: `backend`, `joint_names`, `joint_units`, `model_path`, `model_revision`, `backend_revision`, `calibration_source`, `task`, `observation_spec`, `execute_steps`, `frequency_hz`. Model paths must exist. Provenance strings are declarations, not fingerprint verification; check actual files before deployment. Install optional backends separately; loaders do not install packages or fetch weights.

| 包装 / Wrapper | 描述值、额外字段与约束 / Descriptor and requirements |
| --- | --- |
| `Pi05FrontPolicy` | `backend: pi05-vlash`、`joint_units: motor`；使用SO101发布包对应VLASH `22cbabfee0f57874987c75a35a7dac129e695db0`及作者补丁，不将其state_cond模型接入HF原版。须在进程导入backend前设置`HF_HUB_OFFLINE=1`并缓存获准tokenizer；权重中的统计原值加载，无外部postprocessor再次反归一化 / Author-matched VLASH and offline tokenizer; embedded statistics, no second unnormalization |
| `Gr00tFrontPolicy` | `backend: gr00t-n1.5`，`embodiment_tag`、`data_config`；代码固定`4af2b622892f7dcb5aae5a3fb70bcb02dc217b96`，官方`so100`的单图键为`video.webcam`，可将当前front绑定该真实槽。`so100_dualcam`会拒绝；metadata必须含对应形态。动作horizon不符时拒绝，不重建新头。backend前设置`HF_HUB_OFFLINE=1`并缓存骨干资产 / Match one-view config/embodiment and horizon; offline backbone required |
| `OpenVLAOFTFrontPolicy` | `backend: openvla-oft-so101`，`trained_camera_keys: [front]`、`action_head_path`、`proprio_projector_path`、`unnorm_key`、`chunk_size`、`center_crop`；对应YoungK82 `ac34367e52d52f967fa00b7bfef1ff3d2c9e253e`。导入OFT模型模块前必须已选择其SO101常量（6/6及匹配chunk、BOUNDS）；默认LIBERO常量会拒绝。仅完整本地连续L1导出，LoRA须先按原后端合并导出；不把LoRA目录当完整模型 / Select SO101 constants before model imports; complete local L1 export, not bare LoRA |

GR00T的CPU受检加载入口绕开上游假设“只返回model”的`from_pretrained`包装，以HF基类获取完整loading report；禁止missing/unexpected/mismatched/error。OFT同样检查模型loading report，并严格加载已存在的head/projector（safetensors或可信weights-only state_dict），不调用会改写本地config/模型代码的辅助加载器。OFT原生函数有模块级DEVICE，必须与请求设备一致；不要在同一进程中混用不同设备。CPU fixtures不导入这些大型后端，也没有开启GPU。

GR00T uses the HF base loader for a complete loading report, bypassing the native model-only return assumption. OFT checks model loading info and strictly loads existing head/projector files without the helper that edits checkpoint files. Its module-level DEVICE must match the requested device. CPU fixtures import none of these heavy backends and use no GPU.

### GR00T输入分辨率 / GR00T input resolution

默认`image_preprocessing: checkpoint`要求观测尺寸与checkpoint视频元数据一致。若课程仅提供front64，在本地描述中显式设置`"image_preprocessing": "native_resize"`：加载器只在内存副本中将视频输入分辨率声明为实际观测的宽高，再交原生裁剪、缩放和RGB处理。课程环境、原始图像、checkpoint文件、相机标定、状态/动作统计与权重均不改动；原生输入校验仍开启。输入帧率须与决策频率一致。

The default `image_preprocessing: checkpoint` requires the observation size to match checkpoint video metadata. Set `"image_preprocessing": "native_resize"` explicitly to accept front64: only a runtime metadata copy declares the actual input width/height before native cropping, resizing and RGB processing. Environment observations, checkpoint files, calibration, normalization statistics and weights stay unchanged; native validation remains enabled. Video and decision frequencies must match.

这解决格式适配，不恢复低分辨率细节，也不保证视角/标定或任务分布匹配。不要先将64×64拉伸到1280×720再宣称获得高清输入。仅预处理和形态投影CPU通过时，完整VLM/flow推理与闭环仍须另验。

This resolves input formatting, not lost image detail, camera/domain alignment or task performance. Upscaling 64×64 to 1280×720 does not create high-resolution observations. Preprocessing/projection CPU checks do not replace complete VLM/flow inference and rollout acceptance.

OFT的`load_local_processor()`显式使用固定后端的`PrismaticImageProcessor`与标准tokenizer组装处理器，不经过checkpoint的`AutoImageProcessor`代码分派；tokenizer设置`trust_remote_code=False`。无需交互批准或执行checkpoint附带的Python来加载图像处理配置。

OFT constructs its processor from the pinned `PrismaticImageProcessor` and a standard tokenizer with `trust_remote_code=False`. It avoids checkpoint-driven AutoImageProcessor dispatch and its interactive custom-code prompt.

GR00T/OFT遇到不完整加载时抛出`ValueError`，完整原生报告保存在异常的`loading_info`属性。诊断记录留在本地，公开前检查其中的模型路径等信息。

Incomplete GR00T/OFT loads raise `ValueError` with the native report attached as `loading_info`. Keep diagnostic records local and review model paths before publishing them.

OFT后端通过首次导入时的真实进程参数选择常量。公共评测CLI接受`--robot_platform SO101`，应在上述获准仿真命令中同时指定`--policy-class starter.models.openvla_oft:OpenVLAOFTFrontPolicy`和相应描述文件；不要用默认LIBERO常量。程序化调用须在任何OFT模型导入前保证同样的后端初始化，加载器不会改写`sys.argv`或全局常量。此入口只解决参数传递，完整模型与仿真兼容性仍需验收。

The pinned OFT backend selects constants from the real process CLI on first import. For approved evaluation, pass `--robot_platform SO101` together with `--policy-class starter.models.openvla_oft:OpenVLAOFTFrontPolicy` and its descriptor. Programmatic callers must initialize the backend before importing dependent models; loaders never rewrite process arguments or constants. This supplies the CLI flag, not full runtime acceptance.

上游契约来源：[GR00T单图配置](https://github.com/NVIDIA/Isaac-GR00T/blob/4af2b622892f7dcb5aae5a3fb70bcb02dc217b96/gr00t/experiment/data_config.py)、[GR00T模型加载](https://github.com/NVIDIA/Isaac-GR00T/blob/4af2b622892f7dcb5aae5a3fb70bcb02dc217b96/gr00t/model/gr00t_n1.py)、[SO101 PI05](https://github.com/jinnymo/so101-pi05-base)、[SO101 OFT实现](https://github.com/YoungK82/OpenVLA-oft/tree/ac34367e52d52f967fa00b7bfef1ff3d2c9e253e)。

## CPU检查 / CPU checks

在仓库根目录使用独立Python 3.11/3.12环境；无需安装模拟器或模型库。以下为Linux示例，Windows将`.local/vla-cpu/bin/python`替换为`.local/vla-cpu/Scripts/python.exe`：

Use an isolated Python 3.11/3.12 environment at the repository root. Linux example below; on Windows use `.local/vla-cpu/Scripts/python.exe` instead of `.local/vla-cpu/bin/python`:

```bash
python -m venv .local/vla-cpu
.local/vla-cpu/bin/python -m pip install numpy==1.26.4 PyYAML==6.0.3
.local/vla-cpu/bin/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
.local/vla-cpu/bin/python -B -m unittest discover -s tests -p 'test_vla*.py' -v
```

依赖已齐全时也可分组运行 / With dependencies installed, run each group separately:

```bash
python -m unittest discover -s tests -p test_vla_interfaces.py -v
python -m unittest discover -s tests -p test_vla_native_backends.py -v
```

需要NumPy、CPU Torch和PyYAML，不需要LeRobot安装、权重、GPU或模拟器。测试执行冻结上游的单个图像mask方法，不导入VLM。GPU与真实模型验收另行进行；软件通过不代表抓取成功。

Requires NumPy, CPU Torch and PyYAML, not a model, GPU, simulator or installed LeRobot. One pinned upstream masking method is exercised without its VLM. Real-model/GPU and grasp acceptance remain separate.

PR工作流的`vla-cpu`会安装上述CPU依赖并运行两组测试；`source-check`继续做来源与文档检查。缺少依赖时unittest会skip，必须与通过区分。feature分支普通推送不触发此工作流，PR建立后自动运行，也可由维护者手动触发。

The PR workflow's `vla-cpu` job installs these CPU dependencies and runs both groups. Missing dependencies cause skips, not acceptance. Feature pushes alone do not trigger this workflow; PRs and maintainer-dispatched runs do.

## 后续完整模型验收 / Full-model acceptance

获准运行后，逐后端记录实际代码/权重/统计身份与严格加载结果，用固定RGB/q检查前后处理、输出与动作映射，再检查新回合及任务切换后无旧chunk。最后沿同一环境/评分协议做少步闭环，并记录模型raw、实际发送和反馈、资源峰值与终态。CPU替身证据可复用，但不能填补完整模型、依赖兼容或物理执行的空缺。

After execution approval, record each backend's actual code/checkpoint/statistics identity and strict-loading result. Check fixed RGB/q preprocessing and output mapping, then reset/task-switch behavior. Follow with a short rollout under the same environment/scoring protocol, recording raw targets, sent commands, feedback, resource peaks and terminal state. Reuse CPU evidence without treating it as full-model or physical execution evidence.
