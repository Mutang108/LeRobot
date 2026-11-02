# 数据与模型获取 / Data and model acquisition

先在[适配状态表](TASKS.md)选择任务和后端，再获取本页资产。Git只保存代码、配置和来源；权重、数据、标定与运行产物放在仓库外。下载模型不会启动训练或测试。访问权限和再分发许可分别确认。

Choose a task/backend in the [status matrix](TASKS.md). Git contains integration code, not model/data binaries. Store assets externally and respect each resource's access and redistribution terms.

## 专家演示 / Demonstrations

| 任务 / Task | 数据源 / Source | 使用前检查 / Before training |
| --- | --- | --- |
| LiftCube | [azazdeaz/teleop_so101_leisaac_lift_cube](https://huggingface.co/datasets/azazdeaz/teleop_so101_leisaac_lift_cube/tree/08b442d9cc7db587d811e2ddace3e0a6c22d519e)，17回合、front+wrist、30fps | 六关节顺序、标定/动作单位、视角选择、全回合成功、当前环境重放；当前仅完成元数据/统计核对 / Metadata checks only so far |
| PickOrange | [LightwheelAI/leisaac-pick-orange](https://huggingface.co/datasets/LightwheelAI/leisaac-pick-orange)，60回合、双RGB、30fps | 下载时固定实际revision；核对重力/辅助、动作、终态与数据划分，不能把episode结束标记当成功 / Pin the dataset revision and audit assistance, labels and outcomes |

历史`H2Ozone/lift_cube`动作范围问题属于另一候选数据，详见[状态表](TASKS.md)，不与上面17回合数据混用。所有候选均需经过数据准入；这里只提供来源，不承诺全部是合格训练样本。

## 四种模型 / Four model families

表内链接指向已识别版本。下载完整导出时保留配置、处理器、统计、索引及全部权重分片；只有`.safetensors`或LoRA文件通常不够。

Preserve complete exports, including processors, normalization, indices and every shard. A bare tensor file or LoRA adapter is not necessarily a runnable policy.

| 任务 / Task | 模型 / Model | 固定资源 / Pinned resource | 本仓库入口 / Integration |
| --- | --- | --- | --- |
| LiftCube | SmolVLA任务权重 | [azazdeaz/smolvla_so101_leisaac_lift_cube@a124aa1](https://huggingface.co/azazdeaz/smolvla_so101_leisaac_lift_cube/tree/a124aa19de5e6cc2be8e2731b79b3bec15235e5e) | `starter.models.smolvla:SmolVLAFrontPolicy`；原双视角，单front使用缺槽mask |
| LiftCube | GR00T N1.6任务权重 | [azazdeaz/groot_n1_6_so101_leisaac_lift_cube@b0c11df](https://huggingface.co/azazdeaz/groot_n1_6_so101_leisaac_lift_cube/tree/b0c11dfae768d97fc200b6010100c0d2c57bc3e1) | **暂未接入完整N1.6推理**；不能交给N1.5包装器 / N1.6 inference integration pending |
| LiftCube / PickOrange | π₀.₅ SO101域基座 | [dongyoonkim/so101-pi05-base@56325ba](https://huggingface.co/dongyoonkim/so101-pi05-base/tree/56325bafcca9ec3155b8bfdf655be7df17e7f09f) | `starter.models.pi05:Pi05FrontPolicy`为单front作者模型图；不是两任务的已验收专家 |
| PickOrange | GR00T N1.5任务权重 | [LightwheelAI/leisaac-pick-orange-v0@10f2014](https://huggingface.co/LightwheelAI/leisaac-pick-orange-v0/tree/10f20142f96f0b9ef139aa90855a73e28a83ca74) | [独立双相机服务与原生辅助评测](../course/pickorange/README.md) |
| PickOrange | SmolVLA任务权重 | [edge-inference/smolvla-so101-pick-orange@71cf4a9](https://huggingface.co/edge-inference/smolvla-so101-pick-orange/tree/71cf4a9d35ce317f6706efe1a9f9d4cbb2b8fb4d)、[wsagi/SmolVLA-PickOrange@c8c3318](https://huggingface.co/wsagi/SmolVLA-PickOrange/tree/c8c3318dba152b0ba671ff07b4314418d5aa4b4a) | 双视角模型；当前公共GR00T服务不加载SmolVLA，需接原生LeRobot处理器 / Separate dual-view backend work remains |
| PickOrange | π₀.₅任务候选 | [wsagi/Pi0.5-PickOrange@f6ffd30](https://huggingface.co/wsagi/Pi0.5-PickOrange/tree/f6ffd30124b31e192303ab99d5ce91466485a660) | 作者标注负面结果，尚未本地验收；与上述VLASH图不互换 / Unaccepted candidate, different graph |
| 两任务 / Both | OpenVLA-OFT | [SO101代码](https://github.com/YoungK82/OpenVLA-oft/tree/ac34367e52d52f967fa00b7bfef1ff3d2c9e253e)；[需访问批准的模型候选](https://huggingface.co/Atabaku/so101-openvla-oft-5000) | `starter.models.openvla_oft:OpenVLAOFTFrontPolicy`；缺已验收的完整SO101受训头/投影/统计组合，不提供随机头替代 |

### 下载示例 / Download examples

在独立资产准备环境安装Hugging Face CLI，不改模拟器依赖。下列`/models`、`/datasets`是使用者自己的外部存储挂载；先确认空间、许可及访问权限。需要登录时使用CLI交互登录，不把token写进命令、脚本或Git。

Use an asset-preparation environment, not the simulator environment. Replace the external storage paths. Use interactive authentication when required; never embed tokens in commands or Git.

```bash
hf download azazdeaz/smolvla_so101_leisaac_lift_cube \
  --revision a124aa19de5e6cc2be8e2731b79b3bec15235e5e \
  --local-dir /models/liftcube/smolvla-a124aa1

hf download LightwheelAI/leisaac-pick-orange-v0 \
  --revision 10f20142f96f0b9ef139aa90855a73e28a83ca74 \
  --local-dir /models/pickorange/groot-10f2014

hf download azazdeaz/teleop_so101_leisaac_lift_cube --repo-type dataset \
  --revision 08b442d9cc7db587d811e2ddace3e0a6c22d519e \
  --local-dir /datasets/liftcube/source-17ep
```

其余资源使用同一`hf download REPO --revision REV --local-dir DIR`形式，从上表完整URL取revision。PickOrange数据先从Hub提交历史选择版本再下载，记录到数据清单；不要使用未固定的`main`做正式训练输入。网络或访问审批失败时保留缺口，不用其他架构的权重替代。

## 后端、骨干与统计 / Backends, backbones and normalization

| 模型 / Model | 后端与额外资产 / Required components |
| --- | --- |
| SmolVLA | 仓库冻结[LeRobot v0.4.2](../third_party/lerobot/docs/source/smolvla.mdx)；匹配[SmolVLM2-500M配置/tokenizer@7b375e1](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct/tree/7b375e1b73b11138ff12fe22c8f2822d8fe03467)。任务权重保留自身pre/postprocessor与统计；LiftCube单front描述须填写真实state/action mean/std，详见[接口字段](VLA_INTERFACES.md) |
| GR00T N1.5 | [Isaac-GR00T@4af2b62](https://github.com/NVIDIA/Isaac-GR00T/tree/4af2b622892f7dcb5aae5a3fb70bcb02dc217b96)。按该版本安装说明准备Eagle、RADIO、tokenizer等依赖及缓存，保留训练形态metadata；PickOrange使用`so100_dualcam`/`new_embodiment`，不是单front配置 |
| GR00T N1.6 | [任务导出对应源版本@ead5283](https://github.com/NVIDIA/Isaac-GR00T/tree/ead52833afbbf4243f8cd5e7664f48a94de03b19)；独立环境。当前仅原生数据变换CPU检查，不提供假定兼容的N1.5命令 |
| π₀.₅域基座 | [作者SO101实现](https://github.com/jinnymo/so101-pi05-base)与其指定VLASH `22cbabfee0f57874987c75a35a7dac129e695db0`及作者补丁；保留state_cond图、内嵌统计和获准tokenizer。Gemma等底座许可单独遵守 |
| OpenVLA-OFT | 上表固定SO101后端、其Transformers/处理器依赖，以及同一次训练的完整核心、连续L1动作头、proprio projector、六维统计。通用OpenVLA-7B仅能验证核心，不能补齐这些部件 |

加载器使用本地资产。将后台缓存准备齐后设置`HF_HUB_OFFLINE=1`，避免推理时隐式下载；不以stub模块、`strict=False`或假统计通过加载。GR00T完整原生构造含CUDA探测，CPU-only组件检查不等于模型已加载。

Loaders use local assets. Cache required dependencies before offline inference. Preserve strict loading and model-specific normalization; CPU component checks do not establish full GR00T construction.

后续步骤见[训练与评测](TRAIN_AND_EVALUATE.md)。每项实验记录资源URL、完整revision、资产完整性、视角、数据划分和统计来源；不把本页状态当作自动开训或BC准入。
