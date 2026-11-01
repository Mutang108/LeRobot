# 版本记录 / Changelog

## Unreleased

修正OFT本地处理器的自动分派：显式使用固定后端图像类与标准tokenizer，避免非交互加载陷入checkpoint自定义代码确认；补充CPU回归及原生依赖说明。

Load OFT processors through pinned implementation classes and a standard tokenizer, avoiding checkpoint custom-code prompts during noninteractive local loading. Add CPU regression coverage and native dependency guidance.

新增SmolVLA、SO101版π₀.₅、GR00T N1.5及OpenVLA-OFT单front推理包装与本地加载入口，复用六关节chunk执行、单位校验和回合/任务边界清缓存。补齐加载前参数检查、BF16输出转换、严格加载报告及OFT平台参数；同步接口、依赖和Agent指南，并增加独立CPU接口CI。完整模型、GPU及闭环验收待完成，未增加VLA训练入口。

Add four single-front VLA wrappers/local loaders with shared joint-chunk execution and boundary resets. Validate execution settings before loading, handle BF16 outputs, retain strict-loading reports and expose the OFT platform flag. Update interface/dependency/agent guides and add CPU interface CI. Full-model/GPU/rollout acceptance and VLA training remain separate.

同步首页双语导航、开发/训练入口、架构与依赖、接口/标定边界及来源说明。区分历史v0.1.x源码快照与LiftCube规则开发基准，明确64像素是参考默认值、规则推理无需演示或训练、自研末端层不继承六关节训练验收。运行控制、配置和历史成绩不变。

Synchronize README navigation, development/training routes, architecture, dependencies, calibration contracts and provenance. Distinguish initial v0.1.x snapshots from the LiftCube rule reference, clarify default image sizing and dataset-free rule inference, and keep planar-control acceptance separate from joint training. Runtime behavior, configurations and historical results are unchanged.

整理LiftCube视觉规则脚本作为开发基准，补充正式标定/坐标与控制SDK的分层适配方法。记录冻结定位调度变体45/100的开发结果及其输入/验证范围，与旧协议和未验收的自研末端执行层区分；不更改运行控制或既有成绩。

Prepare the LiftCube visual rule script as a development reference, with a calibration/frame and control-SDK migration guide. Document the frozen scheduling variant's 45/100 result and evidence scope separately from older protocols and the unaccepted planar EE controller, without changing runtime control or historical results.

新增显式可选`planar_ee_dev`：绝对桌面XY米＋夹爪motor，固定参考高度/接近轴、允许绕轴转动，带原范围/限步/跟踪与失败保持；保留六关节默认，拒绝旧抓取脚本误接。自研接口不标作课程正式IK，补充标定来源边界和软件检查。

Add opt-in `planar_ee_dev` absolute table-XY/jaw control with fixed height/approach-axis references, free axial rotation, bounded commands and latched fault holds. Preserve joint defaults and reject accidental use by the old grasp runner. Document its non-official status, calibration provenance and software checks.

记录闭合几何的独立对照：动态中点候选4/10，现有定位调度方案6/10；候选推移和实测运动指标未改善，不并入运行代码，现有控制与默认保持。

Document the independent closure-geometry comparison: 4/10 for the dynamic-midpoint candidate versus 6/10 for existing localization scheduling. Reject the candidate; runtime code and defaults remain unchanged.

新增可选`plan_while_opening`：在原开爪期间积累视觉定位，保留完整开爪轨迹、实测到位等待和旧默认。原辅助几何方案扩展后累计13/30成功；新选项十场景配对仍为5/10，共同成功回合中位耗时9.40→9.33仿真秒，无精度改善证据。新增配置/回退测试和完整输入命令等价检查。

Add optional `plan_while_opening` localization without changing jaw-opening commands, measured-state dwell or defaults. The assisted baseline now has 13/30 successes. The option retains 5/10 on paired scenes, with common-success median time of 9.40→9.33 simulation seconds and no demonstrated accuracy gain. Add configuration/fallback tests and full-input command-equivalence checks.

规则入口新增可选固定CAD抓取姿态选择及“低速离桌→转动抬升”，中间阶段按实测FK高度切换、不冒充成功。实际相机外参读取需显式开关并逐reset记录，与固定标定分组；保留原动作/物理/评分边界，不与旧输入/场景集作单因素比较。

Add opt-in CAD grasp selection and slow-clearance/transport staging with a measured-FK height gate, not a success shortcut. Explicitly opt into and record actual camera/base calibration per reset, separately from fixed-calibration results. Preserve motor/physics/scoring bounds; different inputs/scene sets are not a single-factor comparison.

规则入口支持独立标定的128/256像素单front输入，默认64不变；按尺寸缩放检测面积/像素阈值并校验配置与标定一致性。运行清单分别记录实际策略类及基类源码身份，避免诊断子类与默认实现混淆。

Allow explicitly calibrated 128/256-square front RGB in the rule runner, retaining the 64 default. Scale detection thresholds, validate shape agreement, and record actual-policy-class and base-code identities separately. Document the randomized-camera pose-cache pitfall and keep calibration-assisted variants distinct.

规则基线增加可选夹爪位置反馈闭合：限制目标领先量，在停滞代理出现后保持夹持目标；不读取接触/物体真值，不改执行器、原保护或成功条件。新增接触代理、空抓闭合及reset回归；说明课程规则基线、完整末端动作接口和RL主方法的边界。

Add optional jaw-position feedback closure with bounded command lead and a stall proxy, retaining the grasp target through lifting. No contact/object truth, actuator or scoring changes. Cover blocked/empty closure and reset; clarify the rule-baseline, full-task action-interface and required RL-method boundaries.

修正规则标定预检的朝向回算，阶段预检显式指定朝向。接入固定SO101关节链FK/IK及CPU导出工具，使已有成功控制方案可通过公共入口运行；保留URDF/Placo可选后端，新增来源/限位与朝向回归检查。

Fix calibration round trips to use each reference orientation and explicitly preflight phase orientations. Add fixed SO101 joint-chain FK/IK and a CPU exporter for the public runner, retaining optional URDF/Placo support and adding provenance/limit/orientation regressions.
规则基线支持独立的抬升朝向和可选有界稳态补偿；同一LiftCube开发场景三组TCP候选中两组通过原生抓取判定。保留失败，公开控制逻辑在完整记录输入上的命令/阶段逐项一致；不将开发选参结果当作泛化成功率或完整Placo入口验收。

Add separate lift orientation and optional bounded steady-bias correction. Two of three TCP variants pass native grasp scoring in one development scene; retain the failure and verify published commands/phases against complete recorded inputs. This is neither a generalization benchmark nor full Placo-runner acceptance.

规则策略保留原始关节反馈，不再将接触/求解器造成的微小反馈越界误判为非法动作；发送目标仍按原范围检查，跟踪保护不变。增加边界反馈回归测试。

Keep raw joint feedback instead of validating contact/solver overshoot as an outgoing action. Sent targets retain the original bounds and tracking guard; add a boundary-feedback regression test.

关闭LiftCube适配器中默认开启的原生演示记录器，避免GPU图像历史拼接导致显存增长，保留显式轨迹/RGB记录。规则策略改为从上一发送目标推进参考轨迹，实测反馈仍用于到位与滞后保护；增加针对性测试。不改物理和评分。

Disable the implicit LiftCube demonstration recorder to avoid growing GPU image concatenations while retaining explicit trajectory/RGB logging. Advance the rule-policy reference from the last sent target, retaining measured-state arrival and lag checks. Add targeted tests without changing physics or scoring.

修正规则策略的朝向突变：位置与朝向同步渐变，关节限速保留IK方向，阶段切换同时检查朝向；新增针对性回归测试。保持原物理、关节限位与成功判据。

Interpolate position and orientation together in the rule policy, preserve the IK joint direction under slew limiting, and require orientation arrival before stage transitions. Add targeted regression tests without changing physics, joint limits or success criteria.

新增单front RGB＋实际关节位置的LiftCube规则策略、标定检查与独立评测入口。保存阶段、请求/反馈和原生终态；附合成CPU测试。真实标定、IK后端及抓取仍待验收，不改变物理、演示标签或训练配置。

Add a front-RGB/joint-feedback LiftCube rule policy, calibration checks and an evaluation entry point with stage, command, feedback and terminal-image logging. Synthetic CPU coverage is included; matching calibration, real IK and physical grasp validation remain pending. Physics, demonstration labels and training configurations are unchanged.

补齐演示批量校验：拒绝空数据、非有限本体、错误图像类型和不对齐/非法逐帧元数据；保留未知成功标记和原动作标签。新增CPU回归与原生请求/执行限幅配对诊断说明；不改变物理、公共执行接口或训练配置。

Validate canonical demonstration metadata, empty inputs, image dtype and finite proprioception without changing action labels. Add CPU regressions and document native-request versus execution-clipping diagnostics; physics, public control and training configurations remain unchanged.

补充SO101公共接口的真实16tick验收范围与候选演示准入结论；区分已验证的命令/反馈/图像边界与尚待确认的数据标定、成功证据和真机时序。共同起点已合并，后续使用从最新main创建的短期分支。

Document the 16-tick SO101 runtime interface check and candidate-demonstration acceptance limits. Separate verified command/feedback/image boundaries from unresolved data calibration, success evidence and hardware timing. Continue development on short-lived branches from current main.

新增分层智能体开发指令：根目录负责项目导航与共同约定，LiftCube和PickOrange诊断目录提供运行前提、命令与验收。补充ACT/VLA接入、数据验收、新任务扩展、故障定位与交接流程。明确接口、来源清单和文档的同步要求；新增PR复核模板，公开指令不依赖个人部署或特定agent插件。

Add scoped agent instructions: repository navigation and shared conventions at the root, with prerequisites, commands and validation for LiftCube and PickOrange diagnostics. Cover ACT/VLA integration, data acceptance, task extensions, troubleshooting and handoff. Define contract, provenance and documentation updates, and add a PR checklist. Public instructions remain independent of personal deployments and agent plugins.

SO101适配器拒绝缺失/异常关节反馈与非有限图像，外部state别名需显式声明单位；读取实际环境周期，并在诊断信息中分开记录请求目标、发送目标和动作后实测位置。新增5项CPU接口检查；不改变物理、关节限位或成功判据。

Reject missing/malformed SO101 feedback and non-finite images; require explicit units for external state aliases. Read the environment clock and report requested, sent and measured joint values separately. Add five CPU interface checks without changing physics, limits or success criteria.

修正LiftCube SmallCNN在均匀网格上使用普通平均池化，避免64×64输入的严格CUDA确定性反向报错；保留权重结构，并补充与自适应池化的数值/梯度等价检查。非整除网格保留原实现。

Use ordinary average pooling for uniform LiftCube SmallCNN grids, avoiding the strict deterministic CUDA backward error for 64×64 inputs while preserving checkpoint keys. Add value/gradient equivalence checks; non-uniform grids retain adaptive pooling.

接入LiftCube的RGB＋关节/命令融合、30Hz训练动作保持、分离超时bootstrap/GAE和完整BC/PPO边界存档恢复；新增TensorBoard曲线入口及合成CPU检查。补充ACT/VLA任务迁移与专家演示采集路线。演示动作标定及真实设备路径仍待验收，原评测配置和物理保持不变。

Connect LiftCube RGB/proprioceptive fusion, 30Hz training action holding, separate timeout bootstrap/GAE masks and complete BC/PPO boundary checkpoints. Add TensorBoard logging, synthetic CPU checks and task-transfer/data-collection guidance. Demonstration calibration and device-level validation remain pending; evaluation defaults and physics are unchanged.

精简中英文README的重复提醒与内部过程说明，保留任务状态、使用入口和关键接口限制，详细内容链接到开发文档。

Trim repeated caveats and internal process notes from both READMEs, retaining task status, entry points and key interface limits with links to development documentation.

README首页调整为海报置顶、居中项目标题和简介、English | 简体中文切换及文档导航；中英文采用相同结构，保留原海报和正文，不增加网页运行依赖。

Restyle both README headers with the existing poster first, centered project title and introduction, an English | 简体中文 switch and document navigation. Preserve poster assets and body content without adding web runtime dependencies.

根README拆为可切换的中文和英文入口，按rl-garden职责分层整理导航，补充现有starter包的架构、模型扩展与分支协作约定。新增RLinf接入评估，区分VLA训练参考、SO101硬件支持与课程任务适配缺口；不更改运行代码、依赖或冻结评测配置。

Split the root README into linked Chinese and English editions, organize navigation by rl-garden-style responsibilities, and document the existing starter package, model extension points and branch workflow. Add an RLinf assessment distinguishing VLA training references, SO101 hardware support and course-task adaptation gaps. Runtime code, dependencies and frozen evaluation configurations remain unchanged.

导入课程LiftCube起点代码及配置，新增独立运行说明、原始／当前文件哈希和可逆接口补丁。加入SO101补零、相机与时序检查、自动重置前终态捕获、纯渲染准备及超限数据拒绝；保留原评测配置和既有上游快照。镜像、数据、权重及私有部署记录不纳入仓库。

Import the course LiftCube starter and configurations with standalone runtime instructions, original/current hashes and a reversible interface patch. Add SO101 padding, camera/timing checks, pre-reset terminal capture, render-only preparation and out-of-range data rejection; preserve the evaluation configuration and existing upstream snapshots. Images, datasets, weights and private deployment records remain external.

补充LiftCube参考训练器的本体融合、BC→RL时序、存档恢复及PPO超时待办；注明4回合/640tick仅为起点配置，并更新导入清单说明。

Document pending proprioceptive fusion, BC-to-RL timing, checkpoint recovery and PPO timeout handling in the LiftCube reference trainers. Clarify that four episodes/640 ticks are starter defaults and update the import inventory description.

补充配套环境的交大云盘获取入口、跨平台SHA256核验与镜像导入步骤；同步开发起步、依赖和部署说明，区分LiftCube课程环境与历史上游快照。

Add the matching image's SJTU cloud-drive link, cross-platform SHA256 checks and import steps; align the development, dependency and deployment guides while keeping the LiftCube runtime separate from historical upstream snapshots.

依据当前项目任务说明重写README与海报文案，突出视觉多物体抓放、遥操与BC/VLA基线、强化学习、泛化评测和安全部署。海报保留原有字体字号、排版、纯色图形及官方照片，重新生成原生4K PNG和矢量文字SVG。

Rewrite the README and poster copy around the current task specification: visual multi-object pick-and-place, teleoperation and BC/VLA baselines, reinforcement learning, generalization evaluation and safe deployment. Retain the poster's typefaces, sizes, layout, flat-color artwork and official photograph, regenerating the native-4K PNG and outlined-text SVG.

将当前项目说明对齐为视觉抓放，保留早期海报，明确海报不定义控制接口。新增开发起步指南、观测/动作/数据/恢复接口约定和无依赖的实验计划检查器：检查 schema、动作单位/关节顺序、控制频率、chunk 执行长度、数据划分和预算单位。未确认参数保留空白，检查结果不替代训练验收。

Align the project description with visual pick-and-place while retaining the earlier poster and distinguishing it from the control specification. Add a development guide, observation/action/data/recovery contracts and a dependency-free plan inspector covering schemas, action units/joint order, control timing, chunk execution, dataset splits and budget units. Leave unconfirmed values blank; static checks do not replace training acceptance.

仓库检查同时覆盖未暂存且未忽略的项目新文件，避免新增源码和文档遗漏语法/链接检查；上游文件与公开工具哈希清单保持原样。

Repository checks now include non-ignored new first-party files before staging, preventing new source and documentation from missing syntax/link checks. Upstream files and public-utility hash manifests remain unchanged.

新增训练前验证工具：末帧与超时转移检查、ACT padding、匹配的随机数/采样位置存档及独立进程恢复对照。工具独立于上游源码，尚未接入正式训练入口，使用范围见[训练前验证](docs/TRAINING_VALIDATION.md)。

Add pre-training checks for final observations and timeout transitions, ACT padding, matching RNG/sample-position checkpoints and fresh-process recovery. The tools are separate from upstream sources and not yet integrated into a production trainer; see [training validation](docs/TRAINING_VALIDATION.md) for scope.

## v0.1.1-baseline — 2026-10-02

更新开发基准说明与 README，明确版本记录流程。上游源码和依赖版本保持不变，保留 `v0.1.0-baseline` 供历史对照。

Update the baseline guide and README, and document the version-history workflow. Upstream sources and dependency versions are unchanged; `v0.1.0-baseline` remains available for historical comparison.

统一全海报文字层级，修复01/02/03对应字号、字重、行距及字形宽窄不一致；长句换行而不压缩。加入字体层级与等比缩放回归检查，背景及官方照片不变。

Unify typography across the poster, correcting inconsistent sizes, weights, line heights and glyph proportions in sections 01/02/03; wrap long sentences instead of compressing them. Add regression checks for type roles and isotropic scaling, leaving the background and official photograph unchanged.

海报文字改为按已确认双语版位置重绘的字体轮廓，提供原生4K文字PNG与可缩放文字SVG；背景及图形配色校准到最初课程海报的纯色区域，移除后续生成的纹理、渐变及新增装饰，保留官方照片，统一小标题中英文颜色，并核对官方英文术语。

Rebuild poster lettering as font outlines at the approved bilingual positions, supplying native-4K text in PNG and scalable text in SVG; calibrate flat background and graphic colors to the first course poster, remove subsequently generated texture, gradients and added decorations, retain the official photograph, match bilingual subheading colors, and review official English terminology.

将海报中的机器人照片替换为官方 SO-101 清晰原图，仅等比缩放并裁切背景留白；同步更新双语致谢和图片来源。`v0.1.0-baseline` 标签不变。

Replace the robot photograph with the official clear SO-101 image, using proportional resizing and background-margin cropping only; update bilingual acknowledgements and image attribution. The `v0.1.0-baseline` tag remains unchanged.

## v0.1.0-baseline — 2026-10-02

建立项目源码整合初始基准：固定三个上游项目的完整快照、保留原始许可、物化 LFS 测试素材，并加入内容清单与校验、双语项目介绍和4K插值海报。

Establish the initial source-integration baseline: pin complete snapshots of three upstream projects, preserve their licenses, materialize LFS test fixtures, and include manifests, checks, a bilingual overview and an interpolated 4K poster.

增加统一无 GPU 检查入口、检查器单元测试、GitHub Actions 自动检查、基准使用方法和贡献规范。当前版本不宣称正式任务训练或真机部署完成。

Add a unified GPU-free check entry point, checker unit tests, GitHub Actions checks, baseline usage instructions and contribution guidelines. This version does not claim completed formal-task training or hardware deployment.
