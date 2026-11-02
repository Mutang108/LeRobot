# PickOrange运行入口：智能体指令

适用于本目录，与[根AGENTS](../../AGENTS.md)同时遵守。先读[本目录README](README.md)、[环境指南](../../docs/PICK_ORANGE.md)、[资源清单](../../docs/RESOURCES.md)。本目录是独立模型评测；`course/scripts/`保留历史诊断，LiftCube继续使用其专属包。

- `common.py`校验协议、NPZ输入白名单和有界等待；`service.py`只负责固定GR00T N1.5双视角推理；`evaluate.py`负责原生动作转换、完整回合及reset前捕获；`audit.py`独立读盘。不要复制一套LiftCube工厂或将N1.6权重交给N1.5加载器。
- 默认`protocols/groot-native-assisted.json`明确允许上游物体真值夹爪辅助；不把真值发给网络，不称无特权视觉baseline或课程正式评分。换物理、辅助、相机、频率、限幅或时限须新命名协议，不能沿用历史结果。
- `--check`与`--help`不可创建AppLauncher或加载大模型。服务须显式`--run-service`，仿真须`--run-simulation`。文件IPC仅在受信任本地共享目录，不使用pickle或开放网络端口。
- 结果目录唯一，服务创建/evaluator独占claim；不复用旧ready、删除claim、覆盖失败或自动重跑。当前无自动resume；保留partial，不重复计入已完成seed。RNG存档不能冒称任意中途物理恢复。
- 原始模型motor、转换后rad、action manager raw和实测q分开，明确scale/offset与关节顺序；本原生辅助协议不软件限幅，其他包装的动作保护不得因此放宽。
- 官方特权教师代码已在`third_party/leisaac`。其重力/阻尼/直接写状态与成功覆盖必须在采集说明中披露；不得把它的生成成功当成固定物理BC准入。
- 命令与资源说明必须让新使用者从公开仓库操作；权重、数据、个人部署、凭据、设备或私有运维配置不入Git。训练流程见[训练与评测](../../docs/TRAIN_AND_EVALUATE.md)，当前推理服务不等于VLA训练器。

修改后在根目录的独立CPU环境运行：

```bash
python course/pickorange/evaluate.py --check
python -B -m unittest discover -s tests -p test_pickorange.py -v
python -B -m unittest discover -s tests -p 'test_vla*.py' -v
```

`test_pickorange.py`需NumPy，HDF5审核测试另需h5py；不得把skip当通过。测试使用替身模型与合成轨迹，不声称真实抓放通过。源码移植仍需另行获准的真实集成检查，不因CPU测试通过自行启动GPU。同步`manifests/course_files.json`、任务状态、对应README及影响到的根分流；不改冻结第三方来源来消除错误。
