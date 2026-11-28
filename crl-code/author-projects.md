# 大型方法：作者工程与教学子机制

大型方法采用两条相互补充的入口：固定提交的作者项目，及可快速运行的教学子机制。子机制解释一项思想，不替代完整方法；取得源码不代表安装完成，更不代表论文重训完成。登记机器清单见 `integrations/author_projects.json`，固定提交于2026-10-04核对。

| 工程 | 源码身份与实际运行入口 | 教学连接 |
|---|---|---|
| [DreamerV3](https://github.com/danijar/dreamerv3/tree/e01491fad6434b2245a3b8ca201dd7faedcc458c) | 作者维护的reimplementation；README→configs→main→agent→train；`python dreamerv3/main.py --logdir NEW_RUN_DIRECTORY --configs crafter --run.train_ratio 32` | `dyna_q`理解经验、模型、规划；不具备Dreamer潜在世界模型与想象actor–critic |
| [TD-MPC2](https://github.com/nicklashansen/tdmpc2/tree/e9f59321933cbc8e11a002b842adc7d4ffae8ff1) | 官方源码；README→依赖→config→train→tdmpc2→world_model→online_trainer；在`tdmpc2/`运行README训练命令 | `learned_model_mpc`与`one_step_model`：学习同一小型经验模型，比较有限时域滚动规划与一步模型贪心；没有TD-MPC2潜在模型/价值联合训练，不等同完整TD-MPC2 |
| [DiscoRL](https://github.com/google-deepmind/disco_rl/tree/9059a29f7121d60948f25ef165e08e050e9399c8) | minimal JAX harness；真实入口`colabs/eval.ipynb`与`colabs/meta_train.ipynb`，支持固定规则meta-evaluation及meta-training | `idbd`仅元步长适应入门；不等同发现完整更新规则与原始搜索系统 |

## 获取与核对，不自动训练

在教程根目录执行。prepare只接受新目录，固定提交抓取并detached checkout；不安装依赖、不下载外部数据/权重、不执行notebook、不训练。失败留下目录供查证，不能覆盖重试。源码放在被gitignore排除的results目录，不纳入教材版权主树。

```bash
python3 scripts/author_project.py list
python3 scripts/author_project.py inspect dreamerv3
python3 scripts/author_project.py prepare dreamerv3 --directory results/author-checkouts/dreamerv3
python3 scripts/author_project.py verify dreamerv3 --directory results/author-checkouts/dreamerv3
```

其他ID为`tdmpc2`与`disco_rl`。verify核对commit、origin、工作树清洁与阅读路径；不验证模型效果，也不自动安装子模块或Git LFS资产。完整运行前需另登记平台、Python/依赖版本、环境、seed人口、步数、内部/外部评价、模型训练计算、显存/内存、checkpoint来源及新的输出目录。README示例中的700万步是作者例子，未在本次运行。

Dreamer要求Python3.11+与平台匹配JAX；TD-MPC2依赖域专用模拟器与GPU；DiscoRL依赖JAX/Haiku且meta-evaluation与meta-training预算分开。安装和训练应在隔离环境按固定源码核对，不能把本机CPU教学小任务成功当成上游工程兼容验收。

## 接到研究工作台

教学独立文件用Workbench的`rlworkbench.tutorial_adapter`：先按META核对可比任务/指标/计步，生成external协议并freeze，执行、import、audit、report、compare。大型作者工程保留自己的训练器；按Workbench `docs/external-adapters.md`输出完整锁定job的result/producer/原始材料。固定源码工具只完成源码身份接入，没有把作者训练输出自动适配成工作台回执；这项缺口必须继续明确保留。
