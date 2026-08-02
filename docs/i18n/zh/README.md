# Yanshi RL Lab（偃师 RL 实验室）

[![Isaac Lab](https://img.shields.io/badge/Isaac_Lab-2.3.2-76b900?style=flat-square)](https://isaac-sim.github.io/IsaacLab/)
[![Python](https://img.shields.io/badge/Python-3.11-3776ab?style=flat-square)](https://docs.python.org/3/whatsnew/3.11.html)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](../../../LICENSE)
[![Status](https://img.shields.io/badge/Status-v0.1-orange?style=flat-square)](../../../CHANGELOG.md)

<p>
<a href="../../../README.md"><img src="https://img.shields.io/badge/Language-English-2f81f7?style=flat-square" alt="English"></a>
<a href="README.md"><img src="https://img.shields.io/badge/%E8%AF%AD%E8%A8%80-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-e67e22?style=flat-square" alt="简体中文"></a>
<a href="../ja/README.md"><img src="https://img.shields.io/badge/%E8%A8%80%E8%AA%9E-%E6%97%A5%E6%9C%AC%E8%AA%9E-bf3989?style=flat-square" alt="日本語"></a>
<a href="../fr/README.md"><img src="https://img.shields.io/badge/Langue-Fran%C3%A7ais-8250df?style=flat-square" alt="Français"></a>
</p>
面向足式机器人的厂商中立强化学习框架：在 Isaac Lab 里训练，在 MuJoCo 里验证，
所有机器人考同一张卷子。

<p align="center">
  <img src="../../media/hero-g1.gif" width="31%" alt="宇树 G1 平地行走">
  <img src="../../media/hero-x2.gif" width="31%" alt="智元灵犀 X2 行走">
  <img src="../../media/hero-bhl.gif" width="31%" alt="Berkeley Humanoid Lite 行走">
</p>

---

## 项目简介

多数 locomotion 仓库是围绕一台机器人建起来的：那台机器的特性散落在任务配方各处，
换第二台就等于重写。这个仓把关系倒过来——一台机器人是一份*档案*（关节、增益、
身体部位的语义映射），任务配方只通过语义槽位接触身体。接一家新厂商是加一个目录，
不是改公共代码。

训练曲线好看不等于策略做完了。这里它必须扛过第二个物理引擎：每个策略导出成 ONNX、
在 MuJoCo 里跑一条确定性闭环，再对照跑之前就冻结好的阈值判分。阈值只住 YAML、
不进 Python，所以改一条线是一次看得见、可审计的改动。

三台首发机器人差别足够大，才逼得出真正的抽象：一台 1.32 米的人形、
一台来自另一家厂商的 1.3 米人形，以及一台 0.6 米、无腰关节、腿长只有前者四成的
开源可打印机器人。

名字就是主张。偃师是向周穆王献上会走路的人偶的那位巧匠——他不是机器人，
他是教会机器人走路的人。这个仓是那间工坊，不是任何一具身体。

## Yanshi Rank 天梯

每台机器人在平地上考同一张四门卷子，命令固定、推理确定。门线在跑之前按各自训练出的
命令包络预注册，摔倒直接判该门不过。

| 机器人 | 原地转身 | 边走边转 | 直走 8 秒 | 慢走 8 秒 | 过门 |
|---|---|---|---|---|---|
| 宇树 G1（29 自由度） | 275.9° | 半径 0.61 m | 4.49 m | 2.29 m | 4/4 |
| 智元灵犀 X2 | 85.5° | 半径 2.10 m | 4.58 m | 1.79 m | 4/4 |
| Berkeley Humanoid Lite | 7.1° | 半径 1.61 m | 2.75 m | 0.02 m | 2/4 |

Berkeley Humanoid Lite 会走，但只服务一个很窄的命令舒适区：前进命令低于 0.4 m/s
它完全不理会，而没过的那两道门恰好都落在这个区间之外。根因已定位并如实记录，没有粉饰——
训练时用的命令包络远宽于考卷考点，慢速命令在采样分布里占比过低。以上均为单种子结果，
多种子基线是 v0.2 的标准。

## 核心特性

- **机器人即档案**：凡是只对一台机器成立的事实都住在 `robots/<vendor>/<model>/profile.py`，档案完整的机器人用*空*的任务覆盖层训练。
- **sim2sim 是完成判据**：一份策略契约（`schema_version: 2`）把一次训练带到 MuJoCo 回放；只在训练器里能跑的策略不算结果。
- **声明式验收门**：阈值住在 `benchmark/gates/*.yaml`，挪门线就是一次配置文件的改动，而不是藏在判决代码里的一行。
- **资产按引用取**：第三方机器人模型从钉死的上游提交拉取、从不入库——各家许可证不同。
- **纯 CPU 测试套件**：档案、契约、门解析与脚手架都能在没有仿真器和 GPU 的机器上测。

## 安装

需要 [Isaac Lab 2.3.2](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
及其 Isaac Sim 依赖。把本扩展装进同一个环境：

```bash
git clone https://github.com/jeffliulab/yanshi-rl-lab.git
cd yanshi-rl-lab
pip install -e source/yanshi_rl_lab

yanshi doctor          # 环境自检，纯 CPU，不该崩
yanshi assets fetch    # 拉取钉死版本的机器人模型（不随仓库分发）
```

## 快速上手

```bash
# 1. 测试——纯 CPU，不需要仿真器
pytest -q

# 2. 训练（走 Isaac Lab 的启动器，不是裸 python）
./isaaclab.sh -p scripts/rsl_rl/train.py \
    --task Yanshi-Velocity-Flat-Unitree-G1-v0 --headless \
    --num_envs 4096 --max_iterations 10000 --seed 42

# 3. 回放某个 checkpoint 并录像
./isaaclab.sh -p scripts/rsl_rl/play.py \
    --task Yanshi-Velocity-Flat-Unitree-G1-v0 \
    --checkpoint logs/rsl_rl/<run>/model_9999.pt --video --headless

# 4. 在 MuJoCo 里跑 sim2sim 验收门——纯 CPU，这一步才是验收
python scripts/sim2sim/run_gates.py \
    --gates benchmark/gates/velocity-flat-turn.yaml \
    --contract logs/rsl_rl/<run>/params/contract.json \
    --policy logs/rsl_rl/<run>/exported/policy.onnx \
    --scene assets/unitree/g1/mjcf/g1/scene_23dof.xml
```

全部通过时退出码为 `0`，有门未过为 `1`，触发否决线为 `2`。

## 接入一台新机器人

```bash
python scripts/tools/new_robot.py --vendor <vendor> --model <model>
```

脚手架会生成档案骨架和它的测试。填好关节、增益与身体部位的语义映射，
在 `assets/registry.py` 里钉死资产来源，然后用空的任务覆盖层开训——
只有当测量证明确有必要时才加覆盖项，并把那次测量写进 commit message。

## 当前状态

v0.1 开发中。三台首发机器人里两台通过平地验收门；第三台会走，但因上文所述原因缺两道门。
崎岖地形、域随机化与天梯站的发布属于 v0.2。本页结果均为单种子，请照此解读。

## 致谢

构建于 [Isaac Lab](https://github.com/isaac-sim/IsaacLab) 与
[RSL-RL](https://github.com/leggedrobotics/rsl_rl)。机器人模型来自各自厂商：
[宇树](https://github.com/unitreerobotics)、[智元](https://github.com/AgibotTech)、
[Berkeley Humanoid Lite](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite)。
面向 agent 的约定遵循 [agent-rules](https://github.com/jeffliulab/agent-rules)。

MIT © 2026 Jeff Liu。部分启动脚本源自 Isaac Lab 模板（BSD-3-Clause），
机器人资产遵循各自厂商的许可证——详见 [NOTICE](../../../NOTICE)。
