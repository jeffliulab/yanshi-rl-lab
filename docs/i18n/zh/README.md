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
教足式机器人走路的开源框架，不绑任何一家厂商：在 Isaac Lab 里训，
在 MuJoCo 里验，所有机器人考同一张卷子。

<p align="center">
  <img src="../../media/hero-g1.gif" width="31%" alt="宇树 G1 平地行走">
  <img src="../../media/hero-x2.gif" width="31%" alt="智元灵犀 X2 行走">
  <img src="../../media/hero-bhl.gif" width="31%" alt="Berkeley Humanoid Lite 行走">
</p>

---

## 项目简介

大部分 locomotion 仓库都是围着一台机器人长出来的：这台机器的脾气散在各处配方里，
换一台就得从头重写。这个仓反过来做——只跟某一台机器有关的东西，全关在它自己的
*档案*里：有哪些关节、增益多大、哪块算脚哪块算手。训练配方碰不到实体，只认档案里的
名字。所以接一家新厂商，是新建一个目录，不是回头去改大家共用的代码。

训练曲线好看，不等于这条策略做完了。它还得换一个物理引擎再跑一遍：导出成 ONNX，
在 MuJoCo 里跑一条确定性闭环，拿跑之前就写死的那条线判分。线写在 YAML 里、
不写进 Python，所以想挪线就得改配置，diff 里一眼看得见。

挑这三台首发机器人，是因为它们差得够远——差得不够远，所谓的抽象就是自己骗自己：
一台 1.32 米的人形、一台来自另一家厂商的 1.3 米人形，还有一台 0.6 米、没有腰关节、
腿长只有前者四成的开源可打印机器人。每台登记过的机器人配置、它钉在上游哪个版本、
在这个框架里做到了哪一步，都写在 [ROBOTS_INTRO.md](../../../ROBOTS_INTRO.md)。

《列子·汤问》里记过一个叫偃师的匠人，给周穆王献了一个会走路的人偶。故事传到今天，
被记住的是匠人，不是那个人偶。这个仓也是这样：它不绑在任何一台机器人身上，
它是教它们走路的那间工坊。

## Yanshi Rank 天梯

三台在平地上考同一张卷子，四道题：命令写死、推理确定，每道题的及格线在开考前就按
各自的训练命令范围定好，中途摔了这道直接算没过。

| 机器人 | 原地转身 | 边走边转 | 直走 8 秒 | 慢走 8 秒 | 过门 |
|---|---|---|---|---|---|
| 宇树 G1（29 自由度） | 280.4° | 半径 0.60 m | 4.30 m | 2.20 m | 4/4 |
| 智元灵犀 X2 | 85.5° | 半径 2.10 m | 4.58 m | 1.79 m | 4/4 |
| Berkeley Humanoid Lite | 7.1° | 半径 1.61 m | 2.75 m | 0.02 m | 2/4 |

Berkeley Humanoid Lite 会走，但走得舒服的速度区间很窄：前进命令低于 0.4 m/s 它干脆
不理，而没过的那两道题恰好都在这个区间之外。原因查清楚了——训练时给的命令范围比考题
宽得多，慢速命令在采样里占比太低，它基本没练过——就这么记着，没往好听了写。这些都是
单种子跑出来的成绩，多种子要等 v0.2。

## 核心特性

- **机器人即档案**：只跟某一台机器有关的东西全在 `robots/<vendor>/<model>/profile.py` 里；档案写全了的机器人，用一个*空*的任务覆盖层就能开训。
- **sim2sim 才算做完**：一份策略契约（`schema_version: 2`）把一次训练一路带到 MuJoCo 回放；只能在训练器里跑的策略，不算数。
- **及格线写在配置里**：线都在 `benchmark/gates/*.yaml`，想挪线就得改配置文件，而不是在判分代码里悄悄改一行。
- **资产按引用取**：别人家的机器人模型不进这个仓，用的时候按钉死的版本号现拉——各家许可证不一样。
- **测试不挑机器**：档案、契约、及格线解析和脚手架，在一台没有仿真器、没有 GPU 的机器上都能测。

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
    --task Yanshi-Velocity-Flat-Unitree-G1-Dof29-v0 --headless \
    --num_envs 4096 --max_iterations 10000 --seed 42

# 3. 回放某个 checkpoint 并录像
./isaaclab.sh -p scripts/rsl_rl/play.py \
    --task Yanshi-Velocity-Flat-Unitree-G1-Dof29-v0 \
    --checkpoint logs/rsl_rl/<run>/model_9999.pt --video --headless

# 4. 在 MuJoCo 里跑 sim2sim 验收门——纯 CPU，这一步才是验收
python scripts/sim2sim/run_gates.py \
    --gates benchmark/gates/velocity-flat-turn.yaml \
    --contract logs/rsl_rl/<run>/params/contract.json \
    --policy logs/rsl_rl/<run>/exported/policy.onnx
```

全部通过时退出码为 `0`，有门未过为 `1`，触发否决线为 `2`。

## 接入一台新机器人

```bash
python scripts/tools/new_robot.py --vendor <vendor> --model <model>
```

脚手架会生成一份档案骨架和配套测试。把关节、增益、哪块算脚哪块算手填好，
在 `assets/registry.py` 里钉死模型从哪来，然后拿一个空的任务覆盖层开训。
只有当测量结果证明确实需要，才往覆盖层里加东西——并且把那次测量写进 commit message。

## 当前状态

v0.1 开发中。三台首发机器人里两台过了平地的验收，第三台会走，但因为上面说的原因缺两道题。
崎岖地形、域随机化和天梯站上线都排在 v0.2。本页成绩都是单种子跑出来的，请照这个前提读。

## 致谢

构建于 [Isaac Lab](https://github.com/isaac-sim/IsaacLab) 与
[RSL-RL](https://github.com/leggedrobotics/rsl_rl)。机器人模型来自各自厂商：
[宇树](https://github.com/unitreerobotics)、[智元](https://github.com/AgibotTech)、
[Berkeley Humanoid Lite](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite)。
面向 agent 的约定遵循 [agent-rules](https://github.com/jeffliulab/agent-rules)。

MIT © 2026 Jeff Liu。部分启动脚本源自 Isaac Lab 模板（BSD-3-Clause），
机器人资产遵循各自厂商的许可证——详见 [NOTICE](../../../NOTICE)。
