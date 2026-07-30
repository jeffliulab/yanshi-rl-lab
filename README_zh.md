[![Language: English](https://img.shields.io/badge/Language-English-2f81f7?style=flat-square)](README.md) [![语言: 简体中文](https://img.shields.io/badge/语言-简体中文-e67e22?style=flat-square)](README_zh.md)

# Yanshi RL Lab（偃师 RL 实验室）

> 🤖 **如果你是 AI agent，请先读 [AGENTS.md](AGENTS.md)** —— 那是面向机器的入口：
> 分层规则、每个事实住在哪、以及各条命令。

[![IsaacLab](https://img.shields.io/badge/IsaacLab-2.3.2-silver)](https://isaac-sim.github.io/IsaacLab/)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://docs.python.org/3/whatsnew/3.11.html)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
[![Status](https://img.shields.io/badge/status-under_construction-orange.svg)]()

厂商中立的足式机器人强化学习框架：Isaac Lab 训练、MuJoCo 验证（sim2sim）、
所有机器人考同一张考卷（**Yanshi Rank** 天梯）。

名字取自「偃师」——《列子》中为周穆王献上会走路偶人的传奇工匠。偃师不是
那个偶人，他是**教偶人走路的人**。这正是本框架的定位：它不是任何一台机器人
的身体，而是让各家机器人身体学会运动的工匠作坊。

> 🚧 **建设中（v0.1 进行中）。** 首发机器人：宇树 G1、智元灵犀 X2、
> Berkeley Humanoid Lite。三台全部通过训练与 sim2sim 验收后，本 README
> 定稿。

## 设计铁律（已冻结）

- 一台机器人 = 一个档案目录（`robots/<厂商>/<型号>/`）；任务配方只通过语义
  槽位引用机器人——共享代码里零具体身体名。
- 档案齐全的机器人用**空壳**任务覆盖层即可训练；覆盖项只在实测证明必要时
  才加。
- 从训练到部署只有一份策略契约（`schema_version: 2`）；sim2sim 不过门，
  训练不算完成。
- 第三方资产按钉死的上游 commit 拉取（`assets/fetch.py`），绝不入库。
- 验收门线只存 YAML，绝不写进代码。
