# Yanshi RL Lab（偃師 RL ラボ）

[![Isaac Lab](https://img.shields.io/badge/Isaac_Lab-2.3.2-76b900?style=flat-square)](https://isaac-sim.github.io/IsaacLab/)
[![Python](https://img.shields.io/badge/Python-3.11-3776ab?style=flat-square)](https://docs.python.org/3/whatsnew/3.11.html)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)](../../../LICENSE)
[![Status](https://img.shields.io/badge/Status-v0.1-orange?style=flat-square)](../../../CHANGELOG.md)

<p>
<a href="../../../README.md"><img src="https://img.shields.io/badge/Language-English-2f81f7?style=flat-square" alt="English"></a>
<a href="../zh/README.md"><img src="https://img.shields.io/badge/%E8%AF%AD%E8%A8%80-%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87-e67e22?style=flat-square" alt="简体中文"></a>
<a href="README.md"><img src="https://img.shields.io/badge/%E8%A8%80%E8%AA%9E-%E6%97%A5%E6%9C%AC%E8%AA%9E-bf3989?style=flat-square" alt="日本語"></a>
<a href="../fr/README.md"><img src="https://img.shields.io/badge/Langue-Fran%C3%A7ais-8250df?style=flat-square" alt="Français"></a>
</p>

脚式ロボットのためのベンダー中立な強化学習フレームワーク。Isaac Lab で学習し、
MuJoCo で検証し、すべてのロボットを同じ試験で採点します。

<p align="center">
  <img src="../../media/hero-g1.gif" width="31%" alt="平地を歩く Unitree G1">
  <img src="../../media/hero-x2.gif" width="31%" alt="歩行する AgiBot Lingxi X2">
  <img src="../../media/hero-bhl.gif" width="31%" alt="歩行する Berkeley Humanoid Lite">
</p>

---

## 概要

多くの locomotion リポジトリは 1 台のロボットを中心に作られています。その機体固有の癖が
タスク設定のあちこちに散らばり、2 台目へ移そうとすると書き直しになります。本リポジトリは
これを逆にしました。ロボットは *プロファイル*（関節、ゲイン、身体部位の意味的マッピング）
であり、タスク設定はその意味的スロットを通してのみ機体に触れます。新しいベンダーを
加えることは、共有コードを編集することではなく、ディレクトリを 1 つ増やすことです。

学習曲線が良く見えても、方策が完成したことにはなりません。ここでは第 2 の物理エンジンを
生き延びる必要があります。すべての方策は ONNX に書き出され、MuJoCo の決定論的な閉ループで
再生され、その実行が存在する前に凍結された閾値で判定されます。閾値は Python ではなく
YAML にあるため、変更すれば必ず目に見える差分として残ります。

最初の 3 台は、抽象化を誠実に保つのに十分なほど互いに異なります。1.32 m のヒューマノイド、
別のベンダーによる 1.3 m のヒューマノイド、そして腰関節を持たず脚の長さが 4 割しかない
0.6 m のオープンソース 3D プリント機です。登録済みの各構成が何にピン留めされ、
本フレームワークでどこまで進んでいるかは [ROBOTS_INTRO.md](../../../ROBOTS_INTRO.md) に一覧があります。

名前がそのまま主張です。偃師（えんし）は、周の穆王に歩く自動人形を献上した工匠でした。
彼はロボットではなく、ロボットに歩き方を教えた側の人間です。このリポジトリは工房であって、
特定の身体ではありません。

## Yanshi Rank（ランキング）

すべてのロボットが平地で同じ 4 ゲートの試験を受けます。指令は固定、推論は決定論的です。
合格ラインはロボットごとに、そのロボット自身が学習した指令範囲から事前登録され、
転倒したゲートは即座に不合格となります。

| ロボット | その場旋回 | 歩行旋回 | 直進 8 秒 | 低速歩行 8 秒 | ゲート |
|---|---|---|---|---|---|
| Unitree G1（29 自由度） | 280.4° | 半径 0.60 m | 4.30 m | 2.20 m | 4/4 |
| AgiBot Lingxi X2 | 85.5° | 半径 2.10 m | 4.58 m | 1.79 m | 4/4 |
| Berkeley Humanoid Lite | 7.1° | 半径 1.61 m | 2.75 m | 0.02 m | 2/4 |

Berkeley Humanoid Lite は歩けますが、指令の「快適域」が非常に狭く、0.4 m/s を下回ると
指令を完全に無視します。不合格となった 2 つのゲートはいずれもこの域の外にあります。
原因は特定済みで、隠さずここに記します。学習時の指令範囲が試験点より遥かに広く、
低速指令がサンプリング分布の中で希少になっていたためです。以上はすべて単一シードの結果で、
複数シードのベースラインは v0.2 の基準です。

## 主な特徴

- **ロボット＝プロファイル**：ただ 1 台にだけ当てはまる事実はすべて `robots/<vendor>/<model>/profile.py` に置き、プロファイルが完備したロボットは*空の*タスクオーバーレイで学習します。
- **sim2sim を完了条件に**：1 つの方策コントラクト（`schema_version: 2`）が学習から MuJoCo 再生までを貫きます。学習器の中でしか動かない方策は成果ではありません。
- **宣言的なゲート**：閾値は `benchmark/gates/*.yaml` にあり、ゴールを動かせば設定ファイルの差分として現れます。判定コードに埋もれた 1 行にはなりません。
- **アセットは参照で**：第三者のロボットモデルは固定した上流コミットから取得し、リポジトリには入れません。ライセンスがベンダーごとに異なるためです。
- **CPU だけで走るテスト**：プロファイル、コントラクト、ゲート解析、スキャフォルダはシミュレータも GPU もなしに検証できます。

## インストール

[Isaac Lab 2.3.2](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)
とその Isaac Sim 依存が必要です。同じ環境に本拡張をインストールしてください。

```bash
git clone https://github.com/jeffliulab/yanshi-rl-lab.git
cd yanshi-rl-lab
pip install -e source/yanshi_rl_lab

yanshi doctor          # 環境セルフチェック。CPU のみ、決して落ちない
yanshi assets fetch    # 固定版のロボットモデルを取得（本リポジトリには含まれません）
```

## クイックスタート

```bash
# 1. テスト -- CPU のみ、シミュレータ不要
pytest -q

# 2. 学習（素の python ではなく Isaac Lab のランチャ経由）
./isaaclab.sh -p scripts/rsl_rl/train.py \
    --task Yanshi-Velocity-Flat-Unitree-G1-Dof29-v0 --headless \
    --num_envs 4096 --max_iterations 10000 --seed 42

# 3. チェックポイントを再生して録画
./isaaclab.sh -p scripts/rsl_rl/play.py \
    --task Yanshi-Velocity-Flat-Unitree-G1-Dof29-v0 \
    --checkpoint logs/rsl_rl/<run>/model_9999.pt --video --headless

# 4. MuJoCo で sim2sim ゲート -- CPU のみ、これが受け入れ検査
python scripts/sim2sim/run_gates.py \
    --gates benchmark/gates/velocity-flat-turn.yaml \
    --contract logs/rsl_rl/<run>/params/contract.json \
    --policy logs/rsl_rl/<run>/exported/policy.onnx
```

終了コードは全ゲート合格で `0`、不合格が 1 つでもあれば `1`、拒否ラインに触れたら `2` です。

## ロボットを追加する

```bash
python scripts/tools/new_robot.py --vendor <vendor> --model <model>
```

スキャフォルダがプロファイルの雛形とそのテストを生成します。関節、ゲイン、身体部位の
意味的マッピングを埋め、`assets/registry.py` にアセットの取得元を固定し、空のタスク
オーバーレイで学習を始めてください。オーバーライドは測定によって必要性が示されたときにのみ
追加し、その測定結果をコミットメッセージに残します。

## 現状

v0.1 を開発中です。最初の 3 台のうち 2 台が平地ゲートを通過し、3 台目は歩けるものの
上記の理由で 2 つのゲートに届いていません。不整地、ドメインランダマイゼーション、
ランキングサイトの公開は v0.2 の作業です。本ページの結果はすべて単一シードです。

## 謝辞

[Isaac Lab](https://github.com/isaac-sim/IsaacLab) と
[RSL-RL](https://github.com/leggedrobotics/rsl_rl) の上に構築しています。ロボットモデルは
各ベンダー提供です：[Unitree](https://github.com/unitreerobotics)、
[AgiBot](https://github.com/AgibotTech)、
[Berkeley Humanoid Lite](https://github.com/HybridRobotics/Berkeley-Humanoid-Lite)。
エージェント向けの規約は [agent-rules](https://github.com/jeffliulab/agent-rules) に従います。

MIT © 2026 Jeff Liu。一部のランチャスクリプトは Isaac Lab テンプレート（BSD-3-Clause）に
由来し、ロボットアセットは各ベンダーのライセンスに従います。詳細は
[NOTICE](../../../NOTICE) を参照してください。
