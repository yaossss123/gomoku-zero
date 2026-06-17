# Gomoku AlphaZero :black_circle: vs :white_circle:

> **一个从零手搓的五子棋 AI** —— 用 AlphaZero 架构让机器自己学会下棋。
>
> 没有人类棋谱，没有专家知识，只有自我对弈 + 深度学习 + 蒙特卡洛树搜索。

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## :sparkles: 项目亮点

- :brain: **纯 AlphaZero 架构** — 残差网络 (ResNet) + MCTS + 自我对弈，复现 DeepMind 经典论文
- :zap: **10 层残差块** — 128 通道的深度策略价值网络，约 2.5M 参数
- :trophy: **从零训练到击败规则 AI** — 不依赖任何人类棋谱数据
- :video_game: **双模式对战** — 终端命令行 + Tkinter 图形界面
- :bar_chart: **完整训练管线** — 数据增强、学习率调度、定期评估、训练曲线可视化
- :dart: **可选稠密奖励** — 活四/冲四/活三等中间奖励加速早期学习，训练后期自动衰减

## :movie_camera: 效果展示

### 图形界面

```bash
python gomoku_gui.py
```

支持选择 AI 类型（AlphaZero / 规则AI）、先后手切换、悔棋等功能。

### 终端对战

```bash
python play.py
```

```
  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14
 0  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·
 1  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·
 2  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·  ·
 3  ·  ·  ·  ●  ·  ·  ·  ○  ·  ·  ·  ●  ·  ·  ·
 4  ·  ·  ·  ·  ·  ·  ○  ·  ·  ·  ·  ·  ·  ·  ·
 ...
```

## :rocket: 快速开始

### 环境要求

- Python 3.8+
- **NVIDIA GPU**（训练必需，推理可用 CPU）
- CUDA 11.0+

### 安装

```bash
# 克隆项目
git clone https://github.com/your-username/gomoku-alphazero.git
cd gomoku-alphazero

# 安装依赖
pip install -r requirements.txt
```

### 训练

```bash
# 从 0 开始完整训练（5000 局，约 3-4 天 @ RTX 3060）
python train_with_improved_net.py

# 从断点继续训练
# 修改 train_with_improved_net.py 中的 resume_from 参数
```

训练过程中会每 200 局自动保存 checkpoint 并评估胜率：

```
[  100] 步数= 45.2 | Loss=4.8532 (P=2.1543, V=2.6989) | LR=0.002000 | Buffer= 36000 | 速度=25.3局/时
[  200] 步数= 52.1 | Loss=4.2156 (P=1.8765, V=2.3391) | LR=0.002000 | Buffer= 72000 | 速度=24.8局/时
--- 第 200 局评估 ---
  评估: 胜=12, 负=35, 平=3, 胜率=24.0%
```

### 对战

```bash
# 图形界面（推荐）
python gomoku_gui.py

# 终端对战
python play.py
```

## :building_construction: 架构设计

### 网络结构

```
输入 (15x15x3)
  │  通道0: 己方棋子  通道1: 对方棋子  通道2: 先后手标识
  ▼
┌─────────────────────────┐
│  初始卷积 (3→128 ch)     │
│  BatchNorm + ReLU       │
└────────────┬────────────┘
             ▼
┌─────────────────────────┐
│   残差块 x10            │
│  ┌────────────────────┐ │
│  │ Conv3x3 → BN → ReLU│ │
│  │ Conv3x3 → BN       │ │
│  │ + Skip Connection  │ │
│  │ → ReLU             │ │
│  └────────────────────┘ │
└────────────┬────────────┘
             ▼
     ┌───────┴───────┐
     ▼               ▼
┌─────────┐   ┌──────────┐
│ 策略头  │   │  价值头  │
│Conv1x1→2│   │Conv1x1→1 │
│  BN+ReLU│   │ BN+ReLU  │
│  FC→225 │   │ FC→256   │
│LogSoftmax│  │ FC→1     │
│         │   │  Tanh    │
└────┬────┘   └────┬─────┘
     ▼              ▼
 落子概率(225)   胜率(-1~1)
```

### 自我对弈训练循环

```
     ┌──────────────┐
     │  自我对弈     │ ← MCTS + 当前网络
     │  生成数据     │
     └──────┬───────┘
            ▼
     ┌──────────────┐
     │  数据增强     │ ← 8倍对称 (旋转+翻转)
     │  加入Buffer  │
     └──────┬───────┘
            ▼
     ┌──────────────┐
     │  采样训练     │ ← 策略损失 + 价值损失
     │  更新网络     │
     └──────┬───────┘
            ▼
     ┌──────────────┐
     │  评估 & 保存  │ ← vs 规则AI
     └──────┬───────┘
            └──────────→ 循环
```

### 蒙特卡洛树搜索 (MCTS)

每步落子时，MCTS 执行数百次模拟来探索后续变化：

| 阶段 | 说明 |
|------|------|
| **选择 (Selection)** | UCB + 神经网络先验引导，平衡探索与利用 |
| **扩展 (Expansion)** | 遇到未访问节点，用神经网络评估局面 |
| **回溯 (Backpropagation)** | 沿搜索路径回传价值估计 |
| **决策 (Action)** | 选择访问次数最多的动作 |

- 训练: 600 次模拟/步
- 对战: 1600 次模拟/步
- Dirichlet 噪声: 根节点添加随机扰动，保持探索多样性

## :file_folder: 项目结构

```
gomoku-alphazero/
├── config.py                      # 全局配置参数
├── gomoku_env.py                  # 五子棋环境（规则引擎 + 奖励系统）
├── improved_policy_value_net.py   # 策略价值网络（10层残差网络）
├── mcts.py                        # 蒙特卡洛树搜索
├── train_with_improved_net.py     # AlphaZero 训练主循环
├── play.py                        # 终端人机对战
├── gomoku_gui.py                  # Tkinter 图形界面对战
├── rule_based_ai.py               # 规则 AI（评估基准）
├── evaluate_model.py              # 模型评估脚本
├── dqn_agent.py                   # DQN 智能体（对比实验）
├── requirements.txt               # 依赖列表
├── models_standard/               # 训练好的模型权重
│   ├── model_5000.pth
│   └── model_final.pth
└── models/                        # 训练曲线图
    └── training_curves.png
```

## :wrench: 训练配置

核心参数均在 `config.py` 中管理：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `BOARD_SIZE` | 15 | 棋盘大小 |
| `NUM_RES_BLOCKS` | 10 | 残差块层数 |
| `NUM_CHANNELS` | 128 | 网络通道数 |
| `N_GAMES` | 5000 | 训练总局数 |
| `MCTS_SIMULATIONS_TRAIN` | 600 | 训练时每步MCTS模拟数 |
| `MCTS_SIMULATIONS_PLAY` | 1600 | 对战时每步MCTS模拟数 |
| `MCTS_C_PUCT` | 1.5 | UCB 探索常数 |
| `BATCH_SIZE` | 256 | 训练批次大小 |
| `LEARNING_RATE` | 0.002 | 初始学习率 |
| `USE_DENSE_REWARD` | False | 是否启用稠密奖励 |

## :question: 常见问题

<details>
<summary><b>训练多久能打过规则AI？</b></summary>
<br>
大约 1000~2000 局后开始有胜率，3000 局后通常能稳定超过 50% 胜率。
</details>

<details>
<summary><b>GPU 显存不够怎么办？</b></summary>
<br>
在 <code>config.py</code> 中调小：<br>
- <code>NUM_RES_BLOCKS</code>: 10 → 5~7<br>
- <code>BATCH_SIZE</code>: 256 → 128<br>
- <code>NUM_CHANNELS</code>: 128 → 64
</details>

<details>
<summary><b>想加快训练速度？</b></summary>
<br>
减小 <code>MCTS_SIMULATIONS_TRAIN</code>（如 400），但会牺牲 AI 质量。也可以启用稠密奖励 (<code>USE_DENSE_REWARD=True</code>) 加速早期学习。
</details>

<details>
<summary><b>没有 GPU 能训练吗？</b></summary>
<br>
理论上可以，但 15x15 棋盘 + 10 层残差网络在 CPU 上训练极慢（可能需要数周）。建议至少使用一块 NVIDIA GPU。推理/对战对算力要求较低。
</details>

<details>
<summary><b>稠密奖励是什么？</b></summary>
<br>
除了终局胜负奖励，在局中对有意义的着法给予小额即时奖励：
<ul>
<li>形成活四/冲四/活三 → 正奖励</li>
<li>堵住对手的四/三 → 正奖励</li>
<li>靠近中心落子 → 小奖励</li>
</ul>
训练后期稠密奖励权重自动衰减，避免干扰最终策略学习。
</details>

## :computer: 技术栈

| 组件 | 技术 |
|------|------|
| 深度学习框架 | PyTorch 2.0+ |
| 网络架构 | ResNet (10 residual blocks, 128 channels) |
| 搜索算法 | Monte Carlo Tree Search (MCTS) |
| 训练范式 | AlphaZero self-play |
| 数据增强 | 8x symmetry (rotation + flip) |
| GUI | Tkinter |
| 数值计算 | NumPy |
| 可视化 | Matplotlib |

## :books: 参考文献

- [Silver, D. et al. "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm" (AlphaZero)](https://arxiv.org/abs/1712.01815)
- [Silver, D. et al. "Mastering the Game of Go without Human Knowledge" (AlphaGo Zero)](https://www.nature.com/articles/nature24270)
- [Kocsis, L. & Szepesvári, C. "Bandit based Monte-Carlo Planning" (UCT)](https://link.springer.com/chapter/10.1007/11871842_29)

## :page_facing_up: License

MIT License - 自由使用和修改。
