"""
价值头运行演示
详细展示价值头如何从特征图生成局面评估值
"""
import numpy as np
import torch
import torch.nn.functional as F
from improved_policy_value_net import ImprovedPolicyValueAgent

print("="*70)
print("价值头运行演示")
print("="*70)

# 创建智能体
agent = ImprovedPolicyValueAgent(board_size=15, num_res_blocks=10)
net = agent.policy_value_net

# 创建几个不同的测试状态
test_states = []

# 状态1: 空棋盘
state1 = np.zeros((15, 15, 3), dtype=np.float32)
state1[:, :, 2] = 1  # 当前玩家为黑棋
test_states.append(("空棋盘", state1))

# 状态2: 黑棋优势（四连）
state2 = np.zeros((15, 15, 3), dtype=np.float32)
state2[7, 6:10, 0] = 1  # 黑棋四连
state2[8, 7, 1] = 1     # 白棋一子
state2[:, :, 2] = 1     # 当前玩家为黑棋
test_states.append(("黑棋四连（优势）", state2))

# 状态3: 白棋优势
state3 = np.zeros((15, 15, 3), dtype=np.float32)
state3[7, 6:10, 1] = 1  # 白棋四连
state3[8, 7, 0] = 1     # 黑棋一子
state3[:, :, 2] = 1     # 当前玩家为黑棋（面对白棋威胁）
test_states.append(("白棋四连（黑棋劣势）", state3))

# 状态4: 均势
state4 = np.zeros((15, 15, 3), dtype=np.float32)
state4[7, 7, 0] = 1
state4[7, 8, 0] = 1
state4[8, 7, 1] = 1
state4[8, 8, 1] = 1
state4[:, :, 2] = 1
test_states.append(("均势局面", state4))

print("\n准备测试 4 种不同的棋盘局面...")

net.eval()
with torch.no_grad():
    for test_name, state in test_states:
        print("\n" + "="*70)
        print(f"【测试局面】{test_name}")
        print("="*70)
        
        # 显示棋盘
        print("\n棋盘状态（中心区域 5:11, 5:11）:")
        print("     ", end="")
        for c in range(5, 11):
            print(f" {c:2d}", end="")
        print()
        for r in range(5, 11):
            print(f"  {r:2d} ", end="")
            for c in range(5, 11):
                if state[r, c, 0] == 1:
                    print(" ●", end="")  # 黑棋
                elif state[r, c, 1] == 1:
                    print(" ○", end="")  # 白棋
                else:
                    print("  ", end="")  # 空位
            print()
        
        # 转换为张量
        state_tensor = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(agent.device)
        
        print("\n" + "-"*70)
        print("价值头处理流程：")
        print("-"*70)
        
        # 共享特征提取
        x = net.conv_init(state_tensor)
        x = net.bn_init(x)
        x = F.relu(x)
        
        for res_block in net.res_blocks:
            x = res_block(x)
        
        print(f"1️⃣  共享特征: {x.shape} -> (1, 128, 15, 15)")
        print(f"   通过初始卷积 + 10个残差块提取")
        
        # 价值头卷积
        value = net.value_conv(x)
        print(f"\n2️⃣  卷积降维: {x.shape} -> {value.shape}")
        print(f"   Conv2d(128 -> 2, kernel_size=1)")
        print(f"   作用: 将128维特征压缩到2维，聚焦关键信息")
        
        value = net.value_bn(value)
        value = F.relu(value)
        print(f"   BatchNorm + ReLU: 归一化并激活")
        
        # 展平
        value_flat = value.reshape(value.size(0), -1)
        print(f"\n3️⃣  展平操作: {value.shape} -> {value_flat.shape}")
        print(f"   (1, 2, 15, 15) -> (1, 450)")
        print(f"   将2个通道的15x15特征图拉成一维向量")
        
        # 第一个全连接层
        value_fc1 = F.relu(net.value_fc1(value_flat))
        print(f"\n4️⃣  第一层全连接: {value_flat.shape} -> {value_fc1.shape}")
        print(f"   Linear(450 -> 256) + ReLU")
        print(f"   作用: 降维并提取高级特征")
        
        # 第二个全连接层
        value_fc2 = net.value_fc2(value_fc1)
        print(f"\n5️⃣  第二层全连接: {value_fc1.shape} -> {value_fc2.shape}")
        print(f"   Linear(256 -> 1)")
        print(f"   输出单个值（未归一化）: {value_fc2.item():.4f}")
        
        # Tanh激活
        value_final = torch.tanh(value_fc2)
        print(f"\n6️⃣  Tanh激活: {value_fc2.item():.4f} -> {value_final.item():.4f}")
        print(f"   将值压缩到 (-1, 1) 范围")
        
        # 解释结果
        value_score = value_final.item()
        print(f"\n" + "="*70)
        print(f"📊 最终评估结果: {value_score:+.4f}")
        print("="*70)
        
        if value_score > 0.5:
            evaluation = "黑棋大优 🎯"
        elif value_score > 0.2:
            evaluation = "黑棋小优 ↗️"
        elif value_score > -0.2:
            evaluation = "局面均势 ⚖️"
        elif value_score > -0.5:
            evaluation = "黑棋小劣 ↘️"
        else:
            evaluation = "黑棋大劣 ⚠️"
        
        print(f"局面评估: {evaluation}")
        print(f"  +1.0 = 黑棋必胜")
        print(f"  {value_score:+.4f} = 当前评估 ← 这里")
        print(f"  -1.0 = 黑棋必败")

# 总结
print("\n" + "="*70)
print("【价值头工作原理总结】")
print("="*70)

print("""
价值头的任务：评估当前局面对当前玩家的胜率

工作流程：
  128维深度特征 (1, 128, 15, 15)
      ↓ [1x1卷积降维]
  2维紧凑特征 (1, 2, 15, 15)
      ↓ [展平]
  450维向量
      ↓ [全连接1: 450→256]
  256维抽象特征
      ↓ [全连接2: 256→1]
  1个得分值
      ↓ [Tanh激活]
  范围 (-1, +1) 的评估值

关键设计：
  1. 1x1卷积快速降维（128→2），聚焦核心信息
  2. 两层全连接逐步抽象（450→256→1）
  3. Tanh激活保证输出在[-1, 1]范围
     +1 = 当前玩家胜
      0 = 平局/均势
     -1 = 当前玩家败

训练方式：
  - 对局结束后，用真实胜负（+1/0/-1）作为标签
  - 价值头学习预测最终结果
  - 通过均方误差（MSE）训练

与MCTS配合：
  - 价值头评估叶子节点的价值
  - MCTS利用这个评估进行搜索剪枝
  - 训练时用实际对局结果监督价值头
""")

print("="*70)
print("✅ 价值头演示完成")
print("="*70)
print("\n注意: 由于网络未训练，当前评估值接近随机，训练后会更准确")
