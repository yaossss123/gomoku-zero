"""
策略头运行演示
详细展示策略头如何从特征图生成动作概率
"""
import numpy as np
import torch
import torch.nn.functional as F
from improved_policy_value_net import ImprovedPolicyValueAgent

print("="*70)
print("策略头运行演示")
print("="*70)

# 创建智能体
agent = ImprovedPolicyValueAgent(board_size=15, num_res_blocks=10)
net = agent.policy_value_net

# 创建一个测试状态
print("\n【步骤1】输入状态")
state = np.zeros((15, 15, 3), dtype=np.float32)
# 模拟一个简单局面：中间有几颗棋子
state[7, 7, 0] = 1  # 黑棋
state[7, 8, 0] = 1  # 黑棋
state[8, 7, 1] = 1  # 白棋
state[:, :, 2] = 1  # 当前玩家为黑棋

print(f"输入形状: (15, 15, 3)")
print(f"黑棋位置: (7,7), (7,8)")
print(f"白棋位置: (8,7)")

# 转换为张量
state_tensor = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(agent.device)
print(f"张量形状: {state_tensor.shape} -> (batch=1, channels=3, H=15, W=15)")

# 通过网络
net.eval()
with torch.no_grad():
    print("\n" + "="*70)
    print("【步骤2】共享特征提取（初始卷积 + 10个残差块）")
    print("="*70)
    
    # 初始卷积
    x = net.conv_init(state_tensor)
    x = net.bn_init(x)
    x = F.relu(x)
    print(f"初始卷积后: {x.shape} -> (1, 128, 15, 15)")
    print(f"  - 3通道扩展到128通道")
    print(f"  - 棋盘尺寸保持15x15")
    
    # 通过残差块
    for i, res_block in enumerate(net.res_blocks):
        x_before = x.clone()
        x = res_block(x)
        if i < 2 or i == 9:  # 只显示前2个和最后1个
            diff = torch.abs(x - x_before).mean().item()
            print(f"残差块{i+1}: 特征微调 (平均变化={diff:.4f})")
        elif i == 2:
            print(f"  ... (中间{len(net.res_blocks)-3}个残差块)")
    
    print(f"残差块处理后: {x.shape} -> 仍然是 (1, 128, 15, 15)")
    
    print("\n" + "="*70)
    print("【步骤3】策略头处理")
    print("="*70)
    
    # 策略头卷积
    policy = net.policy_conv(x)
    print(f"1️⃣  卷积降维: {x.shape} -> {policy.shape}")
    print(f"   Conv2d(128 -> 4, kernel_size=1)")
    print(f"   作用: 将128维特征压缩到4维，减少计算量")
    
    policy = net.policy_bn(policy)
    policy = F.relu(policy)
    print(f"   BatchNorm + ReLU: 归一化并激活")
    
    # 展平
    policy_flat = policy.reshape(policy.size(0), -1)
    print(f"\n2️⃣  展平操作: {policy.shape} -> {policy_flat.shape}")
    print(f"   (1, 4, 15, 15) -> (1, 900)")
    print(f"   将4个通道的15x15特征图拉成一维向量")
    
    # 全连接层
    fc_output = net.policy_fc(policy_flat)
    print(f"\n3️⃣  全连接层: {policy_flat.shape} -> {fc_output.shape}")
    print(f"   Linear(900 -> 225)")
    print(f"   作用: 将特征映射到225个位置的得分")
    
    # Log Softmax
    log_action_probs = F.log_softmax(fc_output, dim=1)
    action_probs = torch.exp(log_action_probs)
    
    print(f"\n4️⃣  Softmax归一化: {fc_output.shape} -> {action_probs.shape}")
    print(f"   将得分转换为概率分布")
    print(f"   概率和: {action_probs.sum().item():.6f} (应该=1.0)")
    
    print("\n" + "="*70)
    print("【步骤4】策略头输出分析")
    print("="*70)
    
    # 转换为numpy
    probs_np = action_probs.cpu().numpy()[0]
    probs_2d = probs_np.reshape(15, 15)
    
    # 找出概率最高的5个位置
    top_5_indices = np.argsort(probs_np)[-5:][::-1]
    
    print("\n✨ 概率最高的5个落子位置：")
    for rank, idx in enumerate(top_5_indices, 1):
        row, col = idx // 15, idx % 15
        prob = probs_np[idx]
        print(f"  {rank}. 位置({row:2d}, {col:2d}) - 概率: {prob:.6f} ({prob*100:.3f}%)")
    
    # 显示周围区域的概率分布
    print("\n📊 中心区域(6:10, 6:10)概率分布热力图:")
    center_probs = probs_2d[6:10, 6:10]
    print("     ", end="")
    for c in range(6, 10):
        print(f" {c:2d}   ", end="")
    print()
    for r in range(6, 10):
        print(f"  {r:2d} ", end="")
        for c in range(6, 10):
            prob = probs_2d[r, c]
            # 用星号表示概率高低
            if prob > 0.01:
                stars = "★★★"
            elif prob > 0.005:
                stars = "★★ "
            elif prob > 0.001:
                stars = "★  "
            else:
                stars = "   "
            print(f"{stars}", end=" ")
        print()
    
    print("\n" + "="*70)
    print("【步骤5】策略头工作原理总结")
    print("="*70)
    
    print("""
策略头的任务：从棋盘特征生成每个位置的落子概率

工作流程：
  128维深度特征 
      ↓ [1x1卷积降维]
  4维紧凑特征
      ↓ [展平]
  900维向量
      ↓ [全连接映射]
  225个位置得分
      ↓ [Softmax归一化]
  225个位置概率（总和=1）

关键设计：
  1. 1x1卷积快速降维（128→4），减少参数量
  2. 全连接层学习位置之间的关系
  3. Softmax保证输出是合法的概率分布

与MCTS配合：
  - 网络输出作为MCTS的"先验概率"
  - MCTS通过搜索改进这个概率
  - 训练时用MCTS改进后的概率监督网络
    """)

print("="*70)
print("✅ 策略头演示完成")
print("="*70)
