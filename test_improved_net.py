"""
测试改进的网络
验证：前向传播、显存占用、训练步骤
"""
import numpy as np
import torch
from improved_policy_value_net import ImprovedPolicyValueAgent

print("="*60)
print("测试改进的策略价值网络")
print("="*60)

# 创建智能体
print("\n1. 初始化网络...")
agent = ImprovedPolicyValueAgent(
    board_size=15,
    num_res_blocks=10,
    learning_rate=0.002
)

# 测试前向传播
print("\n2. 测试前向传播...")
try:
    # 创建一个随机状态
    state = np.random.randn(15, 15, 3).astype(np.float32)
    action_probs, value = agent.policy_value_fn(state)
    
    print(f"   ✓ 动作概率形状: {action_probs.shape}")
    print(f"   ✓ 概率和: {action_probs.sum():.4f} (应该接近1.0)")
    print(f"   ✓ 价值评估: {value:.4f} (范围 -1 到 1)")
    print("   ✓ 前向传播成功")
except Exception as e:
    print(f"   ✗ 前向传播失败: {e}")
    exit(1)

# 测试显存占用和训练步骤
print("\n3. 测试显存占用和训练步骤...")

# 模拟一个训练批次
batch_size = 256
state_batch = np.random.randn(batch_size, 15, 15, 3).astype(np.float32)
mcts_probs_batch = np.random.rand(batch_size, 225).astype(np.float32)
mcts_probs_batch = mcts_probs_batch / mcts_probs_batch.sum(axis=1, keepdims=True)
winner_batch = np.random.choice([-1, 0, 1], size=batch_size).astype(np.float32)

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

try:
    loss, policy_loss, value_loss = agent.train_step(
        state_batch, mcts_probs_batch, winner_batch
    )
    
    print(f"   ✓ 总损失: {loss:.4f}")
    print(f"   ✓ 策略损失: {policy_loss:.4f}")
    print(f"   ✓ 价值损失: {value_loss:.4f}")
    
    if torch.cuda.is_available():
        # 获取显存使用
        memory_allocated = torch.cuda.memory_allocated() / 1024**3  # GB
        memory_reserved = torch.cuda.memory_reserved() / 1024**3    # GB
        max_memory = torch.cuda.max_memory_allocated() / 1024**3   # GB
        
        print(f"   ✓ 当前显存占用: {memory_allocated:.2f} GB")
        print(f"   ✓ 保留显存: {memory_reserved:.2f} GB")
        print(f"   ✓ 峰值显存: {max_memory:.2f} GB")
        
        if max_memory > 6.0:
            print(f"   ⚠️  警告: 显存占用({max_memory:.2f}GB)超过6GB限制！")
        else:
            print(f"   ✓ 显存占用在安全范围内 (< 6GB)")
    else:
        print("   ⚠️  未检测到CUDA设备，跳过显存测试")
            
    print("   ✓ 训练步骤成功")
    
except RuntimeError as e:
    if "out of memory" in str(e):
        print(f"   ✗ 显存不足！错误: {e}")
        print("   建议: 减小batch_size或残差块数量")
        exit(1)
    else:
        raise
except Exception as e:
    print(f"   ✗ 训练失败: {e}")
    exit(1)

# 测试模型保存和加载
print("\n5. 测试模型保存/加载...")
try:
    import os
    test_dir = "test_models"
    os.makedirs(test_dir, exist_ok=True)
    test_path = os.path.join(test_dir, "test_model.pth")
    
    # 保存
    agent.save_model(test_path)
    print(f"   ✓ 模型保存成功: {test_path}")
    
    # 加载
    agent.load_model(test_path)
    print(f"   ✓ 模型加载成功")
    
    # 清理测试文件
    os.remove(test_path)
    os.rmdir(test_dir)
    print(f"   ✓ 测试文件已清理")
    
except Exception as e:
    print(f"   ✗ 保存/加载失败: {e}")
    exit(1)

print("\n" + "="*60)
print("✓ 所有测试通过！网络可以正常使用")
print("="*60)
print("\n你可以开始训练了：")
print("python train_with_improved_net.py")
