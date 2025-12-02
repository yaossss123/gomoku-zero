"""测试15个残差块的显存占用"""
import torch
import numpy as np
from improved_policy_value_net import ImprovedPolicyValueAgent

print('='*70)
print('测试15个残差块配置')
print('='*70)

# 创建15残差块的智能体
agent = ImprovedPolicyValueAgent(num_res_blocks=15)

# 重置显存统计
torch.cuda.reset_peak_memory_stats()

# 准备测试数据
batch_size = 512
state_batch = np.random.randn(batch_size, 15, 15, 3).astype(np.float32)
mcts_probs_batch = np.random.rand(batch_size, 225).astype(np.float32)
mcts_probs_batch = mcts_probs_batch / mcts_probs_batch.sum(axis=1, keepdims=True)
winner_batch = np.random.choice([-1, 0, 1], size=batch_size).astype(np.float32)

# 执行训练步骤
loss, _, _ = agent.train_step(state_batch, mcts_probs_batch, winner_batch)

# 获取显存占用
peak_memory = torch.cuda.max_memory_allocated() / 1024**3

print(f'\n配置: batch_size={batch_size}, 15个残差块')
print(f'  峰值显存: {peak_memory:.2f} GB')
print(f'  训练Loss: {loss:.4f}')

if peak_memory > 6.0:
    print(f'  ⚠️  警告: 显存超过6GB限制!')
    print(f'  建议: 降低batch_size或减少残差块数量')
else:
    print(f'  ✅ 显存占用安全 (< 6GB)')
    print(f'  显存余量: {6.0 - peak_memory:.2f} GB')
