"""
训练配置选项
"""

# 超快速测试（验证代码能跑）
FAST_TEST_CONFIG = {
    'board_size': 15,
    'num_res_blocks': 3,       # 最小网络
    'num_channels': 64,
    'n_simulations': 100,      # 最少模拟
    'batch_size': 64,
    'buffer_size': 5000,
    'epochs_per_update': 2,
    'n_games': 50,
    'eval_interval': 25,
    'save_interval': 25,
}

# 快速训练（几小时内看到效果）
FAST_TRAIN_CONFIG = {
    'board_size': 15,
    'num_res_blocks': 5,
    'num_channels': 128,
    'n_simulations': 200,
    'batch_size': 128,
    'buffer_size': 20000,
    'epochs_per_update': 3,
    'n_games': 500,
    'eval_interval': 50,
    'save_interval': 100,
}

# 标准训练（建议配置）
STANDARD_CONFIG = {
    'board_size': 15,
    'num_res_blocks': 10,
    'num_channels': 128,
    'n_simulations': 400,
    'batch_size': 256,
    'buffer_size': 100000,
    'epochs_per_update': 5,
    'n_games': 3000,
    'eval_interval': 100,
    'save_interval': 500,
}

# 高质量训练（需要很长时间）
HIGH_QUALITY_CONFIG = {
    'board_size': 15,
    'num_res_blocks': 15,
    'num_channels': 256,
    'n_simulations': 800,
    'batch_size': 512,
    'buffer_size': 200000,
    'epochs_per_update': 5,
    'n_games': 10000,
    'eval_interval': 200,
    'save_interval': 1000,
}