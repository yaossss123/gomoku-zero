"""
全局配置参数
"""

# ================== MCTS参数 ==================
MCTS_SIMULATIONS_TRAIN = 600      # 训练时的模拟次数
MCTS_SIMULATIONS_EVAL = 1600       # 评估时的模拟次数
MCTS_SIMULATIONS_PLAY = 1600       # 人机对战时的模拟次数
MCTS_C_PUCT = 1.5                 # 探索常数

# ================== 网络参数 ==================
BOARD_SIZE = 15
NUM_RES_BLOCKS = 10
NUM_CHANNELS = 128
LEARNING_RATE = 0.002
L2_WEIGHT_DECAY = 1e-4

# ================== 训练参数 ==================
N_GAMES = 5000                     # 总训练局数（继续训练到5000局）
BATCH_SIZE = 256
BUFFER_SIZE = 100000
EPOCHS_PER_UPDATE = 5
SAVE_INTERVAL = 200
EVAL_INTERVAL = 200
EVAL_GAMES = 10                    # 评估局数
LOG_INTERVAL = 10

LR_MILESTONES = [3600, 4200, 4600]  # 学习率衰减点（从3000局继续到5000局）
LR_DECAY = 0.3                      # 衰减因子（温和）

# ================== MCTS探索参数 ==================
DIRICHLET_ALPHA = 0.3
DIRICHLET_EPSILON = 0.15

# ================== 温度参数 ==================
def get_temperature(move_count, game_progress):
    """动态温度策略"""
    if move_count < 30:
        base_temp = 1.0
    elif move_count < 60:
        base_temp = 0.5
    else:
        base_temp = 0.25
    
    # 训练后期降低随机性
    if game_progress < 0.3:
        decay = 1.0
    elif game_progress < 0.6:
        decay = 0.8
    elif game_progress < 0.8:
        decay = 0.5
    else:
        decay = 0.25
    
    return base_temp * decay

# ================== 稠密奖励开关 ==================
USE_DENSE_REWARD = False  # 关闭稠密奖励，使用纯终局奖励
DENSE_REWARD_DECAY = lambda progress: max(0.0, 0.5 * (1.0 - progress * 1.25))
