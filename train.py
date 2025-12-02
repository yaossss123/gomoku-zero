"""
训练脚本 - 通过自我对弈训练五子棋AI
"""
import numpy as np
from gomoku_env import GomokuEnv
from dqn_agent import DQNAgent
import matplotlib.pyplot as plt
from tqdm import tqdm
import os


def self_play_episode(env, agent1, agent2, training=True):
    """
    自我对弈一局游戏
    Args:
        env: 游戏环境
        agent1: 黑棋智能体
        agent2: 白棋智能体
        training: 是否为训练模式
    Returns:
        episode_data: 对局数据
        winner: 获胜者
    """
    state = env.reset()
    done = False
    episode_data = []
    
    current_agent = agent1
    
    while not done:
        # 获取合法动作
        valid_actions = env.get_valid_actions()
        
        # 选择动作
        action = current_agent.select_action(state, valid_actions, training)
        
        # 执行动作
        next_state, reward, done, info = env.step(action)
        
        # 保存经验（从当前玩家视角）
        episode_data.append({
            'state': state,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'player': env.current_player
        })
        
        state = next_state
        
        # 切换智能体
        current_agent = agent2 if current_agent == agent1 else agent1
    
    winner = info.get('winner', 0)
    return episode_data, winner


def process_episode_data(episode_data, winner):
    """
    处理对局数据，分配奖励
    Args:
        episode_data: 对局数据
        winner: 获胜者 (1: 黑棋, -1: 白棋, 0: 平局)
    Returns:
        processed_data: 处理后的数据
    """
    processed_data_black = []
    processed_data_white = []
    
    for i, data in enumerate(episode_data):
        state = data['state']
        action = data['action']
        next_state = data['next_state']
        done = data['done']
        
        # 根据回合和最终结果分配奖励
        if done:
            # 最后一步的玩家
            last_player = -data['player']  # 因为step后会切换
            if winner == last_player:
                reward = 1.0  # 获胜
            elif winner == 0:
                reward = 0.0  # 平局
            else:
                reward = -1.0  # 失败
        else:
            reward = 0.0  # 中间步骤无奖励
        
        # 判断该步骤是哪个玩家的
        if i % 2 == 0:  # 黑棋
            processed_data_black.append((state, action, reward, next_state, done))
        else:  # 白棋
            processed_data_white.append((state, action, reward, next_state, done))
    
    return processed_data_black, processed_data_white


def train(
    num_episodes=5000,
    board_size=15,
    save_interval=500,
    model_dir='models'
):
    """
    训练智能体
    Args:
        num_episodes: 训练局数
        board_size: 棋盘大小
        save_interval: 保存间隔
        model_dir: 模型保存目录
    """
    # 创建模型目录
    os.makedirs(model_dir, exist_ok=True)
    
    # 创建环境
    env = GomokuEnv(board_size=board_size)
    
    # 创建智能体（两个智能体共享经验池）
    agent = DQNAgent(board_size=board_size)
    
    # 训练统计
    wins_black = 0
    wins_white = 0
    draws = 0
    episode_losses = []
    win_rates = []
    
    print(f"开始训练，共 {num_episodes} 局")
    
    for episode in tqdm(range(num_episodes)):
        # 自我对弈
        episode_data, winner = self_play_episode(env, agent, agent, training=True)
        
        # 统计胜负
        if winner == 1:
            wins_black += 1
        elif winner == -1:
            wins_white += 1
        else:
            draws += 1
        
        # 处理对局数据并添加到经验池
        data_black, data_white = process_episode_data(episode_data, winner)
        
        for state, action, reward, next_state, done in data_black + data_white:
            agent.replay_buffer.push(state, action, reward, next_state, done)
        
        # 训练
        loss = agent.train_step()
        if loss is not None:
            episode_losses.append(loss)
        
        # 更新epsilon
        agent.update_epsilon()
        
        # 每100局统计一次
        if (episode + 1) % 100 == 0:
            total = wins_black + wins_white + draws
            win_rate_black = wins_black / total if total > 0 else 0
            win_rates.append(win_rate_black)
            
            avg_loss = np.mean(episode_losses[-100:]) if episode_losses else 0
            
            print(f"\n第 {episode + 1} 局:")
            print(f"  黑棋胜率: {win_rate_black:.2%}")
            print(f"  白棋胜率: {wins_white/total:.2%}")
            print(f"  平局率: {draws/total:.2%}")
            print(f"  平均损失: {avg_loss:.4f}")
            print(f"  Epsilon: {agent.epsilon:.4f}")
            
            # 重置统计
            wins_black = wins_white = draws = 0
        
        # 保存模型
        if (episode + 1) % save_interval == 0:
            model_path = os.path.join(model_dir, f'gomoku_dqn_episode_{episode + 1}.pth')
            agent.save(model_path)
    
    # 保存最终模型
    final_model_path = os.path.join(model_dir, 'gomoku_dqn_final.pth')
    agent.save(final_model_path)
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(episode_losses)
    plt.title('Training Loss')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(win_rates)
    plt.title('Win Rate (Black)')
    plt.xlabel('Episodes (x100)')
    plt.ylabel('Win Rate')
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_curves.png'))
    print(f"\n训练完成！曲线已保存到 {model_dir}/training_curves.png")


if __name__ == '__main__':
    train(num_episodes=5000, board_size=15)
