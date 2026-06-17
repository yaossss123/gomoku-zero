"""
AlphaZero训练脚本 
"""
import numpy as np
import os
import time
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt

from gomoku_env import GomokuEnv
from improved_policy_value_net import ImprovedPolicyValueAgent
from mcts import MCTS
from rule_based_ai import RuleBasedAI
from config import *


class AlphaZeroTrainer:
    """AlphaZero训练器 """
    
    def __init__(
        self,
        board_size: int = BOARD_SIZE,
        num_res_blocks: int = NUM_RES_BLOCKS,
        num_channels: int = NUM_CHANNELS,
        n_simulations: int = MCTS_SIMULATIONS_TRAIN,
        c_puct: float = MCTS_C_PUCT,
        buffer_size: int = BUFFER_SIZE,
        batch_size: int = BATCH_SIZE,
        epochs_per_update: int = EPOCHS_PER_UPDATE,
        learning_rate: float = LEARNING_RATE,
        lr_milestones: list = None,
        dirichlet_alpha: float = DIRICHLET_ALPHA,
        dirichlet_epsilon: float = DIRICHLET_EPSILON
    ):
        """初始化训练器"""
        self.board_size = board_size
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.epochs_per_update = epochs_per_update
        self.learning_rate = learning_rate
        self.lr_milestones = lr_milestones or LR_MILESTONES
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        
        self.env = GomokuEnv(board_size=board_size)
        
        self.agent = ImprovedPolicyValueAgent(
            board_size=board_size,
            num_res_blocks=num_res_blocks,
            num_channels=num_channels,
            learning_rate=learning_rate,
            force_gpu=True  # 强制使用GPU
        )
        self.num_res_blocks = num_res_blocks
        
        self.data_buffer = deque(maxlen=buffer_size)
        
        self.stats = {
            'episode_lengths': [],
            'losses': [],
            'policy_losses': [],
            'value_losses': [],
            'win_rates': [],
            'game_results': [],
        }
        
        self.rule_ai = RuleBasedAI(board_size=board_size)
        
        print("=" * 60)
        print("AlphaZero训练器初始化完成")
        print(f"  棋盘大小: {board_size}x{board_size}")
        print(f"  残差块数: {num_res_blocks}")
        print(f"  MCTS模拟: {n_simulations}")
        print(f"  c_puct: {c_puct}")
        print(f"  缓冲区: {buffer_size}")
        print(f"  批大小: {batch_size}")
        print("=" * 60)
    
    def get_temperature(self, move_count: int, game_idx: int, total_games: int) -> float:
        """动态温度策略"""
        if move_count < 30:
            base_temp = 1.0
        elif move_count < 60:
            base_temp = 0.5
        else:
            base_temp = 0.25
        
        progress = game_idx / max(total_games, 1)
        if progress < 0.3:
            decay = 1.0
        elif progress < 0.6:
            decay = 0.8
        elif progress < 0.8:
            decay = 0.5
        else:
            decay = 0.25
        
        return base_temp * decay
    
    def adjust_learning_rate(self, game_idx: int):
        """学习率衰减"""
        lr = self.learning_rate
        for milestone in self.lr_milestones:
            if game_idx >= milestone:
                lr *= LR_DECAY
        
        for param_group in self.agent.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
    
    def self_play_game(self, game_idx: int, total_games: int):
        """执行一局自我对弈
        融合稠密奖励，加速早期学习
        """
        self.env.reset()
        
        mcts = MCTS(
            policy_value_fn=self.agent.policy_value_fn,
            c_puct=self.c_puct,
            n_simulations=self.n_simulations,
            dirichlet_alpha=self.dirichlet_alpha,
            dirichlet_epsilon=self.dirichlet_epsilon
        )
        
        play_data = []
        step = 0
        
        while not self.env.done:
            state = self.env._get_observation()
            temp = self.get_temperature(step, game_idx, total_games)
            
            action, action_probs = mcts.get_action(
                self.env, 
                temp=temp, 
                add_noise=True
            )
            
            play_data.append({
                'state': state.copy(),
                'action_probs': action_probs.copy(),
                'player': self.env.current_player,
            })
            
            # 只在配置启用时才捕获稠密奖励
            if USE_DENSE_REWARD:
                _, dense_reward, _, info = self.env.step(int(action))
                play_data[-1]['dense_reward'] = dense_reward
            else:
                self.env.step(int(action), skip_dense=True)
            
            mcts.update_with_move(action)
            step += 1
        
        winner = self.env.winner
        
        # 生成训练数据
        training_data = []
        
        # 计算终局奖励基础值
        final_rewards = []
        for data in play_data:
            if winner == 0:
                final_rewards.append(0.0)
            elif winner == data['player']:
                final_rewards.append(1.0)
            else:
                final_rewards.append(-1.0)
        
        # 使用TD回报计算稠密奖励（从terminal向前累积）
        if USE_DENSE_REWARD:
            progress = game_idx / max(total_games, 1)
            dense_weight = DENSE_REWARD_DECAY(progress)
            gamma = 0.99  # 折扣因子
            
            # ===== 正确的MC回报计算（考虑双方交替）=====
            n = len(play_data)
            
            if n == 0:
                return [], winner, step
            
            td_values = [0.0] * n
            
            # 最后一步：回报 = 当步稠密奖励 + 终局奖励
            last_dense = play_data[n-1].get('dense_reward', 0.0) * dense_weight
            td_values[n-1] = last_dense + final_rewards[n-1]
            
            # 逆序累积：每步TD值 = 当步稠密奖励 + gamma * (对手下一步TD值取负)
            for i in range(n-2, -1, -1):
                dense_r = play_data[i].get('dense_reward', 0.0) * dense_weight
                # 关键：下一步是对手视角，所以取负
                td_values[i] = dense_r + gamma * (-td_values[i+1])
            
            # 混合策略：终局奖励为主(70%)，TD回报为辅(30%)
            alpha = 0.3
            
            for i in range(n):
                value_target = (1 - alpha) * final_rewards[i] + alpha * td_values[i]
                value_target = max(-1.0, min(1.0, value_target))  # 裁剪
                
                training_data.append((
                    play_data[i]['state'],
                    play_data[i]['action_probs'],
                    value_target
                ))
        else:
            # 不使用稠密奖励，纯终局奖励
            for i in range(len(play_data)):
                training_data.append((
                    play_data[i]['state'],
                    play_data[i]['action_probs'],
                    final_rewards[i]
                ))
        
        return training_data, winner, step
    
    def augment_data(self, play_data: list) -> list:
        """
        数据增强：利用棋盘的8倍对称性
        修复：确保数据连续性
        """
        augmented = []
        
        for state, probs, reward in play_data:
            if state.shape != (self.board_size, self.board_size, 3):
                continue
            
            probs_2d = probs.reshape(self.board_size, self.board_size)
            
            for k in range(4):
                # 旋转
                state_rot = np.rot90(state, k=k, axes=(0, 1))
                probs_rot = np.rot90(probs_2d, k=k)
                
                # 使用 np.ascontiguousarray 确保内存连续
                augmented.append((
                    np.ascontiguousarray(state_rot),
                    np.ascontiguousarray(probs_rot.flatten()),
                    reward
                ))
                
                # 水平翻转
                state_flip = np.flip(state_rot, axis=1)
                probs_flip = np.fliplr(probs_rot)
                
                augmented.append((
                    np.ascontiguousarray(state_flip),
                    np.ascontiguousarray(probs_flip.flatten()),
                    reward
                ))
        
        return augmented
    
    def train_step(self) -> dict:
        """执行一次网络更新"""
        if len(self.data_buffer) < self.batch_size:
            return None
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        for _ in range(self.epochs_per_update):
            indices = np.random.choice(
                len(self.data_buffer), 
                size=self.batch_size, 
                replace=False
            )
            
            state_batch = np.array([self.data_buffer[i][0] for i in indices])
            probs_batch = np.array([self.data_buffer[i][1] for i in indices])
            reward_batch = np.array([self.data_buffer[i][2] for i in indices])
            
            loss, p_loss, v_loss = self.agent.train_step(
                state_batch, probs_batch, reward_batch
            )
            
            total_loss += loss
            total_policy_loss += p_loss
            total_value_loss += v_loss
        
        n = self.epochs_per_update
        return {
            'loss': total_loss / n,
            'policy_loss': total_policy_loss / n,
            'value_loss': total_value_loss / n
        }
    
    def evaluate(self, n_games: int = EVAL_GAMES) -> float:
        """与规则AI对战评估"""
        wins, losses, draws = 0, 0, 0
        
        for game_idx in range(n_games):
            env = GomokuEnv(board_size=self.board_size)
            env.reset()
            
            alphazero_player = 1 if game_idx % 2 == 0 else -1
            
            # 评估时用更少的模拟次数
            mcts = MCTS(
                policy_value_fn=self.agent.policy_value_fn,
                c_puct=self.c_puct,
                n_simulations=MCTS_SIMULATIONS_EVAL
            )
            
            while not env.done:
                if env.current_player == alphazero_player:
                    action, _ = mcts.get_action(env, temp=1e-3, add_noise=False)
                    mcts.update_with_move(action)
                else:
                    action = self.rule_ai.get_action(env)
                    mcts.update_with_move(action)
                
                env.step(int(action), skip_dense=True)  # 评估时跳过稠密奖励
            
            if env.winner == 0:
                draws += 1
            elif env.winner == alphazero_player:
                wins += 1
            else:
                losses += 1
        
        win_rate = wins / n_games
        print(f"  评估: 胜={wins}, 负={losses}, 平={draws}, 胜率={win_rate:.1%}")
        
        return win_rate
    
    def train(
        self,
        n_games: int = N_GAMES,
        save_interval: int = SAVE_INTERVAL,
        eval_interval: int = EVAL_INTERVAL,
        log_interval: int = LOG_INTERVAL,
        model_dir: str = 'models_dense_reward',
        resume_from: str = None
    ):
        """训练主循环"""
        os.makedirs(model_dir, exist_ok=True)
        
        start_game = 0
        if resume_from and os.path.exists(resume_from):
            print(f"加载检查点: {resume_from}")
            self.agent.load_model(resume_from)
            import re
            match = re.search(r'model_(\d+)\.pth', resume_from)
            if match:
                start_game = int(match.group(1))
            elif 'final' in resume_from:
                start_game = 3000  # model_final 是3000局训练的
        
        print("\n" + "=" * 60)
        print("开始训练")
        print(f"起始局数: {start_game}")
        print(f"目标局数: {n_games}")
        print("=" * 60 + "\n")
        
        start_time = time.time()
        
        for game_idx in tqdm(range(start_game, n_games), desc="训练进度"):
            # 1. 自我对弈
            game_data, winner, steps = self.self_play_game(game_idx, n_games)
            self.stats['episode_lengths'].append(steps)
            self.stats['game_results'].append(winner)
            
            # 2. 数据增强并加入缓冲区
            augmented = self.augment_data(game_data)
            self.data_buffer.extend(augmented)
            
            # 3. 调整学习率
            current_lr = self.adjust_learning_rate(game_idx)
            
            # 4. 网络更新
            train_stats = self.train_step()
            if train_stats:
                self.stats['losses'].append(train_stats['loss'])
                self.stats['policy_losses'].append(train_stats['policy_loss'])
                self.stats['value_losses'].append(train_stats['value_loss'])
            
            # 5. 日志输出
            if (game_idx + 1) % log_interval == 0:
                avg_len = np.mean(self.stats['episode_lengths'][-log_interval:])
                avg_loss = np.mean(self.stats['losses'][-log_interval*5:]) if self.stats['losses'] else 0
                avg_p = np.mean(self.stats['policy_losses'][-log_interval*5:]) if self.stats['policy_losses'] else 0
                avg_v = np.mean(self.stats['value_losses'][-log_interval*5:]) if self.stats['value_losses'] else 0
                
                elapsed = time.time() - start_time
                games_per_hour = (game_idx + 1 - start_game) / (elapsed / 3600) if elapsed > 0 else 0
                
                tqdm.write(
                    f"[{game_idx+1:5d}] "
                    f"步数={avg_len:5.1f} | "
                    f"Loss={avg_loss:.4f} (P={avg_p:.4f}, V={avg_v:.4f}) | "
                    f"LR={current_lr:.6f} | "
                    f"Buffer={len(self.data_buffer):6d} | "
                    f"速度={games_per_hour:.1f}局/时"
                )
            
            # 6. 定期评估
            if (game_idx + 1) % eval_interval == 0:
                tqdm.write(f"\n--- 第 {game_idx+1} 局评估 ---")
                win_rate = self.evaluate(n_games=EVAL_GAMES)
                self.stats['win_rates'].append((game_idx + 1, win_rate))
                tqdm.write("")
            
            # 7. 保存模型
            if (game_idx + 1) % save_interval == 0:
                path = os.path.join(model_dir, f'model_{game_idx+1}.pth')
                self.agent.save_model(path)
        
        final_path = os.path.join(model_dir, 'model_final.pth')
        self.agent.save_model(final_path)
        
        self.plot_training_curves(model_dir)
        
        total_time = time.time() - start_time
        print(f"\n训练完成！总耗时: {total_time/3600:.2f}小时")
    
    def plot_training_curves(self, save_dir: str):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        ax = axes[0, 0]
        if self.stats['losses']:
            ax.plot(self.stats['losses'], alpha=0.3, label='Loss')
            window = min(100, len(self.stats['losses']))
            if window > 1:
                ma = np.convolve(self.stats['losses'], np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(self.stats['losses'])), ma, label=f'MA-{window}')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Loss')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[0, 1]
        if self.stats['policy_losses']:
            ax.plot(self.stats['policy_losses'], alpha=0.3, color='blue', label='Policy')
            ax.plot(self.stats['value_losses'], alpha=0.3, color='red', label='Value')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('Loss')
        ax.set_title('Policy & Value Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 0]
        if self.stats['episode_lengths']:
            ax.plot(self.stats['episode_lengths'], alpha=0.3)
            window = min(50, len(self.stats['episode_lengths']))
            if window > 1:
                ma = np.convolve(self.stats['episode_lengths'], np.ones(window)/window, mode='valid')
                ax.plot(range(window-1, len(self.stats['episode_lengths'])), ma, label=f'MA-{window}')
        ax.set_xlabel('Games')
        ax.set_ylabel('Steps')
        ax.set_title('Episode Length')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes[1, 1]
        if self.stats['win_rates']:
            games, rates = zip(*self.stats['win_rates'])
            ax.plot(games, rates, 'bo-', markersize=8)
            ax.axhline(y=0.5, color='r', linestyle='--', alpha=0.5)
        ax.set_xlabel('Games')
        ax.set_ylabel('Win Rate vs Rule AI')
        ax.set_title('Evaluation Win Rate')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(save_dir, 'training_curves.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"训练曲线已保存: {save_path}")


if __name__ == '__main__':
    trainer = AlphaZeroTrainer(
        board_size=15,
        num_res_blocks=10,
        num_channels=128,
        n_simulations=400,
        batch_size=256,
        buffer_size=100000,
        epochs_per_update=5,
    )
    
    trainer.train(
        n_games=5000,  # 继续训练到5000局
        save_interval=500,
        eval_interval=100,
        log_interval=10,
        model_dir='models_standard',
        resume_from='models_standard/model_3200.pth'  # 从3200局继续训练
    )