"""
使用规则AI作为对手训练AlphaZero
通过与规则AI对弈学习人类棋感
"""
import numpy as np
import os
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt

from gomoku_env import GomokuEnv
from policy_value_net import PolicyValueAgent
from rule_based_ai import RuleBasedAI
from mcts import MCTS


class AlphaZeroWithRuleAITrainer:
    def __init__(
        self,
        board_size: int = 15,
        n_simulations: int = 400,
        c_puct: float = 5.0,
        buffer_size: int = 10000,
        batch_size: int = 512,
        epochs: int = 5
    ):
        self.board_size = board_size
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.epochs = epochs
        
        # 环境和智能体
        self.env = GomokuEnv(board_size=board_size)
        self.agent = PolicyValueAgent(board_size=board_size)
        self.rule_ai = RuleBasedAI(board_size=board_size)
        
        # 数据缓冲区
        self.data_buffer = deque(maxlen=buffer_size)
        
        # 训练统计
        self.episode_lengths = []
        self.losses = []
        self.win_rates = []  # 对规则AI的胜率
    
    def play_against_rule_ai(self, alphazero_first: bool = True):
        """
        AlphaZero对战规则AI
        Args:
            alphazero_first: AlphaZero是否先手
        Returns:
            play_data: 对局数据
            winner: 胜者 (1=先手胜, -1=后手胜, 0=平局)
        """
        play_data = []
        state = self.env.reset()
        
        # AlphaZero颜色（1=黑棋先手, -1=白棋后手）
        alphazero_color = 1 if alphazero_first else -1
        
        # 创建MCTS
        mcts = MCTS(self.agent.policy_value_fn, c_puct=self.c_puct, n_simulations=self.n_simulations)
        
        step = 0
        max_steps = 225  # 最大步数限制
        
        while step < max_steps:
            current_player = self.env.current_player
            
            if current_player == alphazero_color:
                # AlphaZero下棋
                try:
                    action, action_probs = mcts.get_action(self.env, temp=1.0)
                    # 保存数据
                    play_data.append((state.copy(), action_probs, current_player))
                except Exception as e:
                    print(f"错误: AlphaZero下棋失败: {e}")
                    break
            else:
                # 规则AI下棋
                try:
                    action = self.rule_ai.get_action(self.env)
                except Exception as e:
                    print(f"错误: 规则AI下棋失败: {e}")
                    break
            
            # 执行动作
            state, _, done, info = self.env.step(int(action))
            
            # 更新MCTS树
            if current_player == alphazero_color:
                mcts.update_with_move(action)
            
            step += 1
            
            if done:
                winner = info.get('winner', 0)
                
                # 转换数据：从AlphaZero视角分配奖励
                play_data_with_reward = []
                for s, probs, player in play_data:
                    if winner == 0:
                        reward = 0.0
                    elif winner == alphazero_color:
                        reward = 1.0  # AlphaZero赢
                    else:
                        reward = -1.0  # 规则AI赢
                    play_data_with_reward.append((s, probs, reward))
                
                self.episode_lengths.append(step)
                return play_data_with_reward, winner
        
        # 超过最大步数，判定为平局
        play_data_with_reward = [(s, probs, 0.0) for s, probs, player in play_data]
        self.episode_lengths.append(step)
        return play_data_with_reward, 0
    
    def augment_data(self, play_data):
        """简化的数据增强"""
        augmented_data = []
        
        for state, probs, reward in play_data:
            if state.shape != (self.board_size, self.board_size, 3):
                continue
            
            # 原始数据
            augmented_data.append((state.copy(), probs.copy(), reward))
            
            # 旋转180度
            state_rot180 = np.rot90(state, k=2, axes=(0, 1))
            probs_rot180 = self._rotate_probs(probs, k=2)
            augmented_data.append((state_rot180.copy(), probs_rot180.copy(), reward))
            
            # 水平翻转
            state_flip = np.flip(state, axis=1)
            probs_flip = self._flip_probs(probs)
            augmented_data.append((state_flip.copy(), probs_flip.copy(), reward))
        
        return augmented_data
    
    def _rotate_probs(self, probs, k):
        """旋转概率分布"""
        probs_2d = probs.reshape(self.board_size, self.board_size)
        probs_rot = np.rot90(probs_2d, k=k)
        return probs_rot.flatten()
    
    def _flip_probs(self, probs):
        """翻转概率分布"""
        probs_2d = probs.reshape(self.board_size, self.board_size)
        probs_flip = np.fliplr(probs_2d)
        return probs_flip.flatten()
    
    def policy_update(self):
        """更新策略价值网络"""
        if len(self.data_buffer) < self.batch_size:
            return 0.0, 0.0, 0.0
        
        mini_batch = np.random.choice(len(self.data_buffer), size=self.batch_size, replace=False)
        state_list = [self.data_buffer[i][0] for i in mini_batch]
        mcts_probs_list = [self.data_buffer[i][1] for i in mini_batch]
        winner_list = [self.data_buffer[i][2] for i in mini_batch]
        
        state_batch = np.array(state_list)
        mcts_probs_batch = np.array(mcts_probs_list)
        winner_batch = np.array(winner_list)
        
        # 训练多个epoch
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        for _ in range(self.epochs):
            loss, policy_loss, value_loss = self.agent.train_step(
                state_batch, mcts_probs_batch, winner_batch
            )
            total_loss += loss
            total_policy_loss += policy_loss
            total_value_loss += value_loss
            self.losses.append(loss)
        
        return total_loss / self.epochs, total_policy_loss / self.epochs, total_value_loss / self.epochs
    
    def train(self, n_games: int = 1000, save_interval: int = 100, model_dir: str = 'models_alphazero', resume_from: str = None):
        """
        训练主循环
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # 加载检查点
        start_episode = 0
        if resume_from and os.path.exists(resume_from):
            print(f"正在加载检查点: {resume_from}")
            self.agent.load_model(resume_from)
            import re
            match = re.search(r'model_(\d+)\.pth', resume_from)
            if match:
                start_episode = int(match.group(1))
                print(f"从第 {start_episode} 局继续训练")
        
        print("="*60)
        print("AlphaZero vs 规则AI 训练")
        print(f"开始局数: {start_episode}")
        print(f"目标局数: {n_games}")
        print(f"MCTS模拟次数: {self.n_simulations}")
        print("="*60)
        
        wins = 0
        losses = 0
        draws = 0
        
        for i in tqdm(range(start_episode, n_games)):
            # 交替先后手
            alphazero_first = (i % 2 == 0)
            
            # 对战规则AI
            play_data, winner = self.play_against_rule_ai(alphazero_first)
            
            # 统计胜率
            alphazero_color = 1 if alphazero_first else -1
            if winner == alphazero_color:
                wins += 1
            elif winner == 0:
                draws += 1
            else:
                losses += 1
            
            # 数据增强
            augmented_data = self.augment_data(play_data)
            self.data_buffer.extend(augmented_data)
            
            # 训练更新
            if len(self.data_buffer) >= self.batch_size:
                loss, policy_loss, value_loss = self.policy_update()
                
                if (i + 1) % 10 == 0:
                    total_games = wins + losses + draws
                    win_rate = wins / total_games if total_games > 0 else 0
                    self.win_rates.append(win_rate)
                    
                    avg_length = np.mean(self.episode_lengths[-10:])
                    print(f"\n第 {i+1} 局: 胜率={win_rate:.2%} (胜{wins}/平{draws}/负{losses}), "
                          f"平均步数={avg_length:.1f}, Loss={loss:.4f}")
            
            # 保存模型
            if (i + 1) % save_interval == 0:
                model_path = os.path.join(model_dir, f'alphazero_rule_trained_{i+1}.pth')
                self.agent.save_model(model_path)
        
        # 保存最终模型
        final_path = os.path.join(model_dir, 'alphazero_rule_trained_final.pth')
        self.agent.save_model(final_path)
        
        print(f"\n训练完成！最终胜率: {wins/(wins+losses+draws):.2%}")


if __name__ == '__main__':
    trainer = AlphaZeroWithRuleAITrainer(
        board_size=15,
        n_simulations=400,
        c_puct=5.0,
        buffer_size=10000,
        batch_size=512,
        epochs=5
    )
    
    # 从最新模型继续训练，大幅增加训练局数
    trainer.train(
        n_games=18000,  # 再训练4000局（从14000到18000）
        save_interval=500,
        resume_from='models_alphazero/alphazero_rule_trained_final.pth'
    )
