"""
使用改进网络训练AlphaZero（10个残差块）
与规则AI对战，学习人类棋感
"""
import numpy as np
import os
from tqdm import tqdm
from collections import deque
import matplotlib.pyplot as plt

from gomoku_env import GomokuEnv
from improved_policy_value_net import ImprovedPolicyValueAgent
from mcts import MCTS


class ImprovedAlphaZeroTrainer:
    def __init__(
        self,
        board_size: int = 15,
        num_res_blocks: int = 15,
        n_simulations: int = 400,
        c_puct: float = 5.0,
        buffer_size: int = 10000,
        batch_size: int = 512,  # 显存测试通过，提升到512以增强训练稳定性
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
        self.agent = ImprovedPolicyValueAgent(
            board_size=board_size,
            num_res_blocks=num_res_blocks
        )
        
        # 数据缓冲区
        self.data_buffer = deque(maxlen=buffer_size)
        
        # 训练统计
        self.episode_lengths = []
        self.losses = []
        self.win_rates = []
    
    def self_play(self, temp: float = 1.0):
        """
        自我对弈
        Args:
            temp: 温度参数
        Returns:
            play_data: 对局数据
        """
        play_data = []
        state = self.env.reset()
        
        # 创建MCTS
        mcts = MCTS(self.agent.policy_value_fn, c_puct=self.c_puct, n_simulations=self.n_simulations)
        
        step = 0
        
        while True:
            # 使用MCTS获取动作
            action, action_probs = mcts.get_action(self.env, temp=temp)
            
            # 保存数据
            play_data.append((state.copy(), action_probs, self.env.current_player))
            
            # 执行动作
            state, _, done, info = self.env.step(int(action))
            
            # 更新MCTS树
            mcts.update_with_move(action)
            
            step += 1
            
            if done:
                # 游戏结束，分配奖励
                winner = info.get('winner', 0)
                
                # 从玩家视角分配胜负
                play_data_with_reward = []
                for s, probs, player in play_data:
                    if winner == 0:
                        reward = 0.0
                    elif winner == player:
                        reward = 1.0
                    else:
                        reward = -1.0
                    play_data_with_reward.append((s, probs, reward))
                
                self.episode_lengths.append(step)
                return play_data_with_reward
    
    def augment_data(self, play_data):
        """数据增强"""
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
    
    def train(self, n_games: int = 5000, save_interval: int = 500, model_dir: str = 'models_improved_alphazero', resume_from: str | None = None):
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
        print("改进版AlphaZero 自我对弈训练")
        print(f"网络: {self.agent.num_res_blocks}个残差块, 128通道, {sum(p.numel() for p in self.agent.policy_value_net.parameters())/1e6:.2f}M参数")
        print(f"开始局数: {start_episode}")
        print(f"目标局数: {n_games}")
        print(f"批次大小: {self.batch_size}")
        print(f"MCTS模拟次数: {self.n_simulations}")
        print("="*60)
        
        for i in tqdm(range(start_episode, n_games)):
            # 自我对弈
            play_data = self.self_play(temp=1.0)
            
            # 数据增强
            augmented_data = self.augment_data(play_data)
            self.data_buffer.extend(augmented_data)
            
            # 训练更新
            if len(self.data_buffer) >= self.batch_size:
                loss, policy_loss, value_loss = self.policy_update()
                
                if (i + 1) % 10 == 0:
                    avg_length = np.mean(self.episode_lengths[-10:])
                    print(f"\n第 {i+1} 局: 平均步数={avg_length:.1f}, Loss={loss:.4f}, "
                          f"Policy Loss={policy_loss:.4f}, Value Loss={value_loss:.4f}")
            
            # 保存模型
            if (i + 1) % save_interval == 0:
                model_path = os.path.join(model_dir, f'improved_alphazero_model_{i+1}.pth')
                self.agent.save_model(model_path)
        
        # 保存最终模型
        final_path = os.path.join(model_dir, 'improved_alphazero_final.pth')
        self.agent.save_model(final_path)
        
        # 绘制训练曲线
        self._plot_training_curves(model_dir)
        
        print(f"\n训练完成！共完成{n_games}局自我对弈")
    
    def _plot_training_curves(self, model_dir):
        """绘制训练曲线"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.losses)
        plt.title('Training Loss')
        plt.xlabel('Update Steps')
        plt.ylabel('Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(self.episode_lengths)
        plt.title('Episode Length')
        plt.xlabel('Episodes')
        plt.ylabel('Steps')
        
        plt.tight_layout()
        plt.savefig(os.path.join(model_dir, 'improved_training_curves.png'))
        print(f"训练曲线已保存到 {model_dir}/improved_training_curves.png")


if __name__ == '__main__':
    trainer = ImprovedAlphaZeroTrainer(
        board_size=15,
        num_res_blocks=15,
        n_simulations=400,
        c_puct=5.0,
        buffer_size=10000,
        batch_size=512,  # 显存充足，提升批次大小
        epochs=5
    )
    
    # 开始训练
    trainer.train(
        n_games=5000,
        save_interval=500,
        resume_from=None  # 如果需要继续训练，设置为模型路径
    )
