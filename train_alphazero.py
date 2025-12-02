"""
AlphaZero风格训练 - 结合MCTS和策略价值网络
"""
import numpy as np
from collections import deque, defaultdict
from gomoku_env import GomokuEnv
from mcts import MCTS
from policy_value_net import PolicyValueAgent
import os
from tqdm import tqdm
import matplotlib.pyplot as plt


class AlphaZeroTrainer:
    """AlphaZero训练器"""
    
    def __init__(
        self,
        board_size: int = 15,
        n_simulations: int = 400,
        c_puct: float = 5.0,
        buffer_size: int = 10000,
        batch_size: int = 512,
        epochs: int = 5,
        kl_target: float = 0.02
    ):
        self.board_size = board_size
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.kl_target = kl_target
        
        # 创建环境和智能体
        self.env = GomokuEnv(board_size=board_size)
        self.agent = PolicyValueAgent(board_size=board_size)
        
        # 数据缓冲区
        self.data_buffer = deque(maxlen=buffer_size)
        
        # 训练统计
        self.episode_lengths = []
        self.losses = []
    
    def get_temperature(self, episode: int):
        """根据训练进度调整温度参数"""
        if episode < 1000:
            return 1.0  # 高探索
        elif episode < 5000:
            return 0.5  # 中等探索
        else:
            return 0.1  # 低探索
    
    def self_play(self, episode: int, temp: float = 1.0):
        """
        自我对弈收集数据
        Args:
            episode: 当前训练轮次
            temp: 温度参数
        Returns:
            play_data: 对局数据列表 [(state, mcts_probs, player)]
        """
        play_data = []
        state = self.env.reset()
        
        # 根据训练进度调整温度
        temp = self.get_temperature(episode)
        
        # 创建MCTS
        mcts = MCTS(self.agent.policy_value_fn, c_puct=self.c_puct, n_simulations=self.n_simulations)
        
        step = 0
        while True:
            # 使用MCTS获取动作概率
            action, action_probs = mcts.get_action(self.env, temp=temp)
            
            # 保存数据（状态，MCTS概率，当前玩家）
            play_data.append((state.copy(), action_probs, self.env.current_player))
            
            # 执行动作
            state, _, done, info = self.env.step(int(action))
            
            # 更新MCTS树
            mcts.update_with_move(action)
            
            step += 1
            
            if done:
                # 游戏结束，分配奖励
                winner = info.get('winner', 0)
                
                # 转换数据：从玩家视角分配胜负
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
        """
        数据增强（简化版，只使用基本旋转和翻转）
        Args:
            play_data: 原始数据
        Returns:
            augmented_data: 增强后的数据
        """
        augmented_data = []
        
        for state, probs, reward in play_data:
            # 确保状态形状为 (15, 15, 3)
            if state.shape != (self.board_size, self.board_size, 3):
                continue  # 跳过不合法的状态
            
            # 原始数据
            augmented_data.append((state.copy(), probs.copy(), reward))
            
            # 旋转180度（最稳定的变换）
            state_rot180 = np.rot90(state, k=2, axes=(0, 1))
            probs_rot180 = self._rotate_probs(probs, k=2)
            augmented_data.append((state_rot180.copy(), probs_rot180.copy(), reward))
            
            # 水平翻转
            state_flip = np.flip(state, axis=1)
            probs_flip = self._flip_probs(probs)
            augmented_data.append((state_flip.copy(), probs_flip.copy(), reward))
        
        return augmented_data
    
    def _flip_probs_vertical(self, probs):
        """垂直翻转概率分布"""
        probs_2d = probs.reshape(self.board_size, self.board_size)
        probs_flip = np.flipud(probs_2d)
        return probs_flip.flatten()
    
    def _flip_probs_diagonal(self, probs):
        """对角线翻转概率分布"""
        probs_2d = probs.reshape(self.board_size, self.board_size)
        probs_flip = probs_2d.T
        return probs_flip.flatten()
    
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
        mini_batch = np.random.choice(len(self.data_buffer), size=self.batch_size)
        state_list = [self.data_buffer[i][0] for i in mini_batch]
        mcts_probs_list = [self.data_buffer[i][1] for i in mini_batch]
        winner_list = [self.data_buffer[i][2] for i in mini_batch]
        
        # 转换为numpy数组
        state_batch = np.array(state_list)
        mcts_probs_batch = np.array(mcts_probs_list)
        winner_batch = np.array(winner_list)
        
        # 训练多个epoch
        loss = 0.0
        policy_loss = 0.0
        value_loss = 0.0
        
        for _ in range(self.epochs):
            loss, policy_loss, value_loss = self.agent.train_step(
                state_batch, mcts_probs_batch, winner_batch
            )
            self.losses.append(loss)
        
        return loss, policy_loss, value_loss
    
    def train(self, n_games: int = 1500, save_interval: int = 100, model_dir: str = 'models_alphazero', resume_from: str = None):
        """
        训练主循环
        Args:
            n_games: 自我对弈局数
            save_interval: 保存间隔
            model_dir: 模型保存目录
            resume_from: 恢复训练的模型路径（例如 'models_alphazero/alphazero_model_500.pth'）
        """
        os.makedirs(model_dir, exist_ok=True)
        
        # 如果指定了resume_from，加载检查点
        start_episode = 0
        if resume_from and os.path.exists(resume_from):
            print(f"正在加载检查点: {resume_from}")
            self.agent.load_model(resume_from)
            
            # 从文件名提取已训练的局数
            import re
            match = re.search(r'model_(\d+)\.pth', resume_from)
            if match:
                start_episode = int(match.group(1))
                print(f"从第 {start_episode} 局继续训练")
        
        print("="*60)
        print("开始AlphaZero训练")
        print(f"开始局数: {start_episode}")
        print(f"目标局数: {n_games}")
        print(f"MCTS模拟次数: {self.n_simulations}")
        print(f"缓冲区大小: {self.buffer_size}")
        print("="*60)
        
        for i in tqdm(range(start_episode, n_games)):
            # 自我对弈
            play_data = self.self_play(i, temp=1.0)
            
            # 数据增强
            augmented_data = self.augment_data(play_data)
            self.data_buffer.extend(augmented_data)
            
            # 当缓冲区足够大时开始训练
            if len(self.data_buffer) >= self.batch_size:
                loss, policy_loss, value_loss = self.policy_update()
                
                if (i + 1) % 10 == 0:
                    avg_length = np.mean(self.episode_lengths[-10:])
                    print(f"\n第 {i+1} 局: 平均步数={avg_length:.1f}, "
                          f"Loss={loss:.4f}, Policy Loss={policy_loss:.4f}, Value Loss={value_loss:.4f}")
            
            # 保存模型
            if (i + 1) % save_interval == 0:
                model_path = os.path.join(model_dir, f'alphazero_model_{i+1}.pth')
                self.agent.save_model(model_path)
        
        # 保存最终模型
        final_path = os.path.join(model_dir, 'alphazero_final.pth')
        self.agent.save_model(final_path)
        
        # 绘制训练曲线
        self._plot_training_curves(model_dir)
        
        print("\n训练完成！")
    
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
        plt.savefig(os.path.join(model_dir, 'alphazero_training_curves.png'))
        print(f"训练曲线已保存到 {model_dir}/alphazero_training_curves.png")


if __name__ == '__main__':
    trainer = AlphaZeroTrainer(
        board_size=15,
        n_simulations=400,  # 使用400次模拟（平衡速度和效果）
        c_puct=5.0,
        buffer_size=10000,
        batch_size=512,
        epochs=5
    )
    
    # 从头开始训练
    trainer.train(
        n_games=5000,  # 先训练5000局
        save_interval=500,  # 每500局保存一次
        resume_from=None  # 从头开始训练
    )
