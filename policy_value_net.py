"""
策略价值网络 (Policy-Value Network)
AlphaZero风格的双头网络：策略头+价值头
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class PolicyValueNet(nn.Module):
    """策略价值网络"""
    
    def __init__(self, board_size: int = 15):
        super(PolicyValueNet, self).__init__()
        self.board_size = board_size
        
        # 共享卷积层（特征提取）
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        # 策略头（输出动作概率）
        self.policy_conv = nn.Conv2d(128, 4, kernel_size=1)
        self.policy_fc = nn.Linear(4 * board_size * board_size, board_size * board_size)
        
        # 价值头（输出状态价值）
        self.value_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.value_fc1 = nn.Linear(2 * board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入状态 (batch, 3, board_size, board_size)
        Returns:
            log_action_probs: 动作对数概率
            value: 状态价值
        """
        # 共享特征提取
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        # 策略头
        policy = F.relu(self.policy_conv(x))
        policy = policy.reshape(policy.size(0), -1)
        log_action_probs = F.log_softmax(self.policy_fc(policy), dim=1)
        
        # 价值头
        value = F.relu(self.value_conv(x))
        value = value.reshape(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return log_action_probs, value


class PolicyValueAgent:
    """策略价值智能体"""
    
    def __init__(
        self,
        board_size: int = 15,
        learning_rate: float = 0.002,
        l2_const: float = 1e-4
    ):
        self.board_size = board_size
        self.l2_const = l2_const
        
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 策略价值网络
        self.policy_value_net = PolicyValueNet(board_size).to(self.device)
        self.optimizer = optim.Adam(
            self.policy_value_net.parameters(),
            lr=learning_rate,
            weight_decay=self.l2_const
        )
    
    def policy_value_fn(self, state: np.ndarray):
        """
        策略价值函数（供MCTS使用）
        Args:
            state: 状态 (board_size, board_size, 3)
        Returns:
            action_probs: 动作概率分布
            value: 状态价值
        """
        self.policy_value_net.eval()
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).permute(2, 0, 1).unsqueeze(0).to(self.device)
            log_action_probs, value = self.policy_value_net(state_tensor)
            
            action_probs = np.exp(log_action_probs.cpu().numpy()[0])
            value = value.cpu().numpy()[0][0]
        
        return action_probs, value
    
    def train_step(self, state_batch, mcts_probs_batch, winner_batch):
        """
        训练一步
        Args:
            state_batch: 状态批次
            mcts_probs_batch: MCTS搜索得到的概率分布
            winner_batch: 游戏结果（从当前玩家视角）
        Returns:
            loss: 总损失
            policy_loss: 策略损失
            value_loss: 价值损失
        """
        self.policy_value_net.train()
        
        # 转换为张量
        states = torch.FloatTensor(state_batch).permute(0, 3, 1, 2).to(self.device)
        mcts_probs = torch.FloatTensor(mcts_probs_batch).to(self.device)
        winners = torch.FloatTensor(winner_batch).to(self.device)
        
        # 前向传播
        log_action_probs, values = self.policy_value_net(states)
        values = values.reshape(-1)
        
        # 计算损失
        # 策略损失：交叉熵
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_action_probs, dim=1))
        
        # 价值损失：均方误差
        value_loss = F.mse_loss(values, winners)
        
        # 总损失
        loss = policy_loss + value_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item(), policy_loss.item(), value_loss.item()
    
    def save_model(self, path: str):
        """保存模型"""
        torch.save({
            'net': self.policy_value_net.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
        print(f"模型已保存到: {path}")
    
    def load_model(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_value_net.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        print(f"模型已从 {path} 加载")
