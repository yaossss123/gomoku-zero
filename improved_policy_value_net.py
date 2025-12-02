"""
改进的策略价值网络 - 使用残差块
基于AlphaZero架构：初始卷积 + 10个残差块 + 策略/价值头
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """标准残差块"""
    
    def __init__(self, channels: int = 128):
        """
        初始化残差块
        Args:
            channels: 通道数（输入输出通道数相同）
        """
        super(ResidualBlock, self).__init__()
        # 第一个卷积层
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        
        # 第二个卷积层
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入特征图 (batch, channels, H, W)
        Returns:
            输出特征图（尺寸与输入相同）
        """
        residual = x  # 保存输入用于跳跃连接
        
        # 第一个卷积块
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        
        # 第二个卷积块
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 跳跃连接：将输入加到输出上
        out += residual
        
        # 最后激活
        out = F.relu(out)
        
        return out


class ImprovedPolicyValueNet(nn.Module):
    """改进的策略价值网络（10个残差块）"""
    
    def __init__(self, board_size: int = 15, num_res_blocks: int = 10):
        """
        初始化网络
        Args:
            board_size: 棋盘大小
            num_res_blocks: 残差块数量
        """
        super(ImprovedPolicyValueNet, self).__init__()
        self.board_size = board_size
        self.num_res_blocks = num_res_blocks
        
        # 初始卷积层：3通道 -> 128通道
        self.conv_init = nn.Conv2d(3, 128, kernel_size=3, padding=1)
        self.bn_init = nn.BatchNorm2d(128)
        
        # 10个残差块（每块包含2个卷积层）
        self.res_blocks = nn.ModuleList([
            ResidualBlock(channels=128) for _ in range(num_res_blocks)
        ])
        
        # ===== 策略头 =====
        # 128通道 -> 4通道（降维）
        self.policy_conv = nn.Conv2d(128, 4, kernel_size=1)
        self.policy_bn = nn.BatchNorm2d(4)
        # 全连接层：4*15*15 -> 225（每个位置的概率）
        self.policy_fc = nn.Linear(4 * board_size * board_size, board_size * board_size)
        
        # ===== 价值头 =====
        # 128通道 -> 2通道（降维）
        self.value_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.value_bn = nn.BatchNorm2d(2)
        # 全连接层：2*15*15 -> 256 -> 1
        self.value_fc1 = nn.Linear(2 * board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入状态 (batch, 3, board_size, board_size)
        Returns:
            log_action_probs: 动作对数概率 (batch, 225)
            value: 状态价值 (batch, 1)
        """
        # ===== 共享特征提取 =====
        # 初始卷积：(batch, 3, 15, 15) -> (batch, 128, 15, 15)
        x = self.conv_init(x)
        x = self.bn_init(x)
        x = F.relu(x)
        
        # 通过10个残差块：特征维度保持 (batch, 128, 15, 15)
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # ===== 策略头 =====
        # (batch, 128, 15, 15) -> (batch, 4, 15, 15)
        policy = self.policy_conv(x)
        policy = self.policy_bn(policy)
        policy = F.relu(policy)
        # 展平：(batch, 4, 15, 15) -> (batch, 900)
        policy = policy.reshape(policy.size(0), -1)
        # 全连接：(batch, 900) -> (batch, 225)
        log_action_probs = F.log_softmax(self.policy_fc(policy), dim=1)
        
        # ===== 价值头 =====
        # (batch, 128, 15, 15) -> (batch, 2, 15, 15)
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = F.relu(value)
        # 展平：(batch, 2, 15, 15) -> (batch, 450)
        value = value.reshape(value.size(0), -1)
        # 全连接：(batch, 450) -> (batch, 256) -> (batch, 1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return log_action_probs, value


class ImprovedPolicyValueAgent:
    """改进的策略价值智能体"""
    
    def __init__(
        self,
        board_size: int = 15,
        num_res_blocks: int = 10,
        learning_rate: float = 0.002,
        l2_const: float = 1e-4
    ):
        """
        初始化智能体
        Args:
            board_size: 棋盘大小
            num_res_blocks: 残差块数量
            learning_rate: 学习率
            l2_const: L2正则化系数
        """
        self.board_size = board_size
        self.num_res_blocks = num_res_blocks
        self.l2_const = l2_const
        
        # 设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        
        # 策略价值网络
        self.policy_value_net = ImprovedPolicyValueNet(
            board_size=board_size,
            num_res_blocks=num_res_blocks
        ).to(self.device)
        
        # 优化器
        self.optimizer = optim.Adam(
            self.policy_value_net.parameters(),
            lr=learning_rate,
            weight_decay=self.l2_const
        )
        
        # 打印网络信息
        total_params = sum(p.numel() for p in self.policy_value_net.parameters())
        print(f"网络结构: {num_res_blocks}个残差块, 128通道")
        print(f"总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    
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
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_action_probs, dim=1))
        value_loss = F.mse_loss(values, winners)
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
