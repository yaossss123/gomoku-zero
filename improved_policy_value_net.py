"""
改进的策略价值网络 - 使用残差块
基于AlphaZero架构：初始卷积 + 残差块 + 策略/价值头
支持稠密奖励作为价值标签
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from config import L2_WEIGHT_DECAY, BOARD_SIZE, NUM_RES_BLOCKS, NUM_CHANNELS, LEARNING_RATE


class ResidualBlock(nn.Module):
    """标准残差块"""
    
    def __init__(self, channels: int = 128):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class ImprovedPolicyValueNet(nn.Module):
    """改进的策略价值网络"""
    
    def __init__(self, board_size: int = BOARD_SIZE, num_res_blocks: int = NUM_RES_BLOCKS, num_channels: int = NUM_CHANNELS):
        super(ImprovedPolicyValueNet, self).__init__()
        self.board_size = board_size
        self.num_res_blocks = num_res_blocks
        self.num_channels = num_channels
        
        # ===== 主干网络 =====
        self.conv_init = nn.Conv2d(3, num_channels, kernel_size=3, padding=1, bias=False)
        self.bn_init = nn.BatchNorm2d(num_channels)
        
        # 残差块
        self.res_blocks = nn.ModuleList([
            ResidualBlock(channels=num_channels) for _ in range(num_res_blocks)
        ])
        
        # ===== 策略头 =====
        self.policy_conv = nn.Conv2d(num_channels, 2, kernel_size=1, bias=False)
        self.policy_bn = nn.BatchNorm2d(2)
        self.policy_fc = nn.Linear(2 * board_size * board_size, board_size * board_size)
        
        # ===== 价值头 =====
        self.value_conv = nn.Conv2d(num_channels, 1, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(1 * board_size * board_size, 256)
        self.value_fc2 = nn.Linear(256, 1)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # 共享特征提取
        x = F.relu(self.bn_init(self.conv_init(x)))
        for res_block in self.res_blocks:
            x = res_block(x)
        
        # 策略头 - 修复：使用 reshape 代替 view
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.reshape(policy.size(0), -1)  # 修复：view -> reshape
        log_action_probs = F.log_softmax(self.policy_fc(policy), dim=1)
        
        # 价值头 - 修复：使用 reshape 代替 view
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.reshape(value.size(0), -1)  # 修复：view -> reshape
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return log_action_probs, value


class ImprovedPolicyValueAgent:
    """改进的策略价值智能体
    支持稠密奖励作为价值标签
    """
    
    def __init__(
        self,
        board_size: int = BOARD_SIZE,
        num_res_blocks: int = NUM_RES_BLOCKS,
        num_channels: int = NUM_CHANNELS,
        learning_rate: float = LEARNING_RATE,
        l2_const: float = L2_WEIGHT_DECAY,
        force_gpu: bool = True
    ):
        self.board_size = board_size
        self.num_res_blocks = num_res_blocks
        self.num_channels = num_channels
        self.l2_const = l2_const
        
        # 强制GPU检测
        self.device = torch.device("cuda" if (torch.cuda.is_available() and force_gpu) else "cpu")
        print(f"使用设备: {self.device}")
        
        if force_gpu and self.device.type == "cpu":
            raise RuntimeError("❌ 强制使用GPU，但CUDA不可用！请检查PyTorch安装（需CUDA版本）。")
        
        self.policy_value_net = ImprovedPolicyValueNet(
            board_size=board_size,
            num_res_blocks=num_res_blocks,
            num_channels=num_channels
        ).to(self.device)
        
        self.optimizer = optim.Adam(
            self.policy_value_net.parameters(),
            lr=learning_rate,
            weight_decay=self.l2_const
        )
        
        total_params = sum(p.numel() for p in self.policy_value_net.parameters())
        trainable_params = sum(p.numel() for p in self.policy_value_net.parameters() if p.requires_grad)
        print(f"网络结构: {num_res_blocks}个残差块, {num_channels}通道")
        print(f"总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
        print(f"可训练参数: {trainable_params:,}")
    
    def policy_value_fn(self, state: np.ndarray):
        """策略价值函数（供MCTS使用）"""
        self.policy_value_net.eval()
        
        with torch.no_grad():
            # 修复：确保数据连续
            state_tensor = torch.FloatTensor(
                np.ascontiguousarray(state)
            ).permute(2, 0, 1).unsqueeze(0).to(self.device)
            
            log_action_probs, value = self.policy_value_net(state_tensor)
            
            action_probs = np.exp(log_action_probs.cpu().numpy()[0])
            value = value.cpu().numpy()[0][0]
        
        return action_probs, value
    
    def train_step(self, state_batch, mcts_probs_batch, value_labels_batch):
        """训练一步
        输入从 winner_batch → value_labels_batch（适配稠密奖励）
        """
        self.policy_value_net.train()
        
        state_batch = np.ascontiguousarray(state_batch)
        mcts_probs_batch = np.ascontiguousarray(mcts_probs_batch)
        
        # 转换为张量
        states = torch.FloatTensor(state_batch).permute(0, 3, 1, 2).to(self.device)
        mcts_probs = torch.FloatTensor(mcts_probs_batch).to(self.device)
        value_labels = torch.FloatTensor(value_labels_batch).to(self.device)
        
        # 前向传播
        log_action_probs, values = self.policy_value_net(states)
        values = values.view(-1)
        
        # 计算损失
        policy_loss = -torch.mean(torch.sum(mcts_probs * log_action_probs, dim=1))
        value_loss = F.mse_loss(values, value_labels)
        loss = policy_loss + value_loss
        
        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_value_net.parameters(), max_norm=5.0)
        self.optimizer.step()
        
        return loss.item(), policy_loss.item(), value_loss.item()
    
    def save_model(self, path: str):
        torch.save({
            'net': self.policy_value_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'num_res_blocks': self.num_res_blocks,
            'num_channels': self.num_channels,
            'board_size': self.board_size
        }, path)
        print(f"模型已保存到: {path}")
    
    def load_model(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_value_net.load_state_dict(checkpoint['net'])
        self.policy_value_net = self.policy_value_net.to(self.device)
        
        if 'optimizer' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                print("优化器状态加载失败，使用新优化器")
        
        print(f"模型已从 {path} 加载，并绑定到 {self.device}")