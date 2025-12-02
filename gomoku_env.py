"""
五子棋游戏环境
实现标准的五子棋规则和状态管理
"""
import numpy as np
from typing import Tuple, Optional


class GomokuEnv:
    """五子棋环境类"""
    
    def __init__(self, board_size: int = 15):
        """
        初始化五子棋环境
        Args:
            board_size: 棋盘大小，默认15x15
        """
        self.board_size = board_size
        self.board = None
        self.current_player = None
        self.done = False
        self.winner = None
        self.action_space = board_size * board_size
        self.observation_space_shape = (board_size, board_size, 3)  # 3个通道：黑棋、白棋、当前玩家
        self.reset()
    
    def reset(self) -> np.ndarray:
        """重置游戏环境"""
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1  # 1代表黑棋，-1代表白棋
        self.done = False
        self.winner = None
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """
        获取当前状态观测
        Returns:
            3通道的状态表示 [黑棋位置, 白棋位置, 当前玩家]
        """
        obs = np.zeros((self.board_size, self.board_size, 3), dtype=np.float32)
        obs[:, :, 0] = (self.board == 1).astype(np.float32)  # 黑棋
        obs[:, :, 1] = (self.board == -1).astype(np.float32)  # 白棋
        obs[:, :, 2] = np.full((self.board_size, self.board_size), 
                               self.current_player, dtype=np.float32)  # 当前玩家
        return obs
    
    def get_valid_actions(self) -> np.ndarray:
        """
        获取所有合法动作
        Returns:
            合法动作的mask数组
        """
        return (self.board.flatten() == 0).astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        执行动作
        Args:
            action: 动作索引 (0 到 board_size^2 - 1)
        Returns:
            observation: 新状态
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        if self.done:
            raise ValueError("游戏已结束，请调用reset()重新开始")
        
        row = action // self.board_size
        col = action % self.board_size
        
        # 检查动作是否合法
        if self.board[row, col] != 0:
            # 非法动作，给予负奖励并结束游戏
            return self._get_observation(), -10, True, {"winner": -self.current_player, "illegal_move": True}
        
        # 放置棋子
        self.board[row, col] = self.current_player
        
        # 检查是否获胜
        if self._check_win(row, col):
            self.done = True
            self.winner = self.current_player
            reward = 1.0  # 获胜奖励
            return self._get_observation(), reward, True, {"winner": self.current_player}
        
        # 检查是否平局
        if not np.any(self.board == 0):
            self.done = True
            self.winner = 0
            return self._get_observation(), 0, True, {"winner": 0, "draw": True}
        
        # 切换玩家
        self.current_player = -self.current_player
        
        return self._get_observation(), 0, False, {}
    
    def _check_win(self, row: int, col: int) -> bool:
        """
        检查当前位置是否形成五子连珠
        Args:
            row: 行索引
            col: 列索引
        Returns:
            是否获胜
        """
        player = self.board[row, col]
        directions = [
            [(0, 1), (0, -1)],   # 水平
            [(1, 0), (-1, 0)],   # 垂直
            [(1, 1), (-1, -1)],  # 主对角线
            [(1, -1), (-1, 1)]   # 副对角线
        ]
        
        for direction_pair in directions:
            count = 1  # 当前棋子
            for dr, dc in direction_pair:
                r, c = row + dr, col + dc
                while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
                    count += 1
                    r += dr
                    c += dc
            if count >= 5:
                return True
        return False
    
    def render(self):
        """打印棋盘"""
        print("\n  ", end="")
        for i in range(self.board_size):
            print(f"{i:2d}", end=" ")
        print()
        
        for i in range(self.board_size):
            print(f"{i:2d} ", end="")
            for j in range(self.board_size):
                if self.board[i, j] == 1:
                    print(" ●", end=" ")  # 黑棋
                elif self.board[i, j] == -1:
                    print(" ○", end=" ")  # 白棋
                else:
                    print(" ·", end=" ")  # 空位
            print()
        print()
    
    def get_board_copy(self) -> np.ndarray:
        """获取棋盘副本"""
        return self.board.copy()
