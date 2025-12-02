"""
基于规则的五子棋AI
用于对战评估和基准测试
"""
import numpy as np
from typing import Tuple


class RuleBasedAI:
    """基于规则的五子棋AI"""
    
    def __init__(self, board_size: int = 15):
        self.board_size = board_size
        
        # 棋型评分表
        self.patterns = {
            'five': 100000,      # 五连
            'live_four': 10000,  # 活四 _●●●●_
            'rush_four': 1000,   # 冲四 ●X●●●_ 或 _●●●X●
            'dead_four': 10,     # 死四 X●●●●X (两头堵，无价值)
            'live_three': 1000,  # 活三 _●●●_
            'dead_three': 100,   # 眠三
            'live_two': 100,     # 活二
            'dead_two': 10,      # 眠二
        }
    
    def get_action(self, env):
        """
        获取AI的落子位置
        Args:
            env: 游戏环境
        Returns:
            action: 动作索引
        """
        board = env.board
        player = env.current_player
        
        # 获取所有合法动作
        valid_actions = env.get_valid_actions()
        valid_positions = np.where(valid_actions > 0)[0]
        
        if len(valid_positions) == 0:
            raise ValueError("没有合法动作")
        
        # 如果是第一步，下在中心
        if np.sum(board != 0) == 0:
            center = self.board_size // 2
            return center * self.board_size + center
        
        # 评估所有合法位置
        best_score = -float('inf')
        best_action = valid_positions[0]
        
        for action in valid_positions:
            row = action // self.board_size
            col = action % self.board_size
            
            # 评估该位置的价值
            score = self._evaluate_position(board, row, col, player)
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return int(best_action)
    
    def _evaluate_position(self, board, row, col, player):
        """
        评估某个位置的价值
        Args:
            board: 棋盘
            row, col: 位置
            player: 玩家
        Returns:
            score: 评分
        """
        # 临时落子
        board[row, col] = player
        
        # 评估我方得分
        my_score = self._count_patterns(board, row, col, player)
        
        # 评估对手得分（防守）
        board[row, col] = -player
        opp_score = self._count_patterns(board, row, col, -player)
        
        # 恢复棋盘
        board[row, col] = 0
        
        # 综合评分：进攻和防守
        return my_score + opp_score * 0.9
    
    def _count_patterns(self, board, row, col, player):
        """
        统计某个位置的棋型得分
        Args:
            board: 棋盘
            row, col: 位置
            player: 玩家
        Returns:
            score: 总分
        """
        score = 0
        directions = [
            (0, 1),   # 水平
            (1, 0),   # 垂直
            (1, 1),   # 主对角线
            (1, -1)   # 副对角线
        ]
        
        for dr, dc in directions:
            # 统计该方向的连子数
            count = 1
            empty_before = 0  # 0:被堵, 1:有空位
            empty_after = 0
            
            # 向前数
            r, c = row - dr, col - dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size:
                if board[r, c] == player:
                    count += 1
                elif board[r, c] == 0 and empty_before == 0:
                    empty_before = 1
                    break
                else:
                    break
                r -= dr
                c -= dc
            
            # 向后数
            r, c = row + dr, col + dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size:
                if board[r, c] == player:
                    count += 1
                elif board[r, c] == 0 and empty_after == 0:
                    empty_after = 1
                    break
                else:
                    break
                r += dr
                c += dc
            
            # 根据连子数和两端情况评分
            if count >= 5:
                score += self.patterns['five']
            elif count == 4:
                if empty_before and empty_after:
                    # 两端都有空: 活四 _●●●●_
                    score += self.patterns['live_four']
                elif empty_before or empty_after:
                    # 一端有空: 冲四 _●●●●X 或 X●●●●_
                    score += self.patterns['rush_four']
                else:
                    # 两端都堵: 死四 X●●●●X
                    score += self.patterns['dead_four']
            elif count == 3:
                if empty_before and empty_after:
                    score += self.patterns['live_three']
                else:
                    score += self.patterns['dead_three']
            elif count == 2:
                if empty_before and empty_after:
                    score += self.patterns['live_two']
                else:
                    score += self.patterns['dead_two']
        
        return score


class RandomAI:
    """随机AI（最弱基准）"""
    
    def __init__(self):
        pass
    
    def get_action(self, env):
        """随机选择合法动作"""
        valid_actions = env.get_valid_actions()
        valid_positions = np.where(valid_actions > 0)[0]
        return int(np.random.choice(valid_positions))
