"""
五子棋游戏环境
实现标准的五子棋规则和状态管理
"""
import numpy as np
from typing import Tuple, Optional
import copy


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
        self.observation_space_shape = (board_size, board_size, 3)
        
        # ====================== 奖励配置 ======================
        self.reward_config = {
            # 终局奖励
            "win": 1.0,
            "lose": -1.0,
            "draw": 0.0,
            "illegal": -10.0,
            # 局中奖励
            "live_four": 0.3,
            "chong_four": 0.15,
            "live_three": 0.08,
            "block_live_four": 0.25,
            "block_chong_four": 0.12,
            "block_live_three": 0.05,
            # 位置奖励
            "center_bonus": 0.02,
            "edge_penalty": -0.01,
        }
        
        # ====================== Episode监控机制 ======================
        self.episode_reward = 0.0  # 当前episode累计奖励
        self.episode_step = 0      # 当前episode步数
        self.episode_stats = []    # 历史episode统计
        
        self.reset()
    
    def reset(self) -> np.ndarray:
        """重置游戏环境"""
        # 保存上一局统计（如果有）
        if hasattr(self, 'episode_step') and self.episode_step > 0:
            self.episode_stats.append({
                'total_reward': self.episode_reward,
                'steps': self.episode_step,
                'winner': self.winner
            })
            # 保留最近100局统计
            if len(self.episode_stats) > 100:
                self.episode_stats.pop(0)
        
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int8)
        self.current_player = 1  # 1代表黑棋，-1代表白棋
        self.done = False
        self.winner = None
        
        # 重置episode监控
        self.episode_reward = 0.0
        self.episode_step = 0
        
        return self._get_observation()
    
    def _get_observation(self) -> np.ndarray:
        """
        获取当前状态观测
        
        始终从当前玩家视角表示状态
        这样网络学到的是"如何为当前要走的人选择最佳动作"
        
        Returns:
            3通道的状态表示:
            - 通道0: 当前玩家的棋子位置
            - 通道1: 对手的棋子位置
            - 通道2: 颜色指示器（当前玩家是黑棋=1，白棋=0）
        """
        obs = np.zeros((self.board_size, self.board_size, 3), dtype=np.float32)
        
        # 通道0: 当前玩家的棋子
        obs[:, :, 0] = (self.board == self.current_player).astype(np.float32)
        
        # 通道1: 对手的棋子
        obs[:, :, 1] = (self.board == -self.current_player).astype(np.float32)
        
        # 通道2: 当前玩家颜色指示（黑棋=1，白棋=0）
        # 这让网络知道当前是谁在走，可以学习先手/后手的不同策略
        if self.current_player == 1:
            obs[:, :, 2] = 1.0
        else:
            obs[:, :, 2] = 0.0
        
        return obs
    
    def get_valid_actions(self) -> np.ndarray:
        """
        获取所有合法动作
        Returns:
            合法动作的mask数组
        """
        return (self.board.flatten() == 0).astype(np.float32)
    
    def step(self, action: int, skip_dense: bool = False) -> Tuple[np.ndarray, float, bool, dict]:
        """
        执行动作
        Args:
            action: 动作索引 (0 到 board_size^2 - 1)
            skip_dense: 是否跳过稠密奖励计算（MCTS搜索时为True，加速10倍）
        Returns:
            observation: 新状态
            reward: 奖励
            done: 是否结束
            info: 额外信息
        """
        if self.done:
            raise ValueError("游戏已结束，请调用reset()重新开始")
        
        self.episode_step += 1
        row = action // self.board_size
        col = action % self.board_size
        
        # 检查动作是否合法
        if not (0 <= row < self.board_size and 0 <= col < self.board_size):
            self.done = True
            self.winner = -self.current_player
            reward = self.reward_config["illegal"]
            self.episode_reward += reward
            return self._get_observation(), reward, True, {
                "winner": self.winner,
                "illegal_move": True,
                "reward_type": "illegal",
                "loser_reward": self.reward_config["lose"]
            }
        
        if self.board[row, col] != 0:
            # 非法动作，给予负奖励并结束游戏
            self.done = True
            self.winner = -self.current_player
            reward = self.reward_config["illegal"]
            self.episode_reward += reward
            return self._get_observation(), reward, True, {
                "winner": self.winner,
                "illegal_move": True,
                "reward_type": "illegal",
                "loser_reward": self.reward_config["lose"]
            }
        
        # 放置棋子
        self.board[row, col] = self.current_player
        
        # 检查是否获胜
        if self._check_win(row, col):
            self.done = True
            self.winner = self.current_player
            reward = self.reward_config["win"]
            self.episode_reward += reward
            return self._get_observation(), reward, True, {
                "winner": self.current_player,
                "reward_type": "win"
            }
        
        # 检查是否平局
        if not np.any(self.board == 0):
            self.done = True
            self.winner = 0
            reward = self.reward_config["draw"]
            self.episode_reward += reward
            return self._get_observation(), reward, True, {
                "winner": 0,
                "draw": True,
                "reward_type": "draw"
            }
        
        # ====================== 计算稠密奖励（非终局） ======================
        if skip_dense:
            dense_reward = 0.0  # MCTS搜索时跳过，加速10倍
        else:
            dense_reward = self._calculate_dense_reward(row, col)
            self.episode_reward += dense_reward
        
        # 切换玩家
        self.current_player = -self.current_player
        
        return self._get_observation(), dense_reward, False, {
            "dense_reward": dense_reward,
            "reward_type": "dense"
        }
    
    def _check_line(self, row: int, col: int, dr: int, dc: int, player: int) -> Tuple[int, bool, bool]:
        """
        检测指定位置沿指定方向的连子情况
        Args:
            row/col: 起始位置
            dr/dc: 方向（如(1,0)是向下，(1,1)是右下）
            player: 玩家（1/-1）
        Returns:
            count: 连子数
            left_empty: 连子左端是否为空
            right_empty: 连子右端是否为空
        """
        count = 1
        # 正向检测（dr/dc方向）
        r, c = row + dr, col + dc
        while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
            count += 1
            r += dr
            c += dc
        right_empty = (0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == 0)
        
        # 反向检测（-dr/-dc方向）
        r, c = row - dr, col - dc
        while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
            count += 1
            r -= dr
            c -= dc
        left_empty = (0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == 0)
        
        return count, left_empty, right_empty
    
    def _calculate_dense_reward(self, row: int, col: int) -> float:
        """
        计算当前落子后的稠密奖励（中间奖励）
        堵子检测时临时移除当前落子，检测对手连子后再放回
        Args:
            row/col: 刚落子的位置
        Returns:
            稠密奖励值
        """
        reward = 0.0
        player = self.current_player
        opponent = -player
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]  # 横、竖、主对角、副对角
        visited = set()  # 避免重复计算同一连子
        
        # ====================== 位置奖励 ======================
        center = self.board_size // 2
        dist_from_center = max(abs(row - center), abs(col - center))
        if dist_from_center <= 2:  # 中心区域
            reward += self.reward_config["center_bonus"]
        elif dist_from_center >= center - 1:  # 边缘区域
            reward += self.reward_config["edge_penalty"]
        
        # ====================== 开局靠近对手奖励 ======================
        total_pieces = np.sum(self.board != 0)
        if total_pieces <= 6:  # 前6步
            opponent_positions = np.argwhere(self.board == opponent)
            if len(opponent_positions) > 0:
                # 计算与最近对手棋子的距离（切比雪夫距离）
                min_dist = min(max(abs(row - r), abs(col - c)) for r, c in opponent_positions)
                if min_dist <= 2:  # 距离对手2格以内
                    reward += 0.03
        
        # 1. 计算当前玩家的连子奖励
        for dr, dc in directions:
            if (row, col, dr, dc) in visited:
                continue
            count, left_empty, right_empty = self._check_line(row, col, dr, dc, player)
            # 标记已检测的连子（避免重复）
            for i in range(-count + 1, 1):
                r = row + dr * i
                c = col + dc * i
                if 0 <= r < self.board_size and 0 <= c < self.board_size:
                    visited.add((r, c, dr, dc))
            
            # 活四（4连子，两端空）
            if count == 4 and left_empty and right_empty:
                reward += self.reward_config["live_four"]
            # 冲四（4连子，一端空）
            elif count == 4 and (left_empty or right_empty):
                reward += self.reward_config["chong_four"]
            # 活三（3连子，两端空）
            elif count == 3 and left_empty and right_empty:
                reward += self.reward_config["live_three"]
        
        # 2. 计算堵对手的奖励【修复：临时移除当前落子】
        self.board[row, col] = 0  # 临时移除
        
        for dr, dc in directions:
            # 检测对手在该方向的连子（假设当前位置为空）
            count_before, left_empty, right_empty = self._check_line_from_empty(row, col, dr, dc, opponent)
            
            # 检查是否堵住了对手的危险连子
            if count_before >= 3 and (left_empty or right_empty):
                if count_before >= 4 and left_empty and right_empty:
                    reward += self.reward_config["block_live_four"]
                    self.board[row, col] = player  # 放回
                    return min(reward, self.reward_config["win"] * 0.9)
                elif count_before >= 4 and (left_empty or right_empty):
                    reward += self.reward_config["block_chong_four"]
                elif count_before == 3 and left_empty and right_empty:
                    reward += self.reward_config["block_live_three"]
        
        self.board[row, col] = player  # 放回
        
        # 奖励上限（避免单步奖励超过终局奖励）
        return min(reward, self.reward_config["win"] * 0.9)
    
    def _check_line_from_empty(self, row: int, col: int, dr: int, dc: int, player: int) -> Tuple[int, bool, bool]:
        """
        从空位置检测指定方向的连子（用于堵子检测）
        检查(row,col)两侧是否有对手的连续棋子
        Returns:
            count: 两侧连子总数
            left_empty: 左侧连子外是否为空
            right_empty: 右侧连子外是否为空
        """
        count = 0
        
        # 正向检测
        r, c = row + dr, col + dc
        right_count = 0
        while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
            right_count += 1
            r += dr
            c += dc
        right_empty = (0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == 0)
        
        # 反向检测
        r, c = row - dr, col - dc
        left_count = 0
        while 0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == player:
            left_count += 1
            r -= dr
            c -= dc
        left_empty = (0 <= r < self.board_size and 0 <= c < self.board_size and self.board[r, c] == 0)
        
        count = left_count + right_count
        return count, left_empty, right_empty
    
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
    
    def check_forbidden(self, row: int, col: int) -> Optional[str]:
        """
        检查黑棋禁手（仅黑棋有禁手）
        Args:
            row/col: 待落子位置
        Returns:
            None: 无禁手
            'double_three': 三三禁手
            'double_four': 四四禁手
            'overline': 长连禁手
        """
        # 只检查黑棋
        if self.current_player != 1:
            return None
        
        # 模拟落子
        self.board[row, col] = 1
        
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        three_count = 0  # 活三数量
        four_count = 0   # 四数量
        
        for dr, dc in directions:
            count, left_empty, right_empty = self._check_line(row, col, dr, dc, 1)
            
            # 检查长连（6子以上）
            if count >= 6:
                self.board[row, col] = 0  # 恢复
                return 'overline'
            
            # 检查活四/冲四
            if count == 4:
                four_count += 1
            
            # 检查活三（两端都空）
            if count == 3 and left_empty and right_empty:
                three_count += 1
        
        self.board[row, col] = 0  # 恢复
        
        # 四四禁手
        if four_count >= 2:
            return 'double_four'
        
        # 三三禁手
        if three_count >= 2:
            return 'double_three'
        
        return None
    
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
    
    # ====================== Episode监控接口 ======================
    def get_episode_stats(self) -> dict:
        """获取当前episode统计信息"""
        return {
            'current_reward': self.episode_reward,
            'current_step': self.episode_step,
            'avg_reward_per_step': self.episode_reward / max(1, self.episode_step)
        }
    
    def get_training_stats(self) -> dict:
        """获取训练统计信息（用于监控）"""
        if not self.episode_stats:
            return {'message': 'No completed episodes yet'}
        
        recent = self.episode_stats[-10:]  # 最近10局
        avg_reward = sum(ep['total_reward'] for ep in recent) / len(recent)
        avg_steps = sum(ep['steps'] for ep in recent) / len(recent)
        
        return {
            'total_episodes': len(self.episode_stats),
            'recent_avg_reward': round(avg_reward, 4),
            'recent_avg_steps': round(avg_steps, 1),
            'warning': 'Reward too high!' if avg_reward > 1.5 else None
        }
    
    # ====================== 供MCTS调用的稠密奖励接口 ======================
    def get_dense_reward(self) -> float:
        """获取当前局面的稠密奖励（供MCTS优化使用）"""
        if self.done:
            if self.winner == self.current_player:
                return self.reward_config["win"]
            elif self.winner == -self.current_player:
                return self.reward_config["lose"]
            else:
                return self.reward_config["draw"]
        # 非终局时返回0（避免重复计算）
        return 0.0
    
    # def __deepcopy__(self, memo):
    #     """
    #     支持深拷贝（MCTS需要）
    #     这是一个优化版本，比默认的deepcopy更快
    #     """
    #     new_env = GomokuEnv(board_size=self.board_size)
    #     new_env.board = self.board.copy()
    #     new_env.current_player = self.current_player
    #     new_env.done = self.done
    #     new_env.winner = self.winner
    #     return new_env
    def copy(self):
        """
         快速复制方法 - 比deepcopy快10倍以上
        """
        new_env = GomokuEnv.__new__(GomokuEnv)
        new_env.board_size = self.board_size
        new_env.board = self.board.copy()
        new_env.current_player = self.current_player
        new_env.done = self.done
        new_env.winner = self.winner
        new_env.action_space = self.action_space
        new_env.observation_space_shape = self.observation_space_shape
        # 复制奖励配置和监控状态
        new_env.reward_config = self.reward_config.copy()
        new_env.episode_reward = self.episode_reward
        new_env.episode_step = self.episode_step
        new_env.episode_stats = []  # 副本不继承历史统计
        return new_env
    
    def __deepcopy__(self, memo):
        """兼容deepcopy，但实际使用快速复制"""
        return self.copy()