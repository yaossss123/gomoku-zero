"""
蒙特卡洛树搜索 (MCTS) 
"""
import numpy as np
import math
import copy
import time
from typing import Tuple, Optional, List
from config import MCTS_SIMULATIONS_TRAIN, DIRICHLET_ALPHA, DIRICHLET_EPSILON, MCTS_C_PUCT


class MCTSNode:
    """MCTS树节点"""
    
    def __init__(self, prior: float, parent=None):
        self.parent = parent
        self.children = {}
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
    
    def expanded(self) -> bool:
        return len(self.children) > 0
    
    def value(self) -> float:
        """
        节点的平均价值
        约定：存储的是"下一个要走的玩家"的期望价值
        """
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    def select_child(self, c_puct: float = 1.5):
        """
        使用PUCT公式选择最优子节点
        
        关键点：
        - self是当前玩家P要走
        - child是P执行动作后的状态，轮到对手-P走
        - child.value()存储的是-P的期望价值
        - 对于P来说，-P的收益就是P的损失，所以取反
        """
        best_score = -float('inf')
        best_action = -1
        best_child = None
        
        sqrt_total = math.sqrt(self.visit_count + 1)
        
        for action, child in self.children.items():
            # Q值：从当前玩家视角看这个动作的价值
            # child存储的是对手的价值，所以取反
            if child.visit_count > 0:
                q_value = -child.value()
            else:
                q_value = 0.0
            
            # U值：探索奖励
            u_value = c_puct * child.prior * sqrt_total / (1 + child.visit_count)
            
            score = q_value + u_value
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def expand(self, action_probs: np.ndarray):
        """扩展节点"""
        for action, prob in enumerate(action_probs):
            if prob > 0:
                self.children[action] = MCTSNode(prior=prob, parent=self)
    
    def backup(self, value: float):
        """
        反向传播
        
        参数:
            value: 从"当前节点对应的下一个要走的玩家"视角的价值
        """
        self.visit_count += 1
        self.value_sum += value
        
        if self.parent is not None:
            # 传给父节点时取反（父节点是对手视角）
            self.parent.backup(-value)


class UrgencyDetector:
    """紧急情况检测器"""
    
    def __init__(self, board_size: int = 15):
        self.board_size = board_size
        self.directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
    
    def get_urgent_moves(self, board: np.ndarray, current_player: int) -> Tuple[List[int], List[int]]:
        """获取必胜点和必堵点"""
        winning_moves = []
        blocking_moves = []
        opponent = -current_player
        
        for row in range(self.board_size):
            for col in range(self.board_size):
                if board[row, col] != 0:
                    continue
                
                action = row * self.board_size + col
                
                if self._can_win(board, row, col, current_player):
                    winning_moves.append(action)
                
                if self._can_win(board, row, col, opponent):
                    blocking_moves.append(action)
        
        return winning_moves, blocking_moves
    
    def _can_win(self, board: np.ndarray, row: int, col: int, player: int) -> bool:
        """检查落子后是否形成五连"""
        for dr, dc in self.directions:
            count = 1
            
            r, c = row + dr, col + dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and board[r, c] == player:
                count += 1
                r += dr
                c += dc
            
            r, c = row - dr, col - dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size and board[r, c] == player:
                count += 1
                r -= dr
                c -= dc
            
            if count >= 5:
                return True
        
        return False


class MCTS:
    """蒙特卡洛树搜索 - 完全修复版"""
    
    def __init__(
        self, 
        policy_value_fn, 
        c_puct: float = MCTS_C_PUCT,
        n_simulations: int = MCTS_SIMULATIONS_TRAIN,
        dirichlet_alpha: float = DIRICHLET_ALPHA,
        dirichlet_epsilon: float = DIRICHLET_EPSILON,
        use_urgency_detection: bool = True
    ):
        self.policy_value_fn = policy_value_fn
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.use_urgency_detection = use_urgency_detection
        self.root = MCTSNode(prior=1.0)
        self.urgency_detector = None
    
    def _init_urgency_detector(self, board_size: int):
        if self.urgency_detector is None:
            self.urgency_detector = UrgencyDetector(board_size)
    
    def add_dirichlet_noise(self, node: MCTSNode):
        if not node.children:
            return
        
        actions = list(node.children.keys())
        noise = np.random.dirichlet([self.dirichlet_alpha] * len(actions))
        
        for i, action in enumerate(actions):
            child = node.children[action]
            child.prior = (1 - self.dirichlet_epsilon) * child.prior + \
                          self.dirichlet_epsilon * noise[i]
    
    def _get_valid_action_probs(self, env, state):
        action_probs, value = self.policy_value_fn(state)
        valid_actions = env.get_valid_actions()
        
        action_probs = action_probs * valid_actions
        prob_sum = np.sum(action_probs)
        
        if prob_sum > 1e-8:
            action_probs = action_probs / prob_sum
        else:
            action_probs = valid_actions / np.sum(valid_actions)
        
        return action_probs, value
    
    def _boost_urgent_priors(self, action_probs: np.ndarray, env) -> np.ndarray:
        """提升紧急落子点的先验概率"""
        if not self.use_urgency_detection:
            return action_probs
        
        self._init_urgency_detector(env.board_size)
        
        board = env.board
        player = env.current_player
        board_size = env.board_size
        
        # ===== 开局靠近对手先验加成 =====
        total_pieces = np.sum(board != 0)
        if total_pieces <= 6:  # 前6步
            opponent = -player
            opponent_positions = np.argwhere(board == opponent)
            if len(opponent_positions) > 0:
                # 对靠近对手的位置加权
                for action in range(len(action_probs)):
                    if action_probs[action] > 0:
                        row, col = action // board_size, action % board_size
                        # 计算与最近对手棋子的距离
                        min_dist = min(max(abs(row - r), abs(col - c)) 
                                      for r, c in opponent_positions)
                        if min_dist <= 2:  # 距离对手2格以内
                            action_probs[action] *= 10.0  # 加权8倍
                        elif min_dist <= 4:  # 距离对手4格以内
                            action_probs[action] *= 3.0  # 加权3倍
        
        winning_moves, blocking_moves = self.urgency_detector.get_urgent_moves(board, player)
        
        if winning_moves:
            boost_factor = 100.0
            for action in winning_moves:
                if env.get_valid_actions()[action] > 0:
                    action_probs[action] = max(action_probs[action], 0.01) * boost_factor
        elif blocking_moves:
            boost_factor = 50.0
            for action in blocking_moves:
                if env.get_valid_actions()[action] > 0:
                    action_probs[action] = max(action_probs[action], 0.01) * boost_factor
        
        prob_sum = np.sum(action_probs)
        if prob_sum > 1e-8:
            action_probs = action_probs / prob_sum
        
        return action_probs
    
    def search(self, env, add_noise: bool = True) -> np.ndarray:
        """执行MCTS搜索"""
        # 确保根节点已扩展
        if not self.root.expanded():
            state = env._get_observation()
            action_probs, _ = self._get_valid_action_probs(env, state)
            action_probs = self._boost_urgent_priors(action_probs, env)
            self.root.expand(action_probs)
        
        if add_noise:
            self.add_dirichlet_noise(self.root)
        
        # 执行模拟
        for _ in range(self.n_simulations):
            # env_copy = copy.deepcopy(env)
            env_copy = env.copy()
            node = self.root
            
            # ===== Selection =====
            while node.expanded() and not env_copy.done:
                action, node = node.select_child(self.c_puct)
                env_copy.step(action, skip_dense=True)  # 跳过稠密奖励计算，加速10倍
            
            # ===== Evaluation =====
            if env_copy.done:
                # 终止状态：使用真实游戏结果
                # 
                # 关键理解：
                # - 游戏结束时，env_copy.current_player是"下一个要走的玩家"
                # - 但实际上游戏已经结束，没有下一步了
                # - 我们需要从node对应的"下一个要走的玩家"视角计算价值
                # - node对应的玩家就是env_copy.current_player
                #
                # 情况分析：
                # - 如果winner == env_copy.current_player: 当前玩家赢了，value = 1
                # - 如果winner == -env_copy.current_player: 对手赢了，value = -1
                # - 如果winner == 0: 平局，value = 0
                #
                # 但这里有个微妙问题：当step()检测到获胜时，current_player没有切换！
                # 所以实际上：
                # - winner就是刚刚落子的玩家
                # - env_copy.current_player仍然是刚刚落子的玩家
                # - 但node存储的应该是"执行动作后，下一个要走的玩家"的价值
                # - 下一个要走的是 -winner (如果游戏没结束的话)
                
                if env_copy.winner == 0:
                    leaf_value = 0.0
                else:
                    # winner是获胜的玩家
                    # node应该存储"下一个要走的玩家"的价值
                    # 下一个要走的是 -winner (对手)
                    # 对于对手来说，输了就是-1
                    next_player = -env_copy.winner
                    if env_copy.winner == next_player:
                        leaf_value = 1.0  # 不可能发生
                    else:
                        leaf_value = -1.0  # 对手输了
            else:
                # 非终止状态：使用神经网络评估
                state = env_copy._get_observation()
                action_probs, value = self._get_valid_action_probs(env_copy, state)
                action_probs = self._boost_urgent_priors(action_probs, env_copy)
                node.expand(action_probs)
                
                # 关键修复：不需要取反！
                # value是从env_copy.current_player视角的评估
                # node存储的正是"下一个要走的玩家"(即env_copy.current_player)的价值
                leaf_value = value
            
            # ===== Backup =====
            node.backup(leaf_value)
        
        # 返回动作概率分布
        action_visits = np.zeros(env.action_space)
        for action, child in self.root.children.items():
            action_visits[action] = child.visit_count
        
        total_visits = np.sum(action_visits)
        if total_visits > 0:
            return action_visits / total_visits
        else:
            valid = env.get_valid_actions()
            return valid / np.sum(valid)
    
    def get_action(
        self, 
        env, 
        temp: float = 1e-3, 
        add_noise: bool = True
    ) -> Tuple[int, np.ndarray]:
        """获取动作"""
        board = env.board
        board_size = env.board_size
        total_pieces = np.sum(board != 0)
        
        # ===== 开局强制规则 =====
        # AI先手（棋盘为空）：下中心
        if total_pieces == 0:
            time.sleep(0.5)  
            center = board_size // 2
            action = center * board_size + center
            action_probs = np.zeros(env.action_space)
            action_probs[action] = 1.0
            return action, action_probs
        
        # AI后手第一步（棋盘只有1子）：强制下在人类第一步周围8格内
        if total_pieces == 1:
            time.sleep(0.5)  
            # 找到人类的棋子位置
            human_pos = np.argwhere(board != 0)[0]
            hr, hc = human_pos[0], human_pos[1]
            
            # 搜集周围8格的有效位置
            candidates = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = hr + dr, hc + dc
                    if 0 <= nr < board_size and 0 <= nc < board_size:
                        if board[nr, nc] == 0:
                            candidates.append(nr * board_size + nc)
            
            if candidates:
                # 优先选择更靠近中心的位置
                center = board_size // 2
                candidates.sort(key=lambda a: max(abs(a // board_size - center), 
                                                   abs(a % board_size - center)))
                action = candidates[0]
                action_probs = np.zeros(env.action_space)
                action_probs[action] = 1.0
                return action, action_probs
        
        # 前6步（棋盘有2-5子）：强制落在已有棋子2格范围内
        if 1 < total_pieces <= 5:
            time.sleep(0.5)  
            all_positions = np.argwhere(board != 0)
            candidates = set()
            for r, c in all_positions:
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < board_size and 0 <= nc < board_size:
                            if board[nr, nc] == 0:
                                candidates.add(nr * board_size + nc)
            
            if candidates:
                # 获取网络预测的动作概率
                state = env._get_observation()
                action_probs_raw, _ = self.policy_value_fn(state)
                
                # 从候选中选择概率最高的
                best_action = max(candidates, key=lambda a: action_probs_raw[a])
                action_probs = np.zeros(env.action_space)
                action_probs[best_action] = 1.0
                return best_action, action_probs
        
        # 首先检查必胜点
        if self.use_urgency_detection:
            self._init_urgency_detector(env.board_size)
            winning_moves, _ = self.urgency_detector.get_urgent_moves(
                env.board, env.current_player
            )
            
            if winning_moves:
                action = winning_moves[0]
                action_probs = np.zeros(env.action_space)
                action_probs[action] = 1.0
                return action, action_probs
        
        # 执行MCTS搜索
        visit_probs = self.search(env, add_noise=add_noise)
        
        # 修复温度采样的数值溢出问题
        if temp < 0.01:
            # 贪婪选择
            action = int(np.argmax(visit_probs))
        else:
            # 带温度的采样
            # 使用对数空间避免溢出
            log_probs = np.log(visit_probs + 1e-10) / temp
            # 减去最大值避免exp溢出
            log_probs = log_probs - np.max(log_probs)
            temp_probs = np.exp(log_probs)
            temp_probs = temp_probs / np.sum(temp_probs)
            
            # 处理可能的NaN
            if np.any(np.isnan(temp_probs)):
                action = int(np.argmax(visit_probs))
            else:
                action = int(np.random.choice(len(temp_probs), p=temp_probs))
        
        return action, visit_probs
    
    def update_with_move(self, action: int):
        """更新根节点"""
        if action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            self.root = MCTSNode(prior=1.0)
    
    def reset(self):
        """重置搜索树"""
        self.root = MCTSNode(prior=1.0)