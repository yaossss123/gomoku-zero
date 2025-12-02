"""
蒙特卡洛树搜索 (MCTS) 实现
结合策略网络和价值网络的AlphaZero风格MCTS
"""
import numpy as np
import math
from typing import Tuple, Optional


class MCTSNode:
    """MCTS树节点"""
    
    def __init__(self, prior: float, parent=None):
        """
        初始化节点
        Args:
            prior: 先验概率（来自策略网络）
            parent: 父节点
        """
        self.parent = parent
        self.children = {}  # 动作 -> 子节点
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
    
    def expanded(self):
        """是否已扩展"""
        return len(self.children) > 0
    
    def value(self):
        """节点平均价值"""
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
    
    def select_child(self, c_puct: float = 1.0):
        """
        选择最优子节点（UCB公式）
        Args:
            c_puct: 探索常数
        Returns:
            (action, child_node)
        """
        best_score = -float('inf')
        best_action = -1
        best_child = None
        
        for action, child in self.children.items():
            # UCB score = Q + U
            # Q: 平均价值
            # U: 探索奖励，基于先验概率和访问次数
            score = child.value() + c_puct * child.prior * math.sqrt(self.visit_count) / (1 + child.visit_count)
            
            if score > best_score:
                best_score = score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def expand(self, action_probs):
        """
        扩展节点
        Args:
            action_probs: 动作概率分布（来自策略网络）
        """
        for action, prob in enumerate(action_probs):
            if prob > 0:
                self.children[action] = MCTSNode(prior=prob, parent=self)
    
    def update(self, value: float):
        """
        反向传播更新节点价值
        Args:
            value: 叶子节点评估值
        """
        self.visit_count += 1
        self.value_sum += value
    
    def update_recursive(self, value: float):
        """递归更新到根节点"""
        self.update(value)
        if self.parent:
            self.parent.update_recursive(-value)  # 对手视角，价值取反


class MCTS:
    """蒙特卡洛树搜索"""
    
    def __init__(self, policy_value_fn, c_puct: float = 1.0, n_simulations: int = 400):
        """
        初始化MCTS
        Args:
            policy_value_fn: 策略价值函数 (state) -> (action_probs, value)
            c_puct: UCB探索常数
            n_simulations: 模拟次数
        """
        self.policy_value_fn = policy_value_fn
        self.c_puct = c_puct
        self.n_simulations = n_simulations
        self.root = MCTSNode(prior=1.0, parent=None)
    
    def search(self, env):
        """
        执行MCTS搜索
        Args:
            env: 游戏环境
        Returns:
            action_probs: 动作概率分布
        """
        for _ in range(self.n_simulations):
            # 复制环境以进行模拟
            env_copy = self._copy_env(env)
            node = self.root
            search_path = [node]
            
            # 1. 选择：沿着树向下选择到叶子节点
            while node.expanded():
                action, node = node.select_child(self.c_puct)
                search_path.append(node)
                # 检查游戏是否已经结束
                if not env_copy.done:
                    env_copy.step(action)
                else:
                    break
            
            # 获取当前状态
            state = env_copy._get_observation()
            valid_actions = env_copy.get_valid_actions()
            
            # 检查游戏是否结束
            if env_copy.done:
                # 游戏结束，使用真实结果
                if env_copy.winner == env_copy.current_player:
                    value = 1.0
                elif env_copy.winner == -env_copy.current_player:
                    value = -1.0
                else:
                    value = 0.0
            else:
                # 2. 扩展：使用策略网络获取先验概率
                action_probs, value = self.policy_value_fn(state)
                
                # 只保留合法动作的概率
                action_probs = action_probs * valid_actions
                prob_sum = np.sum(action_probs)
                if prob_sum > 0:
                    action_probs = action_probs / prob_sum
                else:
                    # 如果没有合法动作，使用均匀分布
                    action_probs = valid_actions / np.sum(valid_actions)
                
                # 扩展节点
                node.expand(action_probs)
            
            # 4. 反向传播：更新路径上所有节点的价值
            for node in reversed(search_path):
                node.update(value)
                value = -value  # 切换玩家视角
        
        # 返回根节点的访问计数作为动作概率
        action_visits = np.zeros(env.action_space)
        for action, child in self.root.children.items():
            action_visits[action] = child.visit_count
        
        # 温度参数控制探索程度
        return action_visits / np.sum(action_visits)
    
    def get_action(self, env, temp: float = 1e-3):
        """
        获取最优动作
        Args:
            env: 游戏环境
            temp: 温度参数（越小越确定性）
        Returns:
            action: 选择的动作
            action_probs: 动作概率分布
        """
        action_probs = self.search(env)
        
        if temp < 1e-3:
            # 确定性选择（选择访问次数最多的）
            action = np.argmax(action_probs)
        else:
            # 按概率采样
            action = np.random.choice(len(action_probs), p=action_probs)
        
        return action, action_probs
    
    def update_with_move(self, action):
        """
        更新根节点（重用搜索树）
        Args:
            action: 执行的动作
        """
        if action in self.root.children:
            self.root = self.root.children[action]
            self.root.parent = None
        else:
            self.root = MCTSNode(prior=1.0, parent=None)
    
    def _copy_env(self, env):
        """复制环境用于模拟"""
        import copy
        return copy.deepcopy(env)
