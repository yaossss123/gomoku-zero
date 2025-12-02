"""
AI对战评估
测试不同AI之间的胜率
"""
import numpy as np
from gomoku_env import GomokuEnv
from mcts import MCTS
from policy_value_net import PolicyValueAgent
from rule_based_ai import RuleBasedAI, RandomAI
from dqn_agent import DQNAgent
from tqdm import tqdm


def evaluate_ai_vs_ai(ai1, ai2, n_games=100, board_size=15, ai1_name="AI1", ai2_name="AI2"):
    """
    AI对战评估
    Args:
        ai1: AI1
        ai2: AI2  
        n_games: 对战局数
        board_size: 棋盘大小
        ai1_name: AI1名称
        ai2_name: AI2名称
    Returns:
        results: 统计结果
    """
    env = GomokuEnv(board_size=board_size)
    
    wins_ai1 = 0
    wins_ai2 = 0
    draws = 0
    
    print(f"\n{'='*60}")
    print(f"{ai1_name} vs {ai2_name} - {n_games}局对战")
    print(f"{'='*60}\n")
    
    for game in tqdm(range(n_games), desc="对战进度"):
        state = env.reset()
        done = False
        
        # 轮流先手
        if game % 2 == 0:
            current_ai = ai1
            ai_black = ai1_name
            ai_white = ai2_name
        else:
            current_ai = ai2
            ai_black = ai2_name
            ai_white = ai1_name
        
        moves = 0
        while not done and moves < 225:  # 最多225步
            # 获取动作
            if isinstance(current_ai, PolicyValueAgent):
                # AlphaZero AI (with MCTS)
                mcts = MCTS(current_ai.policy_value_fn, n_simulations=200)
                action, _ = mcts.get_action(env, temp=1e-3)
            elif isinstance(current_ai, DQNAgent):
                # DQN AI
                valid_actions = env.get_valid_actions()
                action = current_ai.select_action(state, valid_actions, training=False)
            else:
                # Rule-based or Random AI
                action = current_ai.get_action(env)
            
            # 执行动作
            state, _, done, info = env.step(action)
            moves += 1
            
            # 切换AI
            current_ai = ai2 if current_ai == ai1 else ai1
        
        # 统计结果
        if done:
            winner = info.get('winner', 0)
            if winner == 1:  # 黑棋胜
                if ai_black == ai1_name:
                    wins_ai1 += 1
                else:
                    wins_ai2 += 1
            elif winner == -1:  # 白棋胜
                if ai_white == ai1_name:
                    wins_ai1 += 1
                else:
                    wins_ai2 += 1
            else:
                draws += 1
        else:
            draws += 1
    
    # 打印结果
    print(f"\n{'='*60}")
    print("对战结果:")
    print(f"  {ai1_name} 胜: {wins_ai1} ({wins_ai1/n_games*100:.1f}%)")
    print(f"  {ai2_name} 胜: {wins_ai2} ({wins_ai2/n_games*100:.1f}%)")
    print(f"  平局: {draws} ({draws/n_games*100:.1f}%)")
    print(f"{'='*60}\n")
    
    return {
        'ai1_wins': wins_ai1,
        'ai2_wins': wins_ai2,
        'draws': draws,
        'ai1_winrate': wins_ai1 / n_games,
        'ai2_winrate': wins_ai2 / n_games
    }


def main():
    """主评估程序"""
    board_size = 15
    
    print("\n" + "="*60)
    print("五子棋AI评估系统")
    print("="*60)
    
    # 加载AI模型
    print("\n正在加载AI模型...")
    
    # 1. AlphaZero AI
    try:
        alphazero_ai = PolicyValueAgent(board_size=board_size)
        alphazero_ai.load_model('models_alphazero/alphazero_final.pth')
        has_alphazero = True
        print("✓ AlphaZero模型加载成功")
    except:
        has_alphazero = False
        print("✗ AlphaZero模型未找到")
    
    # 2. DQN AI
    try:
        dqn_ai = DQNAgent(board_size=board_size)
        dqn_ai.load('models/gomoku_dqn_final.pth')
        has_dqn = True
        print("✓ DQN模型加载成功")
    except:
        has_dqn = False
        print("✗ DQN模型未找到")
    
    # 3. Rule-based AI
    rule_ai = RuleBasedAI(board_size=board_size)
    print("✓ 规则AI已创建")
    
    # 4. Random AI
    random_ai = RandomAI()
    print("✓ 随机AI已创建")
    
    # 进行对战评估
    print("\n" + "="*60)
    print("开始评估...")
    print("="*60)
    
    n_games = 20  # 每组对战局数
    
    # 评估1: AlphaZero vs 规则AI
    if has_alphazero:
        evaluate_ai_vs_ai(alphazero_ai, rule_ai, n_games, board_size, 
                         "AlphaZero", "规则AI")
    
    # 评估2: DQN vs 规则AI  
    if has_dqn:
        evaluate_ai_vs_ai(dqn_ai, rule_ai, n_games, board_size,
                         "DQN", "规则AI")
    
    # 评估3: 规则AI vs 随机AI
    evaluate_ai_vs_ai(rule_ai, random_ai, n_games, board_size,
                     "规则AI", "随机AI")
    
    # 评估4: AlphaZero vs DQN (如果都有)
    if has_alphazero and has_dqn:
        evaluate_ai_vs_ai(alphazero_ai, dqn_ai, n_games, board_size,
                         "AlphaZero", "DQN")
    
    print("\n评估完成！")


if __name__ == '__main__':
    main()
