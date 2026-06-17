# 创建文件: evaluate_model.py

"""
模型评估和测试脚本
"""
import os
from gomoku_env import GomokuEnv
from mcts import MCTS
from improved_policy_value_net import ImprovedPolicyValueAgent
from rule_based_ai import RuleBasedAI
from config import BOARD_SIZE, MCTS_SIMULATIONS_EVAL, MCTS_C_PUCT, NUM_RES_BLOCKS, NUM_CHANNELS


def evaluate_vs_rule_ai(model_path: str, n_games: int = 20, n_simulations: int = MCTS_SIMULATIONS_EVAL):
    """评估模型对规则AI的胜率"""
    print(f"\n{'='*50}")
    print(f"评估模型: {model_path}")
    print(f"{'='*50}")
    
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return
    
    agent = ImprovedPolicyValueAgent(board_size=BOARD_SIZE, num_res_blocks=NUM_RES_BLOCKS, num_channels=NUM_CHANNELS)
    agent.load_model(model_path)
    
    rule_ai = RuleBasedAI(board_size=BOARD_SIZE)
    
    wins, losses, draws = 0, 0, 0
    
    print(f"\n对战 {n_games} 局中...")
    
    for game_idx in range(n_games):
        env = GomokuEnv(board_size=BOARD_SIZE)
        env.reset()
        
        # 交替先手
        alphazero_player = 1 if game_idx % 2 == 0 else -1
        
        mcts = MCTS(
            policy_value_fn=agent.policy_value_fn,
            n_simulations=n_simulations,
            c_puct=MCTS_C_PUCT
        )
        
        while not env.done:
            if env.current_player == alphazero_player:
                action, _ = mcts.get_action(env, temp=1e-3, add_noise=False)
                mcts.update_with_move(action)
            else:
                action = rule_ai.get_action(env)
                mcts.update_with_move(action)
            
            env.step(int(action), skip_dense=True)
        
        if env.winner == 0:
            draws += 1
            result = "平"
        elif env.winner == alphazero_player:
            wins += 1
            result = "胜"
        else:
            losses += 1
            result = "负"
        
        side = "黑" if alphazero_player == 1 else "白"
        print(f"  第{game_idx+1:2d}局: AI执{side}, {result}")
    
    print(f"\n{'='*50}")
    print(f"结果统计:")
    print(f"  胜: {wins} ({wins/n_games:.1%})")
    print(f"  负: {losses} ({losses/n_games:.1%})")
    print(f"  平: {draws} ({draws/n_games:.1%})")
    print(f"{'='*50}")
    
    return wins / n_games


def quick_play_test(model_path: str):
    """快速人机对战测试"""
    print(f"\n启动人机对战测试...")
    
    if not os.path.exists(model_path):
        print(f"模型文件不存在: {model_path}")
        return
    
    import subprocess
    import sys
    subprocess.run([sys.executable, "play.py"])


def main():
    # 查找可用模型
    model_dirs = ['models_dense_reward', 'models_standard']
    models = []
    
    for model_dir in model_dirs:
        if os.path.exists(model_dir):
            for f in os.listdir(model_dir):
                if f.endswith('.pth'):
                    models.append(os.path.join(model_dir, f))
    
    if not models:
        print("未找到任何模型文件！")
        return
    
    print("\n" + "="*50)
    print("模型评估工具")
    print("="*50)
    
    print("\n可用模型:")
    for i, m in enumerate(models):
        print(f"  {i+1}. {m}")
    
    print("\n选择操作:")
    print("  1. 评估模型 vs 规则AI")
    print("  2. 人机对战")
    print("  3. 评估所有模型")
    
    choice = input("\n请选择 (1/2/3): ").strip()
    
    if choice == '1':
        idx = int(input("选择模型编号: ").strip()) - 1
        if 0 <= idx < len(models):
            n_games = int(input("对战局数 (默认20): ").strip() or "20")
            evaluate_vs_rule_ai(models[idx], n_games=n_games)
    
    elif choice == '2':
        idx = int(input("选择模型编号: ").strip()) - 1
        if 0 <= idx < len(models):
            # 设置环境变量让play.py使用指定模型
            os.environ['MODEL_PATH'] = models[idx]
            quick_play_test(models[idx])
    
    elif choice == '3':
        print("\n评估所有模型...")
        results = []
        for model in models:
            win_rate = evaluate_vs_rule_ai(model, n_games=10)
            results.append((model, win_rate))
        
        print("\n" + "="*50)
        print("所有模型评估结果:")
        print("="*50)
        for model, rate in sorted(results, key=lambda x: x[1], reverse=True):
            print(f"  {rate:.1%} - {model}")


if __name__ == '__main__':
    main()