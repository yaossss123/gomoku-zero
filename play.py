"""
人机对战脚本 - 与训练好的AI下棋
支持多种AI: AlphaZero, 规则AI
"""
import numpy as np
from gomoku_env import GomokuEnv
from mcts import MCTS
from improved_policy_value_net import ImprovedPolicyValueAgent
from rule_based_ai import RuleBasedAI
from config import BOARD_SIZE, MCTS_SIMULATIONS_PLAY, MCTS_C_PUCT, NUM_RES_BLOCKS, NUM_CHANNELS
import os


def human_vs_alphazero(model_path: str, board_size: int = 15, human_first: bool = True):
    """
    人类 vs AlphaZero
    """
    env = GomokuEnv(board_size=board_size)
    agent = ImprovedPolicyValueAgent(board_size=board_size, num_res_blocks=NUM_RES_BLOCKS, num_channels=NUM_CHANNELS)
    
    if os.path.exists(model_path):
        agent.load_model(model_path)
        print(f"已加载模型: {model_path}")
    else:
        print(f"警告: 模型文件 {model_path} 不存在，使用未训练的模型")
    
    state = env.reset()
    
    print("\n" + "=" * 50)
    print("五子棋人机对战 - AlphaZero")
    print("=" * 50)
    print(f"棋盘大小: {board_size}x{board_size}")
    print(f"你执: {'黑棋(●)' if human_first else '白棋(○)'}")
    print(f"AI执: {'白棋(○)' if human_first else '黑棋(●)'}")
    print("\n输入格式: 行号 列号 (例如: 7 7)")
    print("输入 'q' 退出, 'u' 悔棋\n")
    
    env.render()
    
    human_player = 1 if human_first else -1
    move_history = []
    
    while not env.done:
        if env.current_player == human_player:
            print(">>> 轮到你了！")
            while True:
                try:
                    user_input = input("请输入落子位置 (行 列): ").strip()
                    
                    if user_input.lower() == 'q':
                        print("游戏结束")
                        return
                    
                    if user_input.lower() == 'u':
                        if len(move_history) >= 2:
                            # 撤销两步
                            for _ in range(2):
                                last = move_history.pop()
                                row, col = last // board_size, last % board_size
                                env.board[row, col] = 0
                            env.current_player = human_player
                            env.done = False
                            env.render()
                            print("已悔棋")
                        else:
                            print("没有可以悔的棋")
                        continue
                    
                    row, col = map(int, user_input.split())
                    
                    if row < 0 or row >= board_size or col < 0 or col >= board_size:
                        print(f"无效输入！请输入 0-{board_size-1} 之间的数字")
                        continue
                    
                    action = row * board_size + col
                    
                    if env.board[row, col] != 0:
                        print("该位置已有棋子，请重新选择！")
                        continue
                    
                    break
                except ValueError:
                    print("输入格式错误！请输入两个数字，用空格分隔")
        else:
            print(">>> AI正在思考...")
            # 使用MCTS搜索
            mcts = MCTS(
                agent.policy_value_fn, 
                n_simulations=MCTS_SIMULATIONS_PLAY,
                c_puct=MCTS_C_PUCT
            )
            action, _ = mcts.get_action(env, temp=1e-3, add_noise=False)
            row = action // board_size
            col = action % board_size
            print(f"AI落子位置: {row} {col}")
        
        # 记录历史
        move_history.append(action)
        
        # 执行动作
        state, reward, done, info = env.step(action)
        env.render()
        
        if done:
            winner = info.get('winner', 0)
            if winner == 1:
                print("黑棋(●)获胜！")
            elif winner == -1:
                print("白棋(○)获胜！")
            else:
                print("平局！")
            
            if winner == human_player:
                print("🎉 恭喜你赢了！")
            elif winner == -human_player:
                print("💪 AI获胜，继续加油！")
            break
    
    again = input("\n是否再来一局？(y/n): ").strip().lower()
    if again == 'y':
        human_vs_alphazero(model_path, board_size, human_first)


def human_vs_rule_ai(board_size: int = 15, human_first: bool = True):
    """
    人类 vs 规则AI
    """
    env = GomokuEnv(board_size=board_size)
    ai = RuleBasedAI(board_size=board_size)
    
    env.reset()
    
    print("\n" + "=" * 50)
    print("五子棋人机对战 - 规则AI")
    print("=" * 50)
    print(f"你执: {'黑棋(●)' if human_first else '白棋(○)'}")
    print("\n输入格式: 行号 列号 (例如: 7 7)")
    print("输入 'q' 退出\n")
    
    env.render()
    
    human_player = 1 if human_first else -1
    
    while not env.done:
        if env.current_player == human_player:
            print(">>> 轮到你了！")
            while True:
                try:
                    user_input = input("请输入落子位置 (行 列): ").strip()
                    
                    if user_input.lower() == 'q':
                        print("游戏结束")
                        return
                    
                    row, col = map(int, user_input.split())
                    
                    if row < 0 or row >= board_size or col < 0 or col >= board_size:
                        print(f"无效输入！请输入 0-{board_size-1} 之间的数字")
                        continue
                    
                    action = row * board_size + col
                    
                    if env.board[row, col] != 0:
                        print("该位置已有棋子！")
                        continue
                    
                    break
                except ValueError:
                    print("输入格式错误！")
        else:
            print(">>> AI正在思考...")
            action = ai.get_action(env)
            row = action // board_size
            col = action % board_size
            print(f"AI落子位置: {row} {col}")
        
        env.step(action)
        env.render()
        
        if env.done:
            winner = env.winner
            if winner == human_player:
                print("🎉 恭喜你赢了！")
            elif winner == -human_player:
                print("💪 AI获胜！")
            else:
                print("🤝 平局！")
            break
    
    again = input("\n是否再来一局？(y/n): ").strip().lower()
    if again == 'y':
        human_vs_rule_ai(board_size, human_first)


def ai_vs_ai(model_path: str, board_size: int = 15, num_games: int = 10):
    """
    AlphaZero vs 规则AI 展示
    """
    agent = ImprovedPolicyValueAgent(board_size=board_size, num_res_blocks=NUM_RES_BLOCKS, num_channels=NUM_CHANNELS)
    
    if os.path.exists(model_path):
        agent.load_model(model_path)
    else:
        print(f"警告: 模型文件 {model_path} 不存在")
        return
    
    rule_ai = RuleBasedAI(board_size=board_size)
    
    alphazero_wins = 0
    rule_wins = 0
    draws = 0
    
    print(f"\n=== AlphaZero vs 规则AI ({num_games}局) ===\n")
    
    for game in range(num_games):
        env = GomokuEnv(board_size=board_size)
        env.reset()
        
        # 交替先手
        alphazero_player = 1 if game % 2 == 0 else -1
        
        mcts = MCTS(agent.policy_value_fn, n_simulations=MCTS_SIMULATIONS_PLAY, c_puct=MCTS_C_PUCT)
        
        moves = 0
        while not env.done:
            if env.current_player == alphazero_player:
                action, _ = mcts.get_action(env, temp=1e-3, add_noise=False)
                mcts.update_with_move(action)
            else:
                action = rule_ai.get_action(env)
                mcts.update_with_move(action)
            
            env.step(action)
            moves += 1
        
        if env.winner == 0:
            draws += 1
            result = "平局"
        elif env.winner == alphazero_player:
            alphazero_wins += 1
            result = "AlphaZero胜"
        else:
            rule_wins += 1
            result = "规则AI胜"
        
        az_side = "黑" if alphazero_player == 1 else "白"
        print(f"第{game+1}局: AlphaZero执{az_side}, {result}, {moves}步")
    
    print(f"\n统计:")
    print(f"  AlphaZero胜率: {alphazero_wins/num_games:.1%}")
    print(f"  规则AI胜率: {rule_wins/num_games:.1%}")
    print(f"  平局率: {draws/num_games:.1%}")


if __name__ == '__main__':
    # 默认模型路径
    default_model = 'models_dense_reward/model_final.pth'
    
    # 备选路径
    alt_models = [
        'models_dense_reward/model_2000.pth',
        'models_dense_reward/model_1800.pth',
        'models_standard/model_final.pth',
    ]
    
    # 查找可用模型
    model_path = None
    for path in [default_model] + alt_models:
        if os.path.exists(path):
            model_path = path
            break
    
    print("=" * 50)
    print("五子棋AI对战系统")
    print("=" * 50)
    print("\n选择模式:")
    print("1. 人类 vs AlphaZero")
    print("2. 人类 vs 规则AI")
    print("3. AlphaZero vs 规则AI (展示)")
    
    choice = input("\n请选择 (1/2/3): ").strip()
    
    if choice == '1':
        if model_path is None:
            print("未找到模型文件，将使用未训练的模型")
            model_path = default_model
        first = input("你想先手吗？(y/n): ").strip().lower() == 'y'
        human_vs_alphazero(model_path, board_size=BOARD_SIZE, human_first=first)
    elif choice == '2':
        first = input("你想先手吗？(y/n): ").strip().lower() == 'y'
        human_vs_rule_ai(board_size=BOARD_SIZE, human_first=first)
    elif choice == '3':
        if model_path is None:
            print("未找到模型文件")
        else:
            num = int(input("对弈局数 (默认10): ").strip() or "10")
            ai_vs_ai(model_path, board_size=BOARD_SIZE, num_games=num)
    else:
        print("无效选择")