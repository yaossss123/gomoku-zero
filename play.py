"""
人机对战脚本 - 与训练好的AI下棋
支持多种AI: DQN, AlphaZero, 规则AI
"""
import numpy as np
from gomoku_env import GomokuEnv
from dqn_agent import DQNAgent
from mcts import MCTS
from policy_value_net import PolicyValueAgent
from rule_based_ai import RuleBasedAI
import os


def human_vs_ai(model_path, board_size=15, human_first=True):
    """
    人机对战
    Args:
        model_path: AI模型路径
        board_size: 棋盘大小
        human_first: 人类是否先手
    """
    # 创建环境和智能体
    env = GomokuEnv(board_size=board_size)
    agent = DQNAgent(board_size=board_size)
    
    # 加载模型
    if os.path.exists(model_path):
        agent.load(model_path)
        agent.epsilon = 0  # 对弈时不使用随机探索
    else:
        print(f"警告: 模型文件 {model_path} 不存在，使用未训练的模型")
    
    state = env.reset()
    done = False
    
    print("\n=== 五子棋人机对战 ===")
    print(f"棋盘大小: {board_size}x{board_size}")
    print(f"你执: {'黑棋(●)' if human_first else '白棋(○)'}")
    print(f"AI执: {'白棋(○)' if human_first else '黑棋(●)'}")
    print("\n输入格式: 行号 列号 (例如: 7 7 表示中心位置)")
    print("输入 'q' 退出游戏\n")
    
    env.render()
    
    is_human_turn = human_first
    
    while not done:
        if is_human_turn:
            # 人类回合
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
                        print("该位置已有棋子，请重新选择！")
                        continue
                    
                    break
                except ValueError:
                    print("输入格式错误！请输入两个数字，用空格分隔")
                except Exception as e:
                    print(f"错误: {e}")
        else:
            # AI回合
            print(">>> AI正在思考...")
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions, training=False)
            row = action // board_size
            col = action % board_size
            print(f"AI落子位置: {row} {col}")
        
        # 执行动作
        next_state, reward, done, info = env.step(action)
        state = next_state
        
        # 显示棋盘
        env.render()
        
        # 检查游戏是否结束
        if done:
            if 'illegal_move' in info:
                print("检测到非法落子！")
            
            winner = info.get('winner', 0)
            if winner == 1:
                print("黑棋(●)获胜！")
            elif winner == -1:
                print("白棋(○)获胜！")
            else:
                print("平局！")
            
            if human_first:
                if winner == 1:
                    print("恭喜你赢了！🎉")
                elif winner == -1:
                    print("AI获胜，继续加油！💪")
            else:
                if winner == -1:
                    print("恭喜你赢了！🎉")
                elif winner == 1:
                    print("AI获胜，继续加油！💪")
            break
        
        # 切换回合
        is_human_turn = not is_human_turn
    
    # 询问是否再来一局
    again = input("\n是否再来一局？(y/n): ").strip().lower()
    if again == 'y':
        human_vs_ai(model_path, board_size, human_first)


def ai_vs_ai(model_path, board_size=15, num_games=10):
    """
    AI自我对弈展示
    Args:
        model_path: AI模型路径
        board_size: 棋盘大小
        num_games: 对弈局数
    """
    env = GomokuEnv(board_size=board_size)
    agent = DQNAgent(board_size=board_size)
    
    if os.path.exists(model_path):
        agent.load(model_path)
        agent.epsilon = 0
    else:
        print(f"警告: 模型文件 {model_path} 不存在")
        return
    
    wins_black = 0
    wins_white = 0
    draws = 0
    
    print(f"\n=== AI自我对弈展示 ({num_games}局) ===\n")
    
    for game in range(num_games):
        state = env.reset()
        done = False
        moves = 0
        
        print(f"第 {game + 1} 局:")
        
        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions, training=False)
            state, _, done, info = env.step(action)
            moves += 1
        
        winner = info.get('winner', 0)
        if winner == 1:
            wins_black += 1
            result = "黑棋(●)获胜"
        elif winner == -1:
            wins_white += 1
            result = "白棋(○)获胜"
        else:
            draws += 1
            result = "平局"
        
        print(f"  结果: {result}, 步数: {moves}")
    
    print(f"\n统计:")
    print(f"  黑棋胜率: {wins_black/num_games:.2%}")
    print(f"  白棋胜率: {wins_white/num_games:.2%}")
    print(f"  平局率: {draws/num_games:.2%}")


if __name__ == '__main__':
    model_path = 'models/gomoku_dqn_final.pth'
    
    print("选择模式:")
    print("1. 人机对战")
    print("2. AI自我对弈展示")
    
    choice = input("请选择 (1/2): ").strip()
    
    if choice == '1':
        first = input("你想先手吗？(y/n): ").strip().lower() == 'y'
        human_vs_ai(model_path, board_size=15, human_first=first)
    elif choice == '2':
        num = int(input("对弈局数 (默认10): ").strip() or "10")
        ai_vs_ai(model_path, board_size=15, num_games=num)
    else:
        print("无效选择")
