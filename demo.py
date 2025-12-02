"""
演示脚本 - 自动展示AI自我对弈
"""
from gomoku_env import GomokuEnv
from dqn_agent import DQNAgent
import os


def demo_untrained_ai():
    """展示未训练AI的自我对弈"""
    print("\n=== 演示：未训练AI的自我对弈 ===")
    print("这是AI在未经训练时的表现（随机下棋）\n")
    
    env = GomokuEnv(board_size=15)
    agent = DQNAgent(board_size=15)
    agent.epsilon = 1.0  # 完全随机
    
    state = env.reset()
    done = False
    moves = 0
    
    print("初始棋盘:")
    env.render()
    
    while not done and moves < 50:  # 限制最多50步，避免太长
        valid_actions = env.get_valid_actions()
        action = agent.select_action(state, valid_actions, training=True)
        
        row = action // env.board_size
        col = action % env.board_size
        player = "黑棋(●)" if env.current_player == 1 else "白棋(○)"
        
        print(f"\n第 {moves + 1} 步: {player} 落子位置 ({row}, {col})")
        
        state, _, done, info = env.step(action)
        env.render()
        
        moves += 1
        
        if done:
            winner = info.get('winner', 0)
            if winner == 1:
                print("\n🏆 黑棋(●)获胜！")
            elif winner == -1:
                print("\n🏆 白棋(○)获胜！")
            else:
                print("\n🤝 平局！")
            print(f"总步数: {moves}")
            break
    
    if not done:
        print("\n演示结束（限制50步）")
    
    print("\n" + "="*50)
    print("💡 观察：未训练的AI是随机下棋，没有策略")
    print("="*50)


def explain_learning_path():
    """说明学习路径"""
    print("\n" + "="*60)
    print("📚 强化学习五子棋 - 学习路径")
    print("="*60)
    
    print("\n✅ 第一阶段：理解环境（当前阶段）")
    print("   1. 观察未训练AI的表现（随机策略）")
    print("   2. 理解五子棋规则和状态表示")
    print("   3. 理解奖励机制：赢+1，输-1")
    
    print("\n📝 第二阶段：开始训练")
    print("   1. 运行训练脚本（自我对弈）")
    print("   2. 观察训练曲线和胜率变化")
    print("   3. 理解DQN的核心概念：")
    print("      - 经验回放：存储历史对局")
    print("      - 目标网络：稳定训练")
    print("      - ε-贪婪：探索vs利用")
    
    print("\n🎮 第三阶段：测试与改进")
    print("   1. 与训练后的AI对战")
    print("   2. 调整超参数优化性能")
    print("   3. 尝试不同的网络结构")
    
    print("\n" + "="*60)


if __name__ == '__main__':
    # 说明学习路径
    explain_learning_path()
    
    # 询问是否继续
    choice = input("\n是否观看未训练AI的演示对局？(y/n): ").strip().lower()
    
    if choice == 'y':
        demo_untrained_ai()
        
        print("\n\n下一步操作：")
        print("1. 开始训练: python train.py")
        print("2. 人机对战: python play.py")
        print("\n建议：先运行训练，让AI学习策略，然后再进行人机对战")
