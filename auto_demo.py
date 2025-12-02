"""
自动演示脚本 - 完整展示强化学习训练流程
"""
from gomoku_env import GomokuEnv
from dqn_agent import DQNAgent
import time


def print_section(title):
    """打印章节标题"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")


def demo_phase1_untrained():
    """第一阶段：展示未训练AI"""
    print_section("📚 第一阶段：理解环境 - 未训练AI的表现")
    
    print("💡 核心概念：")
    print("   - 状态(State): 棋盘配置 (15x15, 3通道)")
    print("   - 动作(Action): 在空位落子 (225个可能位置)")
    print("   - 奖励(Reward): 赢+1, 输-1, 平0")
    print("   - 策略(Policy): 当前是随机策略 (ε=1.0)\n")
    
    input("按Enter键观看未训练AI的对局...")
    
    env = GomokuEnv(board_size=9)  # 使用9x9小棋盘，更快演示
    agent = DQNAgent(board_size=9)
    agent.epsilon = 1.0  # 完全随机
    
    state = env.reset()
    done = False
    moves = 0
    
    print("\n初始棋盘 (9x9):")
    env.render()
    
    while not done and moves < 30:
        valid_actions = env.get_valid_actions()
        action = agent.select_action(state, valid_actions, training=True)
        
        row = action // env.board_size
        col = action % env.board_size
        player = "●" if env.current_player == 1 else "○"
        
        print(f"第{moves+1}步: {player} → ({row},{col})")
        state, _, done, info = env.step(action)
        
        if (moves + 1) % 5 == 0 or done:  # 每5步显示一次棋盘
            env.render()
        
        moves += 1
        time.sleep(0.3)  # 慢速演示
    
    if done:
        winner = info.get('winner', 0)
        if winner == 1:
            print("\n🏆 结果: 黑棋(●)获胜")
        elif winner == -1:
            print("\n🏆 结果: 白棋(○)获胜")
        else:
            print("\n🤝 结果: 平局")
    
    print(f"\n总步数: {moves}")
    print("\n📊 观察结果: AI完全随机下棋，没有任何策略")


def demo_phase2_training():
    """第二阶段：快速训练演示"""
    print_section("🎓 第二阶段：开始训练 - DQN算法学习过程")
    
    print("💡 DQN核心技术：")
    print("   1. 经验回放(Experience Replay): 存储对局经验，打破数据相关性")
    print("   2. 目标网络(Target Network): 每100步更新，稳定训练")
    print("   3. ε-贪婪策略: ε从1.0衰减到0.01，平衡探索与利用")
    print("   4. 神经网络: 3层卷积层提取棋盘特征\n")
    
    input("按Enter键开始快速训练 (100局演示)...")
    
    env = GomokuEnv(board_size=9)
    agent = DQNAgent(board_size=9, epsilon_decay=0.98)
    
    wins_black = 0
    wins_white = 0
    draws = 0
    
    print("\n开始训练...")
    print("局数  黑胜  白胜  平局  Epsilon  损失")
    print("-" * 50)
    
    for episode in range(100):
        state = env.reset()
        done = False
        episode_data = []
        
        # 自我对弈
        while not done:
            valid_actions = env.get_valid_actions()
            action = agent.select_action(state, valid_actions, training=True)
            next_state, reward, done, info = env.step(action)
            
            episode_data.append((state, action, reward, next_state, done))
            state = next_state
        
        # 统计胜负
        winner = info.get('winner', 0)
        if winner == 1:
            wins_black += 1
        elif winner == -1:
            wins_white += 1
        else:
            draws += 1
        
        # 添加经验并训练
        for exp_state, exp_action, exp_reward, exp_next_state, exp_done in episode_data:
            agent.replay_buffer.push(exp_state, exp_action, exp_reward, exp_next_state, exp_done)
        
        loss = agent.train_step()
        agent.update_epsilon()
        
        # 每10局显示一次
        if (episode + 1) % 10 == 0:
            total = wins_black + wins_white + draws
            loss_str = f"{loss:.4f}" if loss else "N/A"
            print(f"{episode+1:4d}  {wins_black:4d}  {wins_white:4d}  {draws:3d}   {agent.epsilon:.4f}  {loss_str}")
            wins_black = wins_white = draws = 0
    
    print("\n✅ 训练完成！")
    print(f"最终 Epsilon: {agent.epsilon:.4f}")
    print(f"经验池大小: {len(agent.replay_buffer)}")
    
    return agent


def demo_phase3_comparison(trained_agent):
    """第三阶段：对比训练前后"""
    print_section("🎯 第三阶段：效果对比 - 训练后AI的表现")
    
    print("现在让训练后的AI进行对局 (ε=0, 纯利用策略)\n")
    input("按Enter键观看训练后AI的对局...")
    
    env = GomokuEnv(board_size=9)
    trained_agent.epsilon = 0  # 纯利用
    
    state = env.reset()
    done = False
    moves = 0
    
    print("\n初始棋盘:")
    env.render()
    
    while not done and moves < 30:
        valid_actions = env.get_valid_actions()
        action = trained_agent.select_action(state, valid_actions, training=False)
        
        row = action // env.board_size
        col = action % env.board_size
        player = "●" if env.current_player == 1 else "○"
        
        print(f"第{moves+1}步: {player} → ({row},{col})")
        state, _, done, info = env.step(action)
        
        if (moves + 1) % 5 == 0 or done:
            env.render()
        
        moves += 1
        time.sleep(0.3)
    
    if done:
        winner = info.get('winner', 0)
        if winner == 1:
            print("\n🏆 结果: 黑棋(●)获胜")
        elif winner == -1:
            print("\n🏆 结果: 白棋(○)获胜")
        else:
            print("\n🤝 结果: 平局")
    
    print(f"\n总步数: {moves}")
    print("\n📊 观察结果: 训练后的AI会尝试形成连子，展现出策略性")


def final_summary():
    """总结"""
    print_section("🎉 学习总结")
    
    print("✅ 您已经体验了强化学习的完整流程：\n")
    print("1. 环境理解: 五子棋状态、动作、奖励")
    print("2. 训练过程: 自我对弈 + DQN学习")
    print("3. 策略改进: 从随机到有策略\n")
    
    print("📚 核心概念回顾：")
    print("   - Q函数: Q(s,a) = 在状态s执行动作a的期望回报")
    print("   - 经验回放: 重复利用历史经验，提高样本效率")
    print("   - 目标网络: 减少训练过程中的震荡")
    print("   - ε-贪婪: 探索新策略 vs 利用已知最优策略\n")
    
    print("🚀 下一步建议：\n")
    print("1. 完整训练: python train.py  (训练5000局)")
    print("2. 人机对战: python play.py   (与训练好的AI下棋)")
    print("3. 参数调优: 修改学习率、网络结构等")
    print("4. 算法升级: 尝试 Double DQN、Dueling DQN\n")
    
    print("="*70)
    print("🎓 恭喜完成强化学习五子棋入门！")
    print("="*70)


if __name__ == '__main__':
    print("\n" + "="*70)
    print("  🤖 强化学习五子棋 - 自动化完整演示")
    print("="*70)
    print("\n本演示将自动展示：")
    print("  1. 未训练AI的随机表现")
    print("  2. 快速训练过程 (100局)")
    print("  3. 训练后AI的策略表现")
    print("  4. 学习总结与下一步指导\n")
    
    input("准备好了吗？按Enter键开始...")
    
    # 执行三个阶段
    demo_phase1_untrained()
    trained_agent = demo_phase2_training()
    demo_phase3_comparison(trained_agent)
    final_summary()
    
    print("\n感谢体验！开始您的强化学习之旅吧！🚀")
