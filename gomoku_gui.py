"""
五子棋图形界面 (GUI)
使用Tkinter实现可视化人机对战
"""
import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
from gomoku_env import GomokuEnv
from dqn_agent import DQNAgent
from mcts import MCTS
from policy_value_net import PolicyValueAgent
from rule_based_ai import RuleBasedAI
import os


class GomokuGUI:
    """五子棋图形界面"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("五子棋AI对战")
        self.root.resizable(False, False)
        
        # 游戏参数
        self.board_size = 15
        self.cell_size = 40
        self.piece_radius = 15
        
        # 颜色配置
        self.board_color = "#DEB887"
        self.line_color = "#000000"
        self.black_color = "#000000"
        self.white_color = "#FFFFFF"
        self.last_move_color = "#FF0000"
        
        # 游戏状态
        self.env = GomokuEnv(board_size=self.board_size)
        self.game_over = False
        self.human_color = 1  # 1: 黑棋, -1: 白棋
        self.ai_agent = None
        self.ai_type = "DQN"
        self.last_move = None
        
        # 悔棋历史记录
        self.move_history = []  # [(action, player, board_state), ...]
        
        # 创建界面
        self.create_widgets()
        
    def create_widgets(self):
        """创建界面组件"""
        # 顶部控制面板
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)
        
        # AI选择
        tk.Label(control_frame, text="选择AI:").grid(row=0, column=0, padx=5)
        self.ai_var = tk.StringVar(value="DQN")
        ai_choices = ["DQN", "AlphaZero", "规则AI"]
        ai_menu = ttk.Combobox(control_frame, textvariable=self.ai_var, 
                              values=ai_choices, state="readonly", width=12)
        ai_menu.grid(row=0, column=1, padx=5)
        
        # 先后手选择
        tk.Label(control_frame, text="你执:").grid(row=0, column=2, padx=5)
        self.color_var = tk.StringVar(value="黑棋")
        color_menu = ttk.Combobox(control_frame, textvariable=self.color_var,
                                 values=["黑棋", "白棋"], state="readonly", width=8)
        color_menu.grid(row=0, column=3, padx=5)
        
        # 按钮
        tk.Button(control_frame, text="开始游戏", command=self.start_game,
                 bg="#4CAF50", fg="white", width=10).grid(row=0, column=4, padx=5)
        tk.Button(control_frame, text="重新开始", command=self.restart_game,
                 bg="#2196F3", fg="white", width=10).grid(row=0, column=5, padx=5)
        tk.Button(control_frame, text="悔棋", command=self.undo_move,
                 bg="#FF9800", fg="white", width=10).grid(row=0, column=6, padx=5)
        
        # 棋盘画布
        canvas_size = self.cell_size * (self.board_size + 1)
        self.canvas = tk.Canvas(self.root, width=canvas_size, height=canvas_size,
                               bg=self.board_color, highlightthickness=0)
        self.canvas.pack(padx=20, pady=10)
        self.canvas.bind("<Button-1>", self.on_click)
        
        # 绘制棋盘
        self.draw_board()
        
        # 底部状态栏
        self.status_label = tk.Label(self.root, text="请点击'开始游戏'",
                                     font=("Arial", 12), fg="#2196F3")
        self.status_label.pack(pady=10)
        
    def draw_board(self):
        """绘制棋盘网格"""
        self.canvas.delete("all")
        
        # 绘制网格线
        for i in range(self.board_size):
            # 横线
            x1 = self.cell_size
            y1 = self.cell_size * (i + 1)
            x2 = self.cell_size * self.board_size
            y2 = y1
            self.canvas.create_line(x1, y1, x2, y2, fill=self.line_color, width=1)
            
            # 竖线
            x1 = self.cell_size * (i + 1)
            y1 = self.cell_size
            x2 = x1
            y2 = self.cell_size * self.board_size
            self.canvas.create_line(x1, y1, x2, y2, fill=self.line_color, width=1)
        
        # 绘制星位
        star_positions = [(3, 3), (3, 11), (11, 3), (11, 11), (7, 7)]
        for row, col in star_positions:
            x = self.cell_size * (col + 1)
            y = self.cell_size * (row + 1)
            self.canvas.create_oval(x-3, y-3, x+3, y+3, fill=self.line_color)
        
        # 绘制坐标标签
        for i in range(self.board_size):
            # 行号
            self.canvas.create_text(self.cell_size // 2, self.cell_size * (i + 1),
                                   text=str(i), font=("Arial", 10))
            # 列号
            self.canvas.create_text(self.cell_size * (i + 1), self.cell_size // 2,
                                   text=str(i), font=("Arial", 10))
    
    def draw_piece(self, row, col, color, is_last=False):
        """
        绘制棋子
        Args:
            row, col: 棋子位置
            color: 1黑棋, -1白棋
            is_last: 是否为最后一步
        """
        x = self.cell_size * (col + 1)
        y = self.cell_size * (row + 1)
        
        piece_color = self.black_color if color == 1 else self.white_color
        outline_color = self.white_color if color == 1 else self.black_color
        
        # 绘制棋子
        self.canvas.create_oval(
            x - self.piece_radius, y - self.piece_radius,
            x + self.piece_radius, y + self.piece_radius,
            fill=piece_color, outline=outline_color, width=2
        )
        
        # 标记最后一步
        if is_last:
            marker_radius = 5
            marker_color = self.last_move_color
            self.canvas.create_oval(
                x - marker_radius, y - marker_radius,
                x + marker_radius, y + marker_radius,
                fill=marker_color, outline=marker_color
            )
    
    def redraw_board(self):
        """重绘整个棋盘"""
        self.draw_board()
        
        # 绘制所有棋子
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.env.board[row, col] != 0:
                    is_last = (self.last_move == (row, col))
                    self.draw_piece(row, col, self.env.board[row, col], is_last)
    
    def start_game(self):
        """开始新游戏"""
        # 加载AI
        self.ai_type = self.ai_var.get()
        self.update_status(f"正在加载{self.ai_type}...")
        self.root.update()
        
        try:
            if self.ai_type == "DQN":
                self.ai_agent = DQNAgent(board_size=self.board_size)
                self.ai_agent.load('models/gomoku_dqn_final.pth')
            elif self.ai_type == "AlphaZero":
                # 优先加载改进版模型
                model_files = [
                    'models_improved_alphazero/alphazero_model_20000.pth',
                    'models_improved_alphazero/alphazero_model_15000.pth',
                    'models_improved_alphazero/alphazero_model_10000.pth',
                    'models_improved_alphazero/alphazero_final.pth',
                    'models_alphazero/alphazero_model_13000.pth',  # 最新训练模型
                    'models_alphazero/alphazero_model_12500.pth',
                    'models_alphazero/alphazero_model_12000.pth',
                    'models_alphazero/alphazero_model_20000.pth',
                    'models_alphazero/alphazero_model_15000.pth',
                    'models_alphazero/alphazero_model_10000.pth',
                    'models_alphazero/alphazero_final.pth',
                    'models_alphazero/alphazero_model_7620.pth',
                    'models_alphazero/alphazero_model_7500.pth',
                    'models_alphazero/alphazero_model_7000.pth',
                    'models_alphazero/alphazero_model_5000.pth',
                    'models_alphazero/alphazero_model_1500.pth',
                    'models_alphazero/alphazero_model_500.pth'
                ]
                
                model_loaded = False
                for model_file in model_files:
                    try:
                        if os.path.exists(model_file):
                            # 检查是否是改进版模型
                            if 'improved' in model_file:
                                from improved_policy_value_net import ImprovedPolicyValueAgent
                                self.ai_agent = ImprovedPolicyValueAgent(board_size=self.board_size)
                            else:
                                self.ai_agent = PolicyValueAgent(board_size=self.board_size)
                            
                            self.ai_agent.load_model(model_file)
                            print(f"模型已加载: {model_file}")
                            model_loaded = True
                            break
                    except Exception as e:
                        continue
                
                if not model_loaded:
                    messagebox.showwarning("警告", "未找到任何AlphaZero模型文件")
                    self.update_status("未找到模型")
                    return
            else:  # 规则AI
                self.ai_agent = RuleBasedAI(board_size=self.board_size)
        except Exception as e:
            messagebox.showerror("错误", f"加载AI失败: {e}\n请确保模型文件存在")
            self.update_status("加载失败")
            return
        
        # 重置游戏
        self.env.reset()
        self.game_over = False
        self.last_move = None
        self.human_color = 1 if self.color_var.get() == "黑棋" else -1
        self.move_history = []  # 清空历史记录
        
        self.redraw_board()
        
        # 如果AI先手
        if self.human_color == -1:
            self.update_status("AI思考中...")
            self.root.after(500, self.ai_move)
        else:
            self.update_status("轮到你了 (黑棋)")
    
    def restart_game(self):
        """重新开始"""
        self.start_game()
    
    def undo_move(self):
        """悔棋（撤销AI和人类各一步，然后人类重新下）"""
        if self.game_over:
            messagebox.showwarning("提示", "游戏已结束，无法悔棋")
            return
        
        if self.ai_agent is None:
            messagebox.showwarning("提示", "请先开始游戏")
            return
        
        # 需要至少有2步棋（人类+AI）
        if len(self.move_history) < 2:
            messagebox.showwarning("提示", "没有可以悔的棋")
            return
        
        # 检查最后一步是否是AI下的
        last_move = self.move_history[-1]
        if last_move['player'] == self.human_color:
            messagebox.showwarning("提示", "当前轮到你下棋，无需悔棋")
            return
        
        # 1. 撤销AI的一步（最后一步）
        self.move_history.pop()
        
        # 2. 撤销人类的一步（倍数第二步）
        if len(self.move_history) > 0:
            self.move_history.pop()
        
        # 3. 恢复棋盘到悔棋前的状态
        if len(self.move_history) > 0:
            # 还有更早的历史，恢复到那个状态
            last_state = self.move_history[-1]
            # 恢复棋盘：使用历史记录中的board + 手动放回那一步的棋
            self.env.board = last_state['board'].copy()
            # 放回那一步的棋
            action = last_state['action']
            row = action // self.board_size
            col = action % self.board_size
            self.env.board[row, col] = last_state['player']
            # 设置当前玩家（下一个玩家应该是人类）
            self.env.current_player = self.human_color
            # 恢复最后一步标记
            self.last_move = (row, col)
        else:
            # 没有更早的历史，回到开局
            self.env.reset()
            self.last_move = None
            self.env.current_player = self.human_color
        
        self.env.done = False
        
        # 重绘棋盘
        self.redraw_board()
        
        # 更新状态
        player_name = "黑棋" if self.env.current_player == 1 else "白棋"
        self.update_status(f"已悔棋，轮到你了 ({player_name})")
    
    def on_click(self, event):
        """处理鼠标点击"""
        if self.game_over or self.ai_agent is None:
            return
        
        # 检查是否轮到人类
        if self.env.current_player != self.human_color:
            return
        
        # 计算点击位置
        col = round((event.x - self.cell_size) / self.cell_size)
        row = round((event.y - self.cell_size) / self.cell_size)
        
        # 检查位置合法性
        if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
            return
        
        if self.env.board[row, col] != 0:
            messagebox.showwarning("提示", "此位置已有棋子")
            return
        
        # 执行落子
        action = row * self.board_size + col
        self.execute_move(action, is_human=True)
    
    def execute_move(self, action, is_human=False):
        """
        执行一步棋
        Args:
            action: 动作索引
            is_human: 是否为人类落子
        """
        row = action // self.board_size
        col = action % self.board_size
        
        # 保存当前状态（用于悔棋）
        board_backup = self.env.board.copy()
        current_player = self.env.current_player
        done_backup = self.env.done
        
        # 执行动作
        state, reward, done, info = self.env.step(action)
        self.last_move = (row, col)
        
        # 记录历史（动作、玩家、执行前的棋盘状态）
        self.move_history.append({
            'action': action,
            'player': current_player,
            'board': board_backup,  # 执行前的棋盘状态（不包含当前这步）
            'done': done_backup
        })
        
        # 重绘棋盘
        self.redraw_board()
        
        # 检查游戏是否结束
        if done:
            self.game_over = True
            winner = info.get('winner', 0)
            
            if winner == self.human_color:
                self.update_status("🎉 你赢了！", "green")
                messagebox.showinfo("游戏结束", "恭喜你获胜！")
            elif winner == -self.human_color:
                self.update_status("😢 AI获胜", "red")
                messagebox.showinfo("游戏结束", "AI获胜，继续加油！")
            else:
                self.update_status("🤝 平局", "blue")
                messagebox.showinfo("游戏结束", "平局！")
            return
        
        # 如果是人类落子，轮到AI
        if is_human:
            self.update_status("AI思考中...")
            self.root.after(300, self.ai_move)
        else:
            player_name = "黑棋" if self.env.current_player == 1 else "白棋"
            self.update_status(f"轮到你了 ({player_name})")
    
    def ai_move(self):
        """AI落子"""
        if self.game_over:
            return
        
        state = self.env._get_observation()
        valid_actions = self.env.get_valid_actions()
        
        # 根据AI类型选择动作
        if self.ai_type == "DQN":
            if self.ai_agent is None:
                raise ValueError("DQN AI agent not initialized")
            action = self.ai_agent.select_action(state, valid_actions, training=False)
        elif self.ai_type == "AlphaZero":
            if self.ai_agent is None:
                raise ValueError("AlphaZero AI agent not initialized")
            # 大幅增加MCTS搜索次数，显著提升AI智能
            mcts = MCTS(self.ai_agent.policy_value_fn, n_simulations=3200, c_puct=5.0)
            action, _ = mcts.get_action(self.env, temp=1e-3)
        else:  # 规则AI
            if self.ai_agent is None:
                raise ValueError("Rule-based AI agent not initialized")
            action = self.ai_agent.get_action(self.env)
        
        # 执行AI落子
        self.execute_move(int(action), is_human=False)
    
    def update_status(self, text, color="#2196F3"):
        """更新状态栏"""
        self.status_label.config(text=text, fg=color)
        self.root.update()


def main():
    """主函数"""
    root = tk.Tk()
    app = GomokuGUI(root)
    root.mainloop()


if __name__ == '__main__':
    main()
