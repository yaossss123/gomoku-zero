"""
五子棋图形界面 (GUI)
使用Tkinter实现可视化人机对战
"""
import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
from gomoku_env import GomokuEnv
from mcts import MCTS
from improved_policy_value_net import ImprovedPolicyValueAgent
from rule_based_ai import RuleBasedAI
from config import MCTS_SIMULATIONS_PLAY, MCTS_C_PUCT, NUM_RES_BLOCKS, NUM_CHANNELS, BOARD_SIZE
import os


class GomokuGUI:
    """五子棋图形界面"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("五子棋AI对战")
        self.root.resizable(False, False)
        
        # 游戏参数
        self.board_size = BOARD_SIZE
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
        self.human_color = 1
        self.ai_agent = None
        self.ai_type = "AlphaZero"
        self.last_move = None
        self.move_history = []
        
        # 创建界面
        self.create_widgets()
        
    def create_widgets(self):
        """创建界面组件"""
        control_frame = tk.Frame(self.root)
        control_frame.pack(pady=10)
        
        tk.Label(control_frame, text="选择AI:").grid(row=0, column=0, padx=5)
        self.ai_var = tk.StringVar(value="AlphaZero")
        ai_choices = ["AlphaZero", "规则AI"]
        ai_menu = ttk.Combobox(control_frame, textvariable=self.ai_var, 
                              values=ai_choices, state="readonly", width=12)
        ai_menu.grid(row=0, column=1, padx=5)
        
        tk.Label(control_frame, text="你执:").grid(row=0, column=2, padx=5)
        self.color_var = tk.StringVar(value="白棋")
        color_menu = ttk.Combobox(control_frame, textvariable=self.color_var,
                                 values=["黑棋", "白棋"], state="readonly", width=8)
        color_menu.grid(row=0, column=3, padx=5)
        
        tk.Button(control_frame, text="开始游戏", command=self.start_game,
                 bg="#4CAF50", fg="white", width=10).grid(row=0, column=4, padx=5)
        tk.Button(control_frame, text="重新开始", command=self.restart_game,
                 bg="#2196F3", fg="white", width=10).grid(row=0, column=5, padx=5)
        tk.Button(control_frame, text="悔棋", command=self.undo_move,
                 bg="#FF9800", fg="white", width=10).grid(row=0, column=6, padx=5)
        
        canvas_size = self.cell_size * (self.board_size + 1)
        self.canvas = tk.Canvas(self.root, width=canvas_size, height=canvas_size,
                               bg=self.board_color, highlightthickness=0)
        self.canvas.pack(padx=20, pady=10)
        self.canvas.bind("<Button-1>", self.on_click)
        
        self.draw_board()
        
        self.status_label = tk.Label(self.root, text="请点击'开始游戏'",
                                     font=("Arial", 12), fg="#2196F3")
        self.status_label.pack(pady=10)
        
    def draw_board(self):
        """绘制棋盘网格"""
        self.canvas.delete("all")
        
        for i in range(self.board_size):
            x1 = self.cell_size
            y1 = self.cell_size * (i + 1)
            x2 = self.cell_size * self.board_size
            y2 = y1
            self.canvas.create_line(x1, y1, x2, y2, fill=self.line_color, width=1)
            
            x1 = self.cell_size * (i + 1)
            y1 = self.cell_size
            x2 = x1
            y2 = self.cell_size * self.board_size
            self.canvas.create_line(x1, y1, x2, y2, fill=self.line_color, width=1)
        
        star_positions = [(3, 3), (3, 11), (11, 3), (11, 11), (7, 7)]
        for row, col in star_positions:
            x = self.cell_size * (col + 1)
            y = self.cell_size * (row + 1)
            self.canvas.create_oval(x-3, y-3, x+3, y+3, fill=self.line_color)
        
        for i in range(self.board_size):
            self.canvas.create_text(self.cell_size // 2, self.cell_size * (i + 1),
                                   text=str(i), font=("Arial", 10))
            self.canvas.create_text(self.cell_size * (i + 1), self.cell_size // 2,
                                   text=str(i), font=("Arial", 10))
    
    def draw_piece(self, row, col, color, is_last=False):
        """绘制棋子"""
        x = self.cell_size * (col + 1)
        y = self.cell_size * (row + 1)
        
        piece_color = self.black_color if color == 1 else self.white_color
        outline_color = self.white_color if color == 1 else self.black_color
        
        self.canvas.create_oval(
            x - self.piece_radius, y - self.piece_radius,
            x + self.piece_radius, y + self.piece_radius,
            fill=piece_color, outline=outline_color, width=2
        )
        
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
        
        for row in range(self.board_size):
            for col in range(self.board_size):
                if self.env.board[row, col] != 0:
                    is_last = (self.last_move == (row, col))
                    self.draw_piece(row, col, self.env.board[row, col], is_last)
    
    def start_game(self):
        """开始新游戏"""
        self.ai_type = self.ai_var.get()
        self.update_status(f"正在加载{self.ai_type}...")
        self.root.update()
        
        try:
            if self.ai_type == "AlphaZero":
                model_dir = 'models_standard'
                model_files = []
                
                if os.path.exists(model_dir):
                    # 按训练步数排序，优先选择最新模型
                    for f in os.listdir(model_dir):
                        if f.endswith('.pth'):
                            model_files.append(os.path.join(model_dir, f))
                    
                    # 按文件名中的数字排序（提取model_XXX中的数字）
                    import re
                    def extract_number(path):
                        match = re.search(r'model_(\d+)\.pth', os.path.basename(path))
                        return int(match.group(1)) if match else 0
                    model_files.sort(key=extract_number, reverse=True)
                
                # 尝试加载模型
                model_loaded = False
                for model_file in model_files:
                    try:
                        self.ai_agent = ImprovedPolicyValueAgent(board_size=self.board_size, num_res_blocks=NUM_RES_BLOCKS, num_channels=NUM_CHANNELS)
                        self.ai_agent.load_model(model_file)
                        print(f"模型已加载: {model_file}")
                        model_loaded = True
                        break
                    except Exception as e:
                        print(f"加载 {model_file} 失败: {e}")
                        continue
                
                # 如果没有找到可用的模型，使用未训练的模型
                if not model_loaded:
                    messagebox.showwarning("警告", "未找到任何AlphaZero模型文件，将使用未训练的模型")
                    self.ai_agent = ImprovedPolicyValueAgent(board_size=self.board_size, num_res_blocks=NUM_RES_BLOCKS, num_channels=NUM_CHANNELS)
            else:
                self.ai_agent = RuleBasedAI(board_size=self.board_size)
        except Exception as e:
            messagebox.showerror("错误", f"加载AI失败: {e}")
            self.update_status("加载失败")
            return
        
        # 重置游戏
        self.env.reset()
        self.game_over = False
        self.last_move = None
        self.human_color = 1 if self.color_var.get() == "黑棋" else -1
        self.move_history = []
        
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
        """悔棋"""
        if self.game_over:
            messagebox.showwarning("提示", "游戏已结束，无法悔棋")
            return
        
        if self.ai_agent is None:
            messagebox.showwarning("提示", "请先开始游戏")
            return
        
        if len(self.move_history) < 2:
            messagebox.showwarning("提示", "没有可以悔的棋")
            return
        
        last_move = self.move_history[-1]
        if last_move['player'] == self.human_color:
            messagebox.showwarning("提示", "当前轮到你下棋，无需悔棋")
            return
        
        # 撤销AI的一步
        self.move_history.pop()
        
        # 撤销人类的一步
        if len(self.move_history) > 0:
            self.move_history.pop()
        
        # 恢复棋盘状态
        if len(self.move_history) > 0:
            last_state = self.move_history[-1]
            self.env.board = last_state['board'].copy()
            action = last_state['action']
            row = action // self.board_size
            col = action % self.board_size
            self.env.board[row, col] = last_state['player']
            self.env.current_player = self.human_color
            self.last_move = (row, col)
        else:
            self.env.reset()
            self.last_move = None
            self.env.current_player = self.human_color
        
        self.env.done = False
        self.redraw_board()
        
        player_name = "黑棋" if self.env.current_player == 1 else "白棋"
        self.update_status(f"已悔棋，轮到你了 ({player_name})")
    
    def on_click(self, event):
        """处理鼠标点击"""
        if self.game_over or self.ai_agent is None:
            return
        
        if self.env.current_player != self.human_color:
            return
        
        col = round((event.x - self.cell_size) / self.cell_size)
        row = round((event.y - self.cell_size) / self.cell_size)
        
        if row < 0 or row >= self.board_size or col < 0 or col >= self.board_size:
            return
        
        if self.env.board[row, col] != 0:
            messagebox.showwarning("提示", "此位置已有棋子")
            return
        
        # 禁手检查已禁用（训练时不用，与人类对战也不用）
        # if self.human_color == 1 and self.env.current_player == 1:
        #     forbidden = self.env.check_forbidden(row, col)
        #     if forbidden:
        #         forbidden_names = {
        #             'double_three': '三三禁手',
        #             'double_four': '四四禁手',
        #             'overline': '长连禁手'
        #         }
        #         messagebox.showwarning("禁手", f"此位置为{forbidden_names.get(forbidden, '禁手')}，黑棋不能下！")
        #         return
        
        action = row * self.board_size + col
        self.execute_move(action, is_human=True)
    
    def execute_move(self, action, is_human=False):
        """执行一步棋"""
        row = action // self.board_size
        col = action % self.board_size
        
        board_backup = self.env.board.copy()
        current_player = self.env.current_player
        done_backup = self.env.done
        
        state, reward, done, info = self.env.step(action)
        self.last_move = (row, col)
        
        self.move_history.append({
            'action': action,
            'player': current_player,
            'board': board_backup,
            'done': done_backup
        })
        
        self.redraw_board()
        
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
        
        if self.ai_type == "AlphaZero":
            # 使用合理的MCTS参数
            mcts = MCTS(
                self.ai_agent.policy_value_fn, 
                n_simulations=MCTS_SIMULATIONS_PLAY,
                c_puct=MCTS_C_PUCT
            )
            action, _ = mcts.get_action(self.env, temp=1e-3, add_noise=False)
        else:
            action = self.ai_agent.get_action(self.env)
        
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