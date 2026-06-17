"""
快速启动训练脚本 - 多配置版
"""
import os
import sys


def main():
    print("=" * 60)
    print("五子棋 AlphaZero 训练系统")
    print("=" * 60)
    
    print("\n选择操作：")
    print("1. 运行MCTS测试")
    print("2. 超快速测试（验证代码，~10分钟）")
    print("3. 快速训练（几小时看效果）")
    print("4. 标准训练（推荐，1-2天）")
    print("5. 继续训练（从检查点恢复）")
    print("6. 人机对战")
    print("7. 启动GUI")
    print("0. 退出")
    
    choice = input("\n请选择 (0-7): ").strip()
    
    if choice == '0':
        print("退出")
        return
    
    elif choice == '1':
        print("\n运行MCTS测试...")
        from test_mcts import run_all_tests
        run_all_tests()
    
    elif choice == '2':
        print("\n超快速测试模式...")
        from train_with_improved_net import AlphaZeroTrainer
        
        trainer = AlphaZeroTrainer(
            board_size=15,
            num_res_blocks=3,
            num_channels=64,
            n_simulations=100,
            batch_size=64,
            buffer_size=5000,
            epochs_per_update=2,
        )
        
        trainer.train(
            n_games=50,
            save_interval=25,
            eval_interval=25,
            log_interval=5,
            model_dir='models_test',
        )
    
    elif choice == '3':
        print("\n快速训练模式...")
        from train_with_improved_net import AlphaZeroTrainer
        
        trainer = AlphaZeroTrainer(
            board_size=15,
            num_res_blocks=5,
            num_channels=128,
            n_simulations=200,
            batch_size=128,
            buffer_size=20000,
            epochs_per_update=3,
        )
        
        trainer.train(
            n_games=500,
            save_interval=100,
            eval_interval=50,
            log_interval=10,
            model_dir='models_fast',
        )
    
    elif choice == '4':
        print("\n标准训练模式...")
        from train_with_improved_net import AlphaZeroTrainer
        
        trainer = AlphaZeroTrainer(
            board_size=15,
            num_res_blocks=10,
            num_channels=128,
            n_simulations=400,
            batch_size=256,
            buffer_size=100000,
            epochs_per_update=5,
        )
        
        trainer.train(
            n_games=3000,
            save_interval=500,
            eval_interval=100,
            log_interval=10,
            model_dir='models_standard',
        )
    
    elif choice == '5':
        print("\n从检查点继续训练...")
        
        model_dirs = ['models_dense_reward', 'models_test', 'models_fast', 'models_standard', 
                      'models_alphazero', 'models_alphazero_fixed']
        checkpoints = []
        
        for model_dir in model_dirs:
            if os.path.exists(model_dir):
                for f in os.listdir(model_dir):
                    if f.endswith('.pth'):
                        checkpoints.append(os.path.join(model_dir, f))
        
        if not checkpoints:
            print("未找到任何检查点！")
            return
        
        print("\n可用的检查点：")
        for i, cp in enumerate(checkpoints):
            print(f"  {i+1}. {cp}")
        
        idx = int(input("\n选择检查点 (输入编号): ").strip()) - 1
        
        if 0 <= idx < len(checkpoints):
            checkpoint = checkpoints[idx]
            print(f"\n从 {checkpoint} 恢复训练...")
            
            # 尝试加载检查点获取配置
            import torch
            ckpt = torch.load(checkpoint, map_location='cpu')
            num_res_blocks = ckpt.get('num_res_blocks', 10)
            num_channels = ckpt.get('num_channels', 128)
            
            from train_with_improved_net import AlphaZeroTrainer
            
            trainer = AlphaZeroTrainer(
                board_size=15,
                num_res_blocks=num_res_blocks,
                num_channels=num_channels,
                n_simulations=400,
                batch_size=256,
                buffer_size=100000,
            )
            
            additional_games = int(input("继续训练多少局？(默认1000): ").strip() or "1000")
            
            trainer.train(
                n_games=additional_games,
                save_interval=500,
                eval_interval=100,
                model_dir=os.path.dirname(checkpoint),
                resume_from=checkpoint,
            )
        else:
            print("无效选择！")
    
    elif choice == '6':
        print("\n启动人机对战...")
        os.system(f"{sys.executable} play.py")
    
    elif choice == '7':
        print("\n启动GUI...")
        os.system(f"{sys.executable} gomoku_gui.py")
    
    else:
        print("无效选择！")


if __name__ == '__main__':
    main()