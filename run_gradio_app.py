#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频超分辨率 Gradio Web应用启动脚本
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_model_files():
    """检查预训练模型文件"""
    model_files = [
        "pretrained_models/EGVSR_iter420000.pth",
        "pretrained_models/TecoGAN_BD_iter500000.pth", 
        "pretrained_models/FRVSR_BD_iter400000.pth"
    ]
    
    missing_files = []
    for file_path in model_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("⚠️  警告: 以下预训练模型文件缺失:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\n请下载相应的预训练模型文件到 pretrained_models 目录")
        print("可以从以下地址下载:")
        print("https://github.com/Thmen/EGVSR")
        return False
    
    print("✅ 预训练模型文件检查完成")
    return True

def check_cuda():
    """检查CUDA是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA可用 - GPU: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("⚠️  CUDA不可用，将使用CPU模式（处理速度较慢）")
            return False
    except ImportError:
        print("⚠️  PyTorch未安装，无法检查CUDA")
        return False

def main():
    parser = argparse.ArgumentParser(description="启动视频超分辨率Gradio应用")
    parser.add_argument('--port', type=int, default=7860, help='服务器端口 (默认: 7860)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='服务器地址 (默认: 0.0.0.0)')
    parser.add_argument('--share', action='store_true', help='创建公共链接')
    parser.add_argument('--skip-checks', action='store_true', help='跳过依赖检查')
    
    args = parser.parse_args()
    
    print("🎬 视频超分辨率 Gradio Web应用")
    print("=" * 50)
    
    if not args.skip_checks:
        # 检查模型文件
        if not check_model_files():
            print("\n是否继续启动应用? (y/N): ", end="")
            response = input().strip().lower()
            if response not in ['y', 'yes']:
                sys.exit(1)
        
        # 检查CUDA
        check_cuda()
    
    print("\n🚀 启动应用...")
    print(f"   地址: http://{args.host}:{args.port}")
    if args.share:
        print("   公共链接: 将自动生成")
    
    try:
        # 启动应用
        from gradio_vsr_app import create_interface
        
        interface = create_interface()
        interface.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share,
            show_error=True,
            inbrowser=True
        )
        
    except KeyboardInterrupt:
        print("\n👋 应用已停止")
    except Exception as e:
        print(f"\n❌ 启动失败: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 