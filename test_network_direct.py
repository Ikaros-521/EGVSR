#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接测试网络脚本
绕过模型包装器，直接测试网络
"""

import os
import sys
import torch
import numpy as np
import cv2
import yaml

# 添加项目路径
sys.path.append('codes')

from models.networks import define_generator
from utils import data_utils


def test_network_direct():
    """直接测试网络"""
    print("🔍 直接测试网络...")
    
    try:
        # 读取配置文件
        config_file = 'experiments_BD/EGVSR/001/test.yml'
        with open(config_file, 'r', encoding='utf-8') as f:
            opt = yaml.load(f.read(), Loader=yaml.FullLoader)
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        # 创建生成器网络
        net_G = define_generator(opt).to(device)
        print("✅ 生成器网络创建成功")
        
        # 加载预训练权重
        model_file = 'pretrained_models/EGVSR_iter420000.pth'
        if os.path.exists(model_file):
            net_G.load_state_dict(torch.load(model_file, map_location=device))
            print("✅ 预训练权重加载成功")
        else:
            print("⚠️  预训练权重文件不存在，使用随机权重")
        
        # 设置评估模式
        net_G.eval()
        
        # 创建测试数据
        batch_size, channels, height, width = 1, 3, 64, 64
        test_data = np.random.randint(0, 255, (batch_size, height, width, channels), dtype=np.uint8)
        
        print(f"测试数据形状: {test_data.shape}")
        print(f"测试数据范围: [{test_data.min()}, {test_data.max()}]")
        
        # 使用正确的数据处理
        test_tensor = data_utils.canonicalize(test_data)
        print(f"处理后数据范围: [{test_tensor.min():.3f}, {test_tensor.max():.3f}]")
        
        # 转换为tchw格式
        test_tensor = test_tensor.permute(0, 3, 1, 2)  # bhwc -> bchw
        print(f"转换后形状: {test_tensor.shape}")
        
        # 直接调用网络的infer_sequence方法
        with torch.no_grad():
            try:
                output = net_G.infer_sequence(test_tensor, device)
                print(f"输出类型: {type(output)}")
                print(f"输出形状: {output.shape}")
                if output.size > 0:
                    print(f"输出范围: [{output.min()}, {output.max()}]")
                else:
                    print("输出为空")
                return True
            except Exception as e:
                print(f"网络推理失败: {str(e)}")
                import traceback
                traceback.print_exc()
                return False
                
    except Exception as e:
        print(f"❌ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_simple_forward():
    """测试简单的前向传播"""
    print("\n🔍 测试简单前向传播...")
    
    try:
        # 读取配置文件
        config_file = 'experiments_BD/EGVSR/001/test.yml'
        with open(config_file, 'r', encoding='utf-8') as f:
            opt = yaml.load(f.read(), Loader=yaml.FullLoader)
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 创建生成器网络
        net_G = define_generator(opt).to(device)
        
        # 加载预训练权重
        model_file = 'pretrained_models/EGVSR_iter420000.pth'
        if os.path.exists(model_file):
            net_G.load_state_dict(torch.load(model_file, map_location=device))
        
        net_G.eval()
        
        # 创建测试数据
        batch_size, channels, height, width = 1, 3, 64, 64
        test_data = np.random.randint(0, 255, (batch_size, height, width, channels), dtype=np.uint8)
        test_tensor = data_utils.canonicalize(test_data)
        test_tensor = test_tensor.permute(0, 3, 1, 2)  # bhwc -> bchw
        
        # 创建虚拟的前一帧
        lr_prev = torch.zeros_like(test_tensor)
        hr_prev = torch.zeros(batch_size, channels, height * 4, width * 4, device=device)
        
        with torch.no_grad():
            try:
                # 调用forward方法
                output = net_G.forward(test_tensor, lr_prev, hr_prev)
                print(f"Forward输出类型: {type(output)}")
                if isinstance(output, torch.Tensor):
                    print(f"Forward输出形状: {output.shape}")
                    print(f"Forward输出范围: [{output.min():.3f}, {output.max():.3f}]")
                return True
            except Exception as e:
                print(f"Forward失败: {str(e)}")
                import traceback
                traceback.print_exc()
                return False
                
    except Exception as e:
        print(f"❌ Forward测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("🎬 直接网络测试工具")
    print("=" * 50)
    
    # 测试直接网络推理
    if not test_network_direct():
        print("❌ 直接网络测试失败")
        return
    
    # 测试简单前向传播
    if not test_simple_forward():
        print("❌ 简单前向传播测试失败")
        return
    
    print("\n✅ 所有测试通过！")


if __name__ == "__main__":
    main() 