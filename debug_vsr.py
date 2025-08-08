#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频超分辨率调试脚本
用于诊断和修复超分辨率问题
"""

import os
import sys
import torch
import numpy as np
import cv2
import yaml
from pathlib import Path

# 添加项目路径
sys.path.append('codes')

from models import define_model
from utils import data_utils, base_utils


def debug_data_processing():
    """调试数据处理流程"""
    print("🔍 调试数据处理流程...")
    
    # 创建一个测试图像
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    print(f"原始图像范围: [{test_image.min()}, {test_image.max()}]")
    print(f"原始图像形状: {test_image.shape}")
    
    # 测试canonicalize函数
    canonicalized = data_utils.canonicalize(test_image)
    print(f"归一化后范围: [{canonicalized.min():.3f}, {canonicalized.max():.3f}]")
    print(f"归一化后形状: {canonicalized.shape}")
    
    # 测试float32_to_uint8函数
    converted_back = data_utils.float32_to_uint8(canonicalized.numpy())
    print(f"转换回uint8范围: [{converted_back.min()}, {converted_back.max()}]")
    
    return True


def debug_model_loading():
    """调试模型加载"""
    print("\n🔍 调试模型加载...")
    
    try:
        # 检查配置文件
        config_file = 'experiments_BD/EGVSR/001/test.yml'
        if not os.path.exists(config_file):
            print(f"❌ 配置文件不存在: {config_file}")
            return False
            
        with open(config_file, 'r', encoding='utf-8') as f:
            opt = yaml.load(f.read(), Loader=yaml.FullLoader)
        
        print("✅ 配置文件加载成功")
        
        # 检查模型文件
        model_file = 'pretrained_models/EGVSR_iter420000.pth'
        if not os.path.exists(model_file):
            print(f"❌ 模型文件不存在: {model_file}")
            return False
            
        print("✅ 模型文件存在")
        
        # 设置模型参数
        opt['model']['name'] = 'tecogan'
        opt['model']['generator']['load_path'] = model_file
        opt['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
        opt['is_train'] = False
        opt['verbose'] = False
        
        print(f"✅ 使用设备: {opt['device']}")
        
        # 创建模型
        model = define_model(opt)
        print("✅ 模型创建成功")
        
        return model
        
    except Exception as e:
        print(f"❌ 模型加载失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def debug_inference(model):
    """调试推理过程"""
    print("\n🔍 调试推理过程...")
    
    try:
        # 创建测试数据
        batch_size, channels, height, width = 1, 3, 64, 64
        test_data = np.random.randint(0, 255, (batch_size, height, width, channels), dtype=np.uint8)
        
        print(f"测试数据形状: {test_data.shape}")
        print(f"测试数据范围: [{test_data.min()}, {test_data.max()}]")
        
        # 使用正确的数据处理
        test_tensor = data_utils.canonicalize(test_data)
        print(f"处理后数据范围: [{test_tensor.min():.3f}, {test_tensor.max():.3f}]")
        
        # 推理
        with torch.no_grad():
            # 设置模型为评估模式
            if hasattr(model, 'net_G'):
                model.net_G.eval()
            
            print(f"输入tensor形状: {test_tensor.shape}")
            print(f"输入tensor设备: {test_tensor.device}")
            
            try:
                output = model.infer(test_tensor)
                print(f"输出类型: {type(output)}")
                if isinstance(output, torch.Tensor):
                    print(f"输出形状: {output.shape}")
                    if output.numel() > 0:
                        print(f"输出范围: [{output.min():.3f}, {output.max():.3f}]")
                    else:
                        print("输出为空tensor")
                elif isinstance(output, np.ndarray):
                    print(f"输出形状: {output.shape}")
                    if output.size > 0:
                        print(f"输出范围: [{output.min()}, {output.max()}]")
                    else:
                        print("输出为空数组")
            except Exception as e:
                print(f"推理过程中出错: {str(e)}")
                import traceback
                traceback.print_exc()
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ 推理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def debug_video_processing():
    """调试视频处理"""
    print("\n🔍 调试视频处理...")
    
    # 检查是否有测试视频
    test_video = "1.jpg"  # 使用测试图像
    
    if not os.path.exists(test_video):
        print(f"❌ 测试文件不存在: {test_video}")
        print("请确保测试数据集已下载")
        return False
    
    try:
        # 读取测试图像
        test_image = cv2.imread(test_video)
        if test_image is None:
            print("❌ 无法读取测试图像")
            return False
            
        test_image = cv2.cvtColor(test_image, cv2.COLOR_BGR2RGB)
        print(f"测试图像形状: {test_image.shape}")
        
        # 创建低分辨率版本
        h, w = test_image.shape[:2]
        lr_h, lr_w = h // 4, w // 4
        lr_image = cv2.resize(test_image, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
        
        print(f"低分辨率图像形状: {lr_image.shape}")
        
        # 转换为tensor
        lr_tensor = data_utils.canonicalize(lr_image[np.newaxis, ...])
        print(f"Tensor形状: {lr_tensor.shape}")
        print(f"Tensor范围: [{lr_tensor.min():.3f}, {lr_tensor.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ 视频处理失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("🎬 视频超分辨率调试工具")
    print("=" * 50)
    
    # 1. 调试数据处理
    if not debug_data_processing():
        print("❌ 数据处理调试失败")
        return
    
    # 2. 调试模型加载
    model = debug_model_loading()
    if not model:
        print("❌ 模型加载调试失败")
        return
    
    # 3. 调试推理
    if not debug_inference(model):
        print("❌ 推理调试失败")
        return
    
    # 4. 调试视频处理
    if not debug_video_processing():
        print("❌ 视频处理调试失败")
        return
    
    print("\n✅ 所有调试测试通过！")
    print("\n💡 如果仍然有问题，请检查：")
    print("1. 预训练模型文件是否正确")
    print("2. 输入视频质量是否足够好")
    print("3. GPU内存是否充足")
    print("4. 模型配置文件是否正确")


if __name__ == "__main__":
    main() 