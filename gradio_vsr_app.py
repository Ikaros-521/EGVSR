import os
import sys
import torch
import numpy as np
import cv2
import gradio as gr
import yaml
from pathlib import Path

# 添加项目路径
sys.path.append('codes')

from models import define_model
from utils import data_utils, base_utils


class FixedVSRApp:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_name = None
        
    def load_model(self, model_name: str):
        """加载指定的模型"""
        if self.model_name == model_name and self.model is not None:
            return self.model
            
        # 模型配置
        model_configs = {
            'EGVSR': {
                'model_name': 'tecogan',
                'config_file': 'experiments_BD/EGVSR/001/test.yml',
                'checkpoint': 'pretrained_models/EGVSR_iter420000.pth'
            },
            'TecoGAN': {
                'model_name': 'tecogan', 
                'config_file': 'experiments_BD/TecoGAN/001/test.yml',
                'checkpoint': 'pretrained_models/TecoGAN_BD_iter500000.pth'
            },
            'FRVSR': {
                'model_name': 'frvsr',
                'config_file': 'experiments_BD/FRVSR/001/test.yml', 
                'checkpoint': 'pretrained_models/FRVSR_BD_iter400000.pth'
            }
        }
        
        if model_name not in model_configs:
            raise ValueError(f"不支持的模型: {model_name}")
            
        config = model_configs[model_name]
        
        try:
            # 读取配置文件
            with open(config['config_file'], 'r', encoding='utf-8') as f:
                opt = yaml.load(f.read(), Loader=yaml.FullLoader)
            
            # 设置模型参数
            opt['model']['name'] = config['model_name']
            opt['model']['generator']['load_path'] = config['checkpoint']
            opt['device'] = str(self.device)
            opt['is_train'] = False
            opt['verbose'] = False
            
            # 创建模型
            self.model = define_model(opt)
            self.model_name = model_name
            
            return self.model
            
        except Exception as e:
            raise ValueError(f"加载模型失败: {str(e)}")
    
    def extract_frames(self, video_path: str) -> tuple:
        """从视频中提取帧"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 转换为RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            frame_count += 1
            
        cap.release()
        
        return frames, fps
    
    def create_low_resolution_frames(self, frames: list, scale: int = 4, method: str = 'bicubic') -> list:
        """创建低分辨率帧"""
        lr_frames = []
        for frame in frames:
            h, w = frame.shape[:2]
            lr_h, lr_w = h // scale, w // scale
            
            if method == 'bicubic':
                lr_frame = cv2.resize(frame, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
            elif method == 'bilinear':
                lr_frame = cv2.resize(frame, (lr_w, lr_h), interpolation=cv2.INTER_LINEAR)
            elif method == 'nearest':
                lr_frame = cv2.resize(frame, (lr_w, lr_h), interpolation=cv2.INTER_NEAREST)
            else:
                lr_frame = cv2.resize(frame, (lr_w, lr_h), interpolation=cv2.INTER_CUBIC)
                
            lr_frames.append(lr_frame)
        return lr_frames
    
    def process_video_fixed(self, video_path: str, model_name: str, max_frames: int = 20, 
                           create_lr: bool = True, downscale_method: str = 'bicubic'):
        """修复版本的视频处理"""
        try:
            # 加载模型
            model = self.load_model(model_name)
            
            # 提取帧
            frames, fps = self.extract_frames(video_path)
            if not frames:
                raise ValueError("无法从视频中提取帧")
            
            # 根据用户选择决定是否创建低分辨率帧
            if create_lr:
                # 创建低分辨率帧用于演示
                lr_frames = self.create_low_resolution_frames(frames, method=downscale_method)
                input_frames = lr_frames
                print(f"创建低分辨率输入: {len(input_frames)} 帧")
                print(f"低分辨率尺寸: {input_frames[0].shape}")
            else:
                # 直接使用原始帧，不进行任何下采样
                input_frames = frames
                print(f"直接使用原始输入: {len(input_frames)} 帧")
                print(f"原始尺寸: {input_frames[0].shape}")
            
            print(f"将处理完整视频的 {len(input_frames)} 帧进行超分辨率")
            
            # 关键修复：使用正确的数据处理流程
            # 1. 堆叠帧
            stacked_frames = np.stack(input_frames)  # (T, H, W, C)
            
            # 2. 使用项目的数据处理工具进行正确的数据转换
            input_tensor = data_utils.canonicalize(stacked_frames)  # 归一化到[0,1]
            
            print(f"输入tensor形状: {input_tensor.shape}")
            print(f"输入tensor范围: [{input_tensor.min():.3f}, {input_tensor.max():.3f}]")
            
            # 3. 推理
            with torch.no_grad():
                # 设置模型为评估模式
                if hasattr(model, 'net_G'):
                    model.net_G.eval()
                sr_frames = model.infer(input_tensor)
            
            # 4. 处理输出 - 模型输出已经是uint8格式
            if isinstance(sr_frames, torch.Tensor):
                sr_frames = sr_frames.cpu().numpy()
            
            print(f"输出形状: {sr_frames.shape}")
            print(f"输出范围: [{sr_frames.min()}, {sr_frames.max()}]")
            
            # 5. 确保输出是uint8格式并验证帧序列
            sr_frames = np.clip(sr_frames, 0, 255).astype(np.uint8)
            sr_frames_list = [frame for frame in sr_frames]
            
            # 验证帧序列
            print(f"超分帧序列信息:")
            print(f"  - 总帧数: {len(sr_frames_list)}")
            print(f"  - 第一帧形状: {sr_frames_list[0].shape}")
            print(f"  - 第一帧范围: [{sr_frames_list[0].min()}, {sr_frames_list[0].max()}]")
            print(f"  - 最后一帧形状: {sr_frames_list[-1].shape}")
            print(f"  - 最后一帧范围: [{sr_frames_list[-1].min()}, {sr_frames_list[-1].max()}]")
            
            # 检查帧序列是否一致
            if len(sr_frames_list) != len(input_frames):
                print(f"警告: 输入帧数({len(input_frames)})与输出帧数({len(sr_frames_list)})不匹配")
            
            # 6. 创建输出视频 - 修复版本
            output_path = video_path.replace('.mp4', '_sr.mp4')
            h, w = sr_frames[0].shape[:2]
            
            # 尝试不同的编码器
            try:
                # 首先尝试 H.264 编码器
                fourcc = cv2.VideoWriter_fourcc(*'H264')
                out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                if not out.isOpened():
                    raise Exception("H264编码器不可用")
            except:
                try:
                    # 尝试 XVID 编码器
                    output_path = video_path.replace('.mp4', '_sr.avi')
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                    if not out.isOpened():
                        raise Exception("XVID编码器不可用")
                except:
                    # 最后使用 MJPG 编码器
                    output_path = video_path.replace('.mp4', '_sr.avi')
                    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            
            print(f"创建视频: {output_path}, 尺寸: {w}x{h}, FPS: {fps}")
            print(f"超分帧数量: {len(sr_frames)}")
            
            # 写入帧，确保顺序正确
            for i, frame in enumerate(sr_frames):
                # 确保帧是RGB格式
                if frame.shape[2] == 3:
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                
                # 确保数据类型正确
                frame_bgr = frame_bgr.astype(np.uint8)
                
                # 写入帧
                success = out.write(frame_bgr)
                if not success:
                    print(f"警告: 第{i}帧写入失败")
                
            out.release()
            print(f"视频创建完成: {output_path}")
            
            # 保存前几帧用于验证
            try:
                debug_dir = "debug_frames"
                os.makedirs(debug_dir, exist_ok=True)
                
                # 保存前3帧用于对比
                for i in range(min(3, len(sr_frames_list))):
                    # 保存超分帧
                    sr_frame_path = os.path.join(debug_dir, f"sr_frame_{i:03d}.png")
                    cv2.imwrite(sr_frame_path, cv2.cvtColor(sr_frames_list[i], cv2.COLOR_RGB2BGR))
                    
                    # 保存对应的输入帧
                    if i < len(input_frames):
                        input_frame_path = os.path.join(debug_dir, f"input_frame_{i:03d}.png")
                        cv2.imwrite(input_frame_path, cv2.cvtColor(input_frames[i], cv2.COLOR_RGB2BGR))
                
                print(f"调试帧已保存到 {debug_dir} 目录")
            except Exception as e:
                print(f"保存调试帧失败: {str(e)}")
            
            # 返回结果
            if create_lr:
                return output_path, lr_frames[:max_frames], sr_frames_list[:max_frames], None
            else:
                return output_path, frames[:max_frames], sr_frames_list[:max_frames], None
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise gr.Error(f"处理视频时出错: {str(e)}")


def create_interface():
    """创建Gradio界面"""
    app = FixedVSRApp()
    
    with gr.Blocks(title="修复版视频超分辨率系统") as interface:
        gr.Markdown("# 🎬 修复版视频超分辨率系统")
        gr.Markdown("修复了超分结果只有轮廓的问题")
        
        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="输入视频")
                model_selector = gr.Dropdown(
                    choices=['EGVSR', 'TecoGAN', 'FRVSR'],
                    value="EGVSR",
                    label="选择模型"
                )
                max_frames = gr.Slider(
                    minimum=5,
                    maximum=30,
                    value=20,
                    step=5,
                    label="最大处理帧数"
                )
                
                # 添加新的选项
                with gr.Row():
                    create_lr_checkbox = gr.Checkbox(
                        value=False,
                        label="创建低分辨率输入（用于演示）"
                    )
                    downscale_method = gr.Dropdown(
                        choices=['bicubic', 'bilinear', 'nearest'],
                        value='bicubic',
                        label="下采样方法"
                    )
                
                process_btn = gr.Button("🚀 开始处理", variant="primary")
            
            with gr.Column():
                video_output = gr.Video(label="超分辨率结果")
                status_text = gr.Textbox(label="处理状态", interactive=False)
        
        # 帧对比
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 输入帧")
                input_gallery = gr.Gallery(label="输入帧", show_label=False)
            with gr.Column():
                gr.Markdown("### 超分辨率帧")
                sr_gallery = gr.Gallery(label="超分辨率帧", show_label=False)
        
        def process_video_wrapper(video_path, model_name, max_frames, create_lr, downscale_method):
            if video_path is None:
                return None, [], [], "请先上传视频"
            
            try:
                # 处理完整的视频（使用max_frames参数）
                result = app.process_video_fixed(video_path, model_name, max_frames, create_lr, downscale_method)
                output_path, input_frames, sr_frames, _ = result
                
                if create_lr:
                    status_msg = f"处理完成！已创建低分辨率输入进行超分辨率演示。处理了 {len(sr_frames)} 帧。"
                else:
                    status_msg = f"处理完成！直接对原始视频进行超分辨率处理。处理了 {len(sr_frames)} 帧。"
                
                # 只在界面上显示前30帧，但实际处理了完整的视频
                max_display_frames = 30
                input_frames_to_show = input_frames[:max_display_frames]
                sr_frames_to_show = sr_frames[:max_display_frames]
                
                return output_path, input_frames_to_show, sr_frames_to_show, status_msg
            except Exception as e:
                import traceback
                traceback.print_exc()
                return None, [], [], f"处理失败: {str(e)}"
        
        process_btn.click(
            process_video_wrapper,
            inputs=[video_input, model_selector, max_frames, create_lr_checkbox, downscale_method],
            outputs=[video_output, input_gallery, sr_gallery, status_text]
        )
        
        # 添加说明
        with gr.Accordion("ℹ️ 使用说明", open=False):
            gr.Markdown("""
            ### 重要说明：
            
            **"创建低分辨率输入"选项说明：**
            
            ✅ **勾选**：
            - 自动将输入视频下采样4倍，然后进行超分辨率处理
            - 适用于演示超分辨率效果
            - 可以对比原始高分辨率帧和超分结果
            - 适合高分辨率输入视频
            
            ❌ **不勾选**：
            - 直接使用原始视频尺寸，不进行任何下采样
            - 适用于已经是低分辨率的输入视频
            - 如果输入是高分辨率视频，会直接进行超分处理
            
            ### 修复的问题：
            1. **数据范围问题**: 使用 `data_utils.canonicalize()` 正确归一化数据到[0,1]范围
            2. **数据格式问题**: 确保输入tensor格式正确 (T, H, W, C)
            3. **视频编码问题**: 支持多种编码器，确保视频正确生成
            4. **帧序列问题**: 验证输入输出帧数匹配
            
            ### 使用建议：
            - 想要演示超分效果：勾选"创建低分辨率输入"
            - 处理低分辨率视频：不勾选，直接使用原始尺寸
            - 查看控制台输出的调试信息
            - 检查 `debug_frames` 目录中的对比帧
            """)
        
        gr.Markdown("---")
        gr.Markdown("基于 [EGVSR-PyTorch](https://github.com/Thmen/EGVSR) 项目构建")
    
    return interface


if __name__ == "__main__":
    # 检查依赖
    try:
        import gradio
        print("✅ Gradio 已安装")
    except ImportError:
        print("❌ 请先安装 Gradio: pip install gradio")
        sys.exit(1)
    
    # 检查模型文件
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
        print("请下载相应的预训练模型文件到 pretrained_models 目录")
    
    # 启动应用
    interface = create_interface()
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True
    ) 