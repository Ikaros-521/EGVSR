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
    
    def extract_frames(self, video_path: str, max_frames: int = 20) -> tuple:
        """从视频中提取帧"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        frame_count = 0
        while len(frames) < max_frames:
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
            frames, fps = self.extract_frames(video_path, max_frames)
            if not frames:
                raise ValueError("无法从视频中提取帧")
            
            # 根据用户选择决定是否创建低分辨率帧
            if create_lr:
                # 创建低分辨率帧用于演示
                lr_frames = self.create_low_resolution_frames(frames, method=downscale_method)
                input_frames = lr_frames
                original_frames = frames  # 保存原始高分辨率帧用于对比
            else:
                # 直接使用原始帧作为输入
                input_frames = frames
                original_frames = None
            
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
            
            # 5. 确保输出是uint8格式
            sr_frames = np.clip(sr_frames, 0, 255).astype(np.uint8)
            sr_frames_list = [frame for frame in sr_frames]
            
            # 6. 创建输出视频
            output_path = video_path.replace('.mp4', '_sr.mp4')
            h, w = sr_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
            
            for frame in sr_frames:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                out.write(frame_bgr)
                
            out.release()
            
            # 返回结果
            if create_lr:
                return output_path, lr_frames, sr_frames_list, original_frames
            else:
                return output_path, frames, sr_frames_list, None
            
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
                        value=True,
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
        
        # 原始帧对比（如果创建了低分辨率）
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 原始高分辨率帧（对比用）")
                original_gallery = gr.Gallery(label="原始帧", show_label=False)
        
        def process_video_wrapper(video_path, model_name, max_frames, create_lr, downscale_method):
            if video_path is None:
                return None, [], [], [], "请先上传视频"
            
            try:
                result = app.process_video_fixed(video_path, model_name, max_frames, create_lr, downscale_method)
                if create_lr:
                    output_path, input_frames, sr_frames, original_frames = result
                    status_msg = "处理完成！已创建低分辨率输入进行超分辨率演示。"
                    original_frames_to_show = original_frames if original_frames else []
                else:
                    output_path, input_frames, sr_frames, original_frames = result
                    status_msg = "处理完成！直接对原始视频进行超分辨率处理。"
                    original_frames_to_show = []
                
                return output_path, input_frames, sr_frames, original_frames_to_show, status_msg
            except Exception as e:
                import traceback
                traceback.print_exc()
                return None, [], [], [], f"处理失败: {str(e)}"
        
        process_btn.click(
            process_video_wrapper,
            inputs=[video_input, model_selector, max_frames, create_lr_checkbox, downscale_method],
            outputs=[video_output, input_gallery, sr_gallery, original_gallery, status_text]
        )
        
        # 添加说明
        with gr.Accordion("ℹ️ 修复说明", open=False):
            gr.Markdown("""
            ### 修复的问题：
            
            1. **数据范围问题**: 使用 `data_utils.canonicalize()` 正确归一化数据到[0,1]范围
            2. **数据格式问题**: 确保输入tensor格式正确 (T, H, W, C)
            3. **模型输入问题**: 按照项目要求处理数据格式
            
            ### 主要修复：
            - ✅ 使用正确的数据归一化函数
            - ✅ 确保数据格式符合模型要求
            - ✅ 添加详细的调试信息
            - ✅ 改进错误处理
            
            ### 使用建议：
            - 如果仍有问题，请运行 `python debug_vsr.py` 进行诊断
            - 确保预训练模型文件存在且正确
            - 检查输入视频质量
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