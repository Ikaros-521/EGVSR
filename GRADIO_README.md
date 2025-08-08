# 视频超分辨率 Gradio Web应用

这是一个基于EGVSR项目的视频超分辨率Web应用程序，使用Gradio构建用户友好的界面。

## 功能特点

- 🎬 **视频超分辨率**: 支持多种视频格式的超分辨率处理
- 🤖 **多种模型**: 支持EGVSR、TecoGAN、FRVSR等先进模型
- 🖼️ **帧对比**: 实时显示低分辨率和超分辨率帧的对比
- ⚡ **快速处理**: 优化的推理流程，支持GPU加速
- 🌐 **Web界面**: 基于Gradio的现代化Web界面

## 安装依赖

```bash
pip install -r gradio_requirements.txt
```

## 运行应用

```bash
python simple_vsr_app.py
```

应用将在 `http://localhost:7860` 启动。

## 使用说明

### 1. 视频处理
1. 点击"选择文件"上传视频
2. 选择超分辨率模型（推荐EGVSR）
3. 设置最大处理帧数（建议20-30帧）
4. 点击"开始处理"
5. 等待处理完成，查看结果

### 2. 模型选择
- **EGVSR**: 高效且通用的视频超分辨率，推荐使用
- **TecoGAN**: 时间一致性GAN，注重视频时序连贯性
- **FRVSR**: 帧循环视频超分辨率，利用时序信息

### 3. 参数设置
- **最大处理帧数**: 控制处理的视频长度，帧数越多处理时间越长
- **模型选择**: 不同模型有不同的特点和适用场景

## 支持的视频格式

- MP4
- AVI
- MOV
- 其他常见视频格式

## 系统要求

- Python 3.7+
- CUDA支持的GPU（推荐）
- 至少4GB内存
- 足够的磁盘空间存储处理结果

## 注意事项

1. **预训练模型**: 确保预训练模型文件存在于 `pretrained_models` 目录
2. **GPU内存**: 处理高分辨率视频时需要足够的GPU内存
3. **处理时间**: 处理时间取决于视频长度、分辨率和选择的模型
4. **输出格式**: 输出视频为MP4格式

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查预训练模型文件是否存在
   - 确认模型文件路径正确

2. **内存不足**
   - 减少最大处理帧数
   - 使用CPU模式（较慢但内存需求更少）

3. **视频无法播放**
   - 检查视频格式是否支持
   - 确认视频文件未损坏

### 错误信息

- `CUDA out of memory`: GPU内存不足，减少处理帧数或使用CPU
- `Model not found`: 预训练模型文件缺失，请下载相应模型
- `Video format not supported`: 视频格式不支持，请转换为MP4格式

## 技术架构

```
simple_vsr_app.py
├── SimpleVSRApp类
│   ├── load_model() - 模型加载
│   ├── process_video() - 视频处理
│   └── 其他辅助方法
├── create_interface() - Gradio界面创建
└── 主程序入口
```

## 开发说明

### 添加新模型

1. 在 `model_configs` 中添加新模型配置
2. 确保预训练模型文件存在
3. 更新模型选择下拉菜单

### 自定义界面

修改 `create_interface()` 函数来自定义Gradio界面布局和功能。

## 许可证

本项目基于MIT许可证，详见LICENSE文件。

## 致谢

- [EGVSR-PyTorch](https://github.com/Thmen/EGVSR) - 原始项目
- [Gradio](https://gradio.app/) - Web界面框架
- [PyTorch](https://pytorch.org/) - 深度学习框架 