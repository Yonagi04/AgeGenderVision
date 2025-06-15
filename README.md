# 基于深度学习的年龄与性别预测系统

本项目实现了一个基于深度学习的年龄与性别多任务预测系统，使用 UTKFace 数据集，采用 PyTorch 训练 ResNet18 多任务模型，可对人脸图片或视频（基于OpenCV）进行年龄回归与性别分类。

## CUDA 加速支持

本项目支持使用 NVIDIA GPU 进行 CUDA 加速。只需在支持 CUDA 的环境下安装对应的 PyTorch 版本，程序会自动检测并优先使用 GPU 进行训练和推理，大幅提升速度。

- 若设备支持 CUDA，训练和预测时会自动输出 `Using device: cuda`。
- 若无 GPU，则自动回退到 CPU。
- 安装 CUDA 版本 PyTorch 参考：https://pytorch.org/get-started/locally/

## 目录结构

```
face_recognition/
├── check_and_deduplicate_utkface.py   # UTKFace数据去重脚本
├── train_age_gender_multitask.py      # 年龄性别多任务模型训练脚本
├── video_predict.py                   # 视频预测
├── photo_predict.py                   # 图片预测
├── main.py                            # 主程序 
└── data/UTKFace/                     # UTKFace数据集相关目录（需要自行创建）
```

## 主要功能

- **年龄预测**：对输入人脸图片或视频进行年龄回归。
- **性别预测**：对输入人脸图片进行性别二分类（0=男，1=女）。
- **UTKFace数据处理**：自动去重并整理 UTKFace 数据集。
- **中文结果显示**：支持在 OpenCV 窗口中以中文显示预测结果（需 Pillow 和中文字体支持）。

## 环境依赖

- Python 3.8+
- torch
- torchvision
- pillow
- opencv-python
- tqdm

安装依赖：
```bash
pip install torch torchvision pillow opencv-python tqdm
```

## 数据准备
1. 下载 [UTKFace 数据集](https://susanqq.github.io/UTKFace/)，解压到 `data/UTKFace/archive/`。如果使用自定义的数据集，将它放到 `data` 文件夹下即可。
2. 运行 `check_and_deduplicate_utkface.py`，自动去重并整理图片到 `data/UTKFace/cleaned/`。如果使用自定义数据集，则需要调整 `check_and_deduplicate_utkface.py` 以实现自动去重。

## 运行

```bash
python main.py
```

- 选择“训练模型”可自定义参数并训练，支持训练中按 Q 键随时中断。
- 选择“预测图片”会弹出文件选择窗口，选择图片后自动检测人脸并显示预测结果。
- 选择“预测视频”可调用摄像头实时预测，按 Q 键退出。


## 常见问题
- OpenCV 窗口中文乱码：请用 Pillow 绘制中文，详见 `photo_predict.py` 和 `video_predict.py` 示例。
- 运行时缺少字体文件报错：请把可用的字体文件放到 `data` 文件夹下，默认使用的字体为simhei.tff。如果使用其他字体，请修改 `photo_predict.py` 和 `video_predict.py`。
- dlib/face_recognition 仅为可选依赖，主流程不依赖。
- 训练效果不佳：请在 `train_age_gender_multitask.py` 的训练参数项，适当地调整训练轮数、学习率和批量大小；如果在调整后训练效果依旧不佳，请改用更加复杂的模型，如ResNet34、ResNet50等

## 致谢
- [UTKFace Dataset](https://susanqq.github.io/UTKFace/)
- [PyTorch](https://pytorch.org/)

---
如有问题欢迎提 issue 或联系作者，觉得做得不错的话，就点个Star支持一下作者吧。
