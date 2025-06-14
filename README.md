# 基于深度学习的年龄与性别预测系统

本项目实现了一个基于深度学习的年龄与性别多任务预测系统，使用 UTKFace 数据集，采用 PyTorch 训练 ResNet18 多任务模型，可对人脸图片进行年龄回归与性别分类。

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
├── predict.py                        # 年龄性别预测与中文显示示例
└── data/UTKFace/                     # UTKFace数据集相关目录（需要自行创建）
```

## 主要功能

- **年龄预测**：对输入人脸图片进行年龄回归。
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

## 训练年龄性别多任务模型
```bash
python train_age_gender_multitask.py
```
训练完成后会生成 `age_gender_multitask_resnet18.pth`。

## 年龄性别预测与中文显示
- 参考 `predict.py`，可对单张图片进行年龄与性别预测，并支持中文结果显示。
- 如需显示中文，需指定本地字体路径（如 `C:/Windows/Fonts/simhei.ttf`）。

## 常见问题
- OpenCV 窗口中文乱码：请用 Pillow 绘制中文，详见 `predict.py` 示例。
- dlib/face_recognition 仅为可选依赖，主流程不依赖。

## 致谢
- [UTKFace Dataset](https://susanqq.github.io/UTKFace/)
- [PyTorch](https://pytorch.org/)

---
如有问题欢迎提 issue 或联系作者。
