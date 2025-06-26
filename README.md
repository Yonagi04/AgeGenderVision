# 基于深度学习的年龄与性别预测系统

本项目实现了一个基于深度学习的年龄与性别多任务预测系统，使用 UTKFace 数据集，采用 PyTorch 训练 ResNet 多任务模型，可对人脸图片、视频、摄像头画面（基于OpenCV和YOLO）进行年龄回归与性别分类。支持图形界面（PyQt5）和命令行双模式操作。

## CUDA 加速支持

本项目支持使用 NVIDIA GPU 进行 CUDA 加速。只需在支持 CUDA 的环境下安装对应的 PyTorch 版本，程序会自动检测并优先使用 GPU 进行训练和推理，大幅提升速度。

- 若设备支持 CUDA，训练和预测时会自动输出 `使用设备: cuda`。
- 若无 GPU，则自动回退到 CPU。
- 安装 CUDA 版本 PyTorch 参考：https://pytorch.org/get-started/locally/

## 目录结构

```
face_recognition/
├── assets                             # 图形界面依赖资源
├── convention                         # 图形界面Panel层与Service层传输类
├── services                           # 图形界面服务类
├── panels                             # 图形界面面板类
├── threads                            # 图形界面线程类
├── utils                              # 图形界面工具类
├── widgets                            # 自建QT界面组件
├── check_and_deduplicate_utkface.py   # UTKFace数据去重脚本 （可独立运行）
├── train_age_gender_multitask.py      # 年龄性别多任务模型训练脚本 （可独立运行）
├── camera_predict.py                  # 摄像头采集预测 （可独立运行）
├── video_predict.py                   # 视频预测（可独立运行）
├── photo_predict.py                   # 图片预测 （可独立运行）
├── main.py                            # 命令行主程序 
├── qt5_main.py                        # 图形界面主程序
├── yolov8m-face.pt                    # YOLOv8m-face模型
├── requirements.txt                   # 环境依赖
└── data/UTKFace/                      # UTKFace数据集相关目录（需要自行创建）
```

## 主要功能

- **年龄预测**：对输入人脸图片、视频、摄像头画面进行年龄回归。
- **性别预测**：对输入人脸图片、视频、摄像头画面进行性别二分类（0=男，1=女）。
- **UTKFace数据处理**：自动去重并整理 UTKFace 数据集。
- **中文结果显示**：支持在 OpenCV 窗口中以中文显示预测结果（需 Pillow 和中文字体支持）。
- **开发者模式**：支持开发者模式，自动保存详细日志，便于调试和排查问题。可通过命令行参数 `--dev` 或 `--developer` 启动开发者模式，或在主界面切换。
- **模型管理**：支持模型的查看、删除。图形界面可支持搜索模型、对模型排序、将模型列表信息及元信息导出为JSON或CSV文件、导出模型、上传模型、修改模型信息、查看模型结构（需 Netron 支持）等功能
- **模型对比（仅图形界面模式支持）**: 支持同时对比两个模型的基本信息、元信息、训练参数等模型信息
- **训练模型选择**：支持手动选择训练模型，可选 `ResNet18`、`ResNet34`、`ResNet50` 模型
- **训练中断**：训练过程中可随时按 Q 键安全中断并保存当前模型。
- **数据集自动去重**：一键去重整理 UTKFace 数据集，避免重复图片影响训练。
- **图形界面支持**：提供基于 PyQt5 的图形界面，扁平化设计，支持浅色/深色模式，支持所有功能。调用子脚本均采用多线程异步执行，界面不卡死；训练日志和进度条实时显示，支持随时停止训练。
- **命令行模式支持**：支持传统命令行交互，保留核心功能。

## 环境依赖

- Python 3.8+
- torch
- torchvision
- pillow
- opencv-python
- tqdm
- keyboard
- PyQt5
- ultralytics
- netron（如不使用“查看模型结构”功能，可不安装）

安装依赖：
```bash
pip install -r requirements.txt
```

## 数据准备
1. 下载 [UTKFace 数据集](https://susanqq.github.io/UTKFace/)，解压到 `data/UTKFace/archive/`。如果使用自定义的数据集，将它放到 `data` 文件夹下即可。
2. 运行 `check_and_deduplicate_utkface.py`，或在主界面选择数据集去重功能，可自动去重并整理图片到 `data/UTKFace/cleaned/`。如果使用自定义数据集，则需要调整 `check_and_deduplicate_utkface.py` 以实现自动去重。

## 运行

### 主程序图形界面模式（推荐）

```bash
python qt5_main.py
```

### 主程序命令行模式（不再推荐使用）

```bash
python main.py
```

或以开发者模式运行（自动保存日志）:
```bash
python main.py --dev
```

- 选择“训练模型”可自定义参数并训练，支持训练中按 Q 键随时中断，参数输入支持按 Q 键返回上一步。
- 选择“查看训练模型”可查看当前所有模型文件。
- 选择“预测图片”会弹出文件选择窗口，选择图片后自动检测人脸并显示预测结果。
- 选择“预测视频”可调用摄像头实时预测，按 Q 键退出。
- 选择“数据集自动去重”可自动整理并去重 UTKFace 数据集。
- 选择“删除训练模型”可安全删除模型，需二次确认。
- 选择“开发者模式”可在主界面切换开发者模式，开发者模式下所有异常会自动保存到日志文件。

### 其他脚本

除了 `main.py`、`qt5_main.py` 主程序外，其他脚本文件均可独立运行，也可以被其他的程序调用。

## 常见问题
- 使用命令行模式报错，但使用图形界面模式不报错：命令行模式已经不再维护，将在未来的几个版本被淘汰。为了适应图形模式的功能，底层执行脚本做了较大的改动，可能不再适配命令行模式，请注意。
- 使用命令行模式训练或预测时报错可通过开发者模式自动保存详细日志到 `error_log.log`，便于排查。
- 图形界面暂不支持开发者模式的启用。如果在使用图形界面时，遇到缺陷或非预期内结果且该问题能够稳定复现，请携带日志、附有有用信息的截图提出 issue。
- OpenCV 窗口中文乱码：请用 Pillow 绘制中文，详见 `photo_predict.py` 和 `video_predict.py` 示例。
- 运行时缺少字体文件报错：请把可用的字体文件放到 `data` 文件夹下，默认使用的字体为simhei.tff。如果使用其他字体，请修改 `photo_predict.py` 和 `video_predict.py`。
- 运行报错提示：
```txt
Traceback (most recent call last):
  File "video_predict.py", line 84, in main
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
  File "Python3.12\Lib\site-packages\torch\nn\modules\module.py", line 2584, in load_state_dict
    raise RuntimeError
RuntimeError: Error(s) in loading state_dict for MultiTaskResNet:
	Missing key(s) in state_dict
```
请确认所选择的模型类型是否与模型匹配，如果确认模型类型选择无误且该问题能稳定复现，请携带错误日志提出 issue。
- dlib/face_recognition 仅为可选依赖，主流程不依赖。
- 训练效果不佳：请在 `train_age_gender_multitask.py` 或 `main.py` 中 `训练模型` 功能的训练参数项，适当地调整训练轮数、学习率和批量大小；如果在调整后训练效果依旧不佳，请改用更加复杂的模型，如ResNet34、ResNet50等

## 致谢
- [UTKFace Dataset](https://susanqq.github.io/UTKFace/)
- [PyTorch](https://pytorch.org/)
- [Ultralytics](https://github.com/ultralytics/ultralytics)
- [Netron](https://netron.app/)

---
如有问题欢迎提 issue 或联系作者，觉得做得不错的话，就点个 Star 支持一下作者吧。
