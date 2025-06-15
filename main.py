import os
import sys
import tkinter as tk
from tkinter import filedialog

def list_models():
    print("当前已训练模型：")
    models = [f for f in os.listdir('.') if f.endswith('.pth')]
    if not models:
        print("  暂无模型文件")
    else:
        for i, m in enumerate(models):
            print(f"  [{i+1}] {m}")
    return models

def get_input_with_default(prompt, default, type_func, validator=None):
    while True:
        s = input(f"{prompt} (默认: {default}): ").strip()
        if not s:
            return default
        try:
            value = type_func(s)
            if validator and not validator(value):
                print("输入不合法，请重新输入。")
                continue
            return value
        except Exception:
            print("输入格式错误，请重新输入。")

def train():
    print("请输入训练参数，直接回车使用默认值：")
    batch_size = get_input_with_default("Batch size", 64, int, lambda x: x > 0)
    epochs = get_input_with_default("Epochs", 10, int, lambda x: x > 0)
    lr = get_input_with_default("Learning rate", 1e-3, float, lambda x: x > 0)
    img_size = get_input_with_default("Image size", 224, int, lambda x: x > 0)
    data_dir = input("数据集目录 (默认: data/UTKFace/cleaned): ").strip() or 'data/UTKFace/cleaned'
    model_path = input("模型保存路径 (默认: age_gender_multitask_resnet18.pth): ").strip() or 'age_gender_multitask_resnet18.pth'
    # 传递参数给训练脚本
    cmd = (f"{sys.executable} train_age_gender_multitask.py "
           f"--batch_size {batch_size} --epochs {epochs} --lr {lr} "
           f"--img_size {img_size} --data_dir \"{data_dir}\" --model_path \"{model_path}\"")
    print("开始训练...\n命令:", cmd)
    os.system(cmd)

def select_model():
    models = list_models()
    if not models:
        print("未检测到模型文件，无法进行预测。")
        return None
    choice = input(f"请选择模型（输入序号，回车默认[{models[0]}]）：").strip()
    if not choice:
        return models[0]
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(models):
            return models[idx]
        else:
            print("输入序号超出范围，默认使用第一个模型。")
            return models[0]
    except Exception:
        print("输入无效，默认使用第一个模型。")
        return models[0]

def photo_predict():
    model_path = select_model()
    if not model_path:
        return
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="选择图片文件",
        filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")]
    )
    root.destroy()
    if not file_path:
        print("未选择图片文件，已取消。")
        return
    print("将进行图片预测。")
    os.system(f"{sys.executable} photo_predict.py --model_path \"{model_path}\" --img_path \"{file_path}\"")

def video_predict():
    model_path = select_model()
    if not model_path:
        return
    print("将进行视频预测，按 Q 停止预测")
    os.system(f"{sys.executable} video_predict.py --model_path \"{model_path}\"")

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def main():
    while True:
        clear_screen()
        print("[1] 训练模型")
        print("[2] 预测图片")
        print("[3] 预测视频")
        print("[0] 退出")
        choice = input("请选择操作（输入数字）：").strip()
        if choice == '1':
            train()
        elif choice == '2':
            photo_predict()
        elif choice == '3':
            video_predict()
        elif choice == '0':
            print("已退出。")
            break
        else:
            print("无效选择，请重新输入。\n")

if __name__ == "__main__":
    main()