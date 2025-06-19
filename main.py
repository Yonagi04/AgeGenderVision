import os
import sys
import subprocess
import tkinter as tk
import datetime
import traceback
import argparse
import json
from tkinter import filedialog

parser = argparse.ArgumentParser()
parser.add_argument('--dev', '--developer', action='store_true', help='以开发者模式运行')
args, unknown = parser.parse_known_args()

DEVELOPER_MODE = args.dev
ERROR_LOG_FILE = 'error_log.log'
RESULE_LOG_FILE = 'result_log.log'
MODELS_INFO_FILE = 'data/models.json'

def list_models():
    print("当前已训练模型：")
    models = [f for f in os.listdir('.') if f.endswith('.pth')]
    if not models:
        print("  暂无模型文件")
    else:
        for i, m in enumerate(models):
            print(f"  [{i+1}] {m}")
    return models

def update_model_info(model_path, model_type, info_file=MODELS_INFO_FILE):
    try:
        if os.path.exists(info_file):
            with open(info_file, 'r', encoding='utf-8') as f:
                info = json.load(f)
        else:
            info = {}
        info[model_path] = model_type
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
    except Exception as e:
        save_error_log(e)

def get_model_type(model_path, info_file=MODELS_INFO_FILE):
    if os.path.exists(info_file):
        with open(info_file, 'r', encoding='utf-8') as f:
            info = json.load(f)
        return info.get(model_path)
    return None

def delete_model_info(model_path, info_file=MODELS_INFO_FILE):
    if os.path.exists(info_file):
        with open(info_file, 'r', encoding='utf-8') as f:
            info = json.load(f)
        if model_path in info:
            del info[model_path]
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(info, f, ensure_ascii=False, indent=2)

def train():
    try:
        print("请输入训练参数，直接回车使用默认值，输入 Q 返回上一个参数：")
        params = [
            {"name": "Batch size", "default": 64, "type": int, "validator": lambda x: x > 0},
            {"name": "Epochs", "default": 10, "type": int, "validator": lambda x: x > 0},
            {"name": "Learning rate", "default": 1e-3, "type": float, "validator": lambda x: x > 0},
            {"name": "Image size", "default": 224, "type": int, "validator": lambda x: x > 0},
            {"name": "数据集目录", "default": 'data/UTKFace/cleaned', "type": str, "validator": None},
            {"name": "模型类型(resnet18/resnet34/resnet50)", "default": "resnet18", "type": str, "validator": lambda x: x in ["resnet18", "resnet34", "resnet50"]},
            {"name": "模型保存路径", "default": 'age_gender_multitask_resnet18.pth', "type": str, "validator": None},
        ]
        values = [p["default"] for p in params]
        idx = 0
        while idx < len(params):
            p = params[idx]
            prompt = f"{p['name']} (默认: {p['default']}): "
            s = input(prompt).strip()
            if s.lower() == 'q':
                if idx == 0:
                    return
                idx -= 1
                # 显示上一个参数的当前值，并允许重新输入
                prev_p = params[idx]
                prev_val = values[idx]
                while True:
                    prev_input = input(f"{prev_p['name']} (当前: {prev_val}, 默认: {prev_p['default']}): ").strip()
                    if prev_input.lower() == 'q':
                        if idx == 0:
                            return
                        idx -= 1
                        prev_p = params[idx]
                        prev_val = values[idx]
                        continue
                    if not prev_input:
                        break
                    try:
                        val = prev_p["type"](prev_input)
                        if prev_p["validator"] and not prev_p["validator"](val):
                            print("输入不合法，请重新输入。")
                            continue
                        values[idx] = val
                        break
                    except Exception:
                        print("输入格式错误，请重新输入。")
                continue
            if not s:
                idx += 1
                continue
            try:
                val = p["type"](s)
                if p["validator"] and not p["validator"](val):
                    print("输入不合法，请重新输入。")
                    continue
                values[idx] = val
                idx += 1
            except Exception:
                print("输入格式错误，请重新输入。")
        batch_size, epochs, lr, img_size, data_dir, model_type, model_path = values
        if not model_path.lower().endswith('.pth'):
            model_path += '.pth'
        cmd = (f"{sys.executable} train_age_gender_multitask.py "
            f"--batch_size {batch_size} --epochs {epochs} --lr {lr} "
            f"--img_size {img_size} --data_dir \"{data_dir}\" --model_type \"{model_type}\" --model_path \"{model_path}\"")
        print("即将开始训练...\n")
        env = os.environ.copy()
        env["DEVELOPER_MODE"] = "1" if DEVELOPER_MODE else "0"
        subprocess.run(cmd, shell=True, check=True, env=env)
        update_model_info(model_path, model_type)
    except Exception as e:
        save_error_log(e)

def select_model():
    models = list_models()
    if not models:
        input("按任意键返回主菜单")
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

def select_type():
    model_types = [
        'resnet18',
        'resnet34',
        'resnet50'
    ]
    print("\n可用的模型类型：")
    for i, m in enumerate(model_types, 1):
        print(f"[{i}] {m}")
    choice = input(f"请选择模型类型（输入序号，回车默认[{model_types[0]}]）：").strip()
    if not choice:
        return model_types[0]
    try:
        idx = int(choice) - 1
        if 0 <= idx < len(model_types):
            return model_types[idx]
        else:
            print("输入序号超出范围，默认使用第一个模型类型。")
            return model_types[0]
    except Exception:
        print("输入无效，默认使用第一个模型类型。")
        return model_types[0]

def photo_predict():
    try:
        ans = input("是否需要进行多模型预测？可同时比较两个模型的结果。（Y/N）：").strip().lower()
        if ans == 'y':
            models = list_models()
            if not models:
                input("按任意键返回主菜单")
                return
            max_num = len(models)
            while True:
                sel = input(f"请输入要比较的模型序号（用英文逗号分隔，最多{max_num}个，回车取消）：").strip()
                if not sel:
                    print("已取消多模型预测。")
                    return
                try:
                    idxs = [int(x.strip()) - 1 for x in sel.split(',') if x.strip()]
                    if not idxs or any(i < 0 or i >= max_num for i in idxs):
                        print("输入序号有误，请重新输入")
                        continue
                    idxs = list(dict.fromkeys(idxs))
                    if len(idxs) > max_num:
                        print(f"最多只能选择{max_num}个模型。")
                        continue
                    break
                except Exception:
                    print("输入格式有误，请重新输入。")
            model_types = []
            for i in idxs:
                type = get_model_type(models[i])
                if type is None:
                    type = select_type()
                model_types.append(type)
            root = tk.Tk()
            root.withdraw()
            img_path = filedialog.askopenfilename(
                title="请选择要预测的图片", 
                filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")]
            )
            root.destroy()
            if not img_path:
                print("未选择图片文件，已取消。")
                input("按任意键退出。")
                return
            for i, idx in enumerate(idxs):
                model_path = models[idx]
                model_type = model_types[i]
                print(f"正在用模型 {model_path} ({model_type}) 进行预测...")
                env = os.environ.copy()
                env["DEVELOPER_MODE"] = "1" if DEVELOPER_MODE else "0"
                env["IS_SUBPROCESS"] = "1"
                cmd = f"{sys.executable} photo_predict.py --model_path \"{model_path}\" --model_type \"{model_type}\" --img_path \"{img_path}\""
                result = subprocess.run(cmd, shell=True, check=True, env=env, capture_output=True, text=True)
                predict_output = result.stdout.strip()
                with open(RESULE_LOG_FILE, 'a', encoding='utf-8') as f:
                    f.write(
                        f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n"
                        f"预测模型: {model_path}\n"
                        f"模型类型: {model_type}\n"
                        f"预测图片: {img_path}\n"
                        f"预测结果:\n{predict_output}\n"
                        f"{'-'*40}\n"
                    )
                if i < len(idxs) - 1:
                    user_input = input(f"基于 {model_path} ({model_type}) 的预测已完成。\n按任意键以继续下一个模型的预测，或输入 Q 结束预测：").strip().lower()
                    if user_input == 'q':
                        break
            print(f"\n多模型比较已经完成，详细结果请查看 {RESULE_LOG_FILE}")
            input("按任意键返回主菜单")
        else:
            model_path = select_model()
            if not model_path:
                return
            model_type = get_model_type(model_path)
            if model_type is None:
                print("未在数据文件中检测到模型，请手动选择模型类型。\n")
                model_type = select_type()
                if not model_type:
                    return
            print("将进行图片预测。")
            env = os.environ.copy()
            env["DEVELOPER_MODE"] = "1" if DEVELOPER_MODE else "0"
            cmd = f"{sys.executable} photo_predict.py --model_path \"{model_path}\" --model_type \"{model_type}\""
            subprocess.run(cmd, shell=True, check=True, env=env)
    except Exception as e:
        save_error_log(e)

def video_predict():
    try:
        model_path = select_model()
        if not model_path:
            return
        model_type = get_model_type(model_path)
        if model_type is None:
            print("未在数据文件中检测到模型，请手动选择模型类型。\n")
            model_type = select_type()
            if not model_type:
                return
        print("将进行视频预测，按 Q 停止预测")
        env = os.environ.copy()
        env["DEVELOPER_MODE"] = "1" if DEVELOPER_MODE else "0"
        cmd = f"{sys.executable} video_predict.py --model_path \"{model_path}\" --model_type \"{model_type}\""
        subprocess.run(cmd, shell=True, check=True, env=env)
    except Exception as e:
        save_error_log(e)

def check_and_deduplicate_utkface():
    try:
        ans = input("确定要对数据集进行去重吗？（Y/N）：").strip().lower()
        if ans == 'y':
            print("将对UTKFace数据集进行自动去重处理。")
            env = os.environ.copy()
            env["DEVELOPER_MODE"] = "1" if DEVELOPER_MODE else "0"
            cmd = f"{sys.executable} check_and_deduplicate_utkface.py"
            subprocess.run(cmd, shell=True, check=True, env=env)
    except Exception as e:
        save_error_log(e)

def delete_model():
    try:
        models = list_models()
        if not models:
            input("按任意键返回主菜单")
            return
        choice = input(f"请选择要删除的模型（输入序号以删除，回车取消）：").strip()
        if not choice:
            return
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                model_to_delete = models[idx]
            else:
                print("输入序号超出范围，返回主菜单。")
                input("按任意键返回主菜单")
                return
        except Exception:
            print("输入无效，返回主菜单")
            input("按任意键返回主菜单")
            return
        confirm_name = input(f"请完整输入要删除的模型文件名以确认（如：{model_to_delete}）：").strip()
        if confirm_name != model_to_delete:
            print("模型名称输入错误，未删除任何文件。")
            input("按任意键返回主菜单")
            return
        ans = input(f"即将永久删除模型 {model_to_delete} ，确定删除？(Y/N): ").strip().lower()
        if ans == 'y':
            os.remove(model_to_delete)
            delete_model_info(model_to_delete)
            print(f"模型 {model_to_delete} 已成功删除。")
            input("按任意键返回主菜单")
    except Exception as e:
        save_error_log(e)

def clear_screen():
    if not DEVELOPER_MODE:
        os.system('cls' if os.name == 'nt' else 'clear')

def save_error_log(e):
    if DEVELOPER_MODE:
        err_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(ERROR_LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"\n[{err_time}] {repr(e)}\n")
            f.write(traceback.format_exc())
            f.write("\n")
        print(f"发生错误，已记录到 {ERROR_LOG_FILE}")

def developer_mode():
    global DEVELOPER_MODE
    if DEVELOPER_MODE:
        ans = input("已处于开发者模式，是否退出？(Y/N): ").strip().lower()
        if ans == 'y':
            DEVELOPER_MODE = False
            print("已退出开发者模式。")
            input("按任意键返回主菜单")
    else:
        ans = input("是否进入开发者模式？(Y/N): ").strip().lower()
        if ans == 'y':
            DEVELOPER_MODE = True
            print("已进入开发者模式，所有错误将自动记录到日志文件中。")
            input("按任意键返回主菜单")

def main():
    while True:
        clear_screen()
        print("[1] 训练模型")
        print("[2] 查看训练模型")
        print("[3] 预测图片")
        print("[4] 预测视频")
        print("[5] 数据集自动去重（适用于UTKFace）")
        print("[6] 开发者模式")
        print("[7] 删除训练模型")
        print("[0] 退出")
        choice = input("请选择操作（输入数字）：").strip()
        if choice == '1':
            clear_screen()
            train()
        elif choice == '2':
            clear_screen()
            list_models()
            input("按任意键返回主菜单")
        elif choice == '3':
            clear_screen()
            photo_predict()
        elif choice == '4':
            clear_screen()
            video_predict()
        elif choice == '5':
            clear_screen()
            check_and_deduplicate_utkface()
        elif choice == '6':
            clear_screen()
            developer_mode()
        elif choice == '7':
            clear_screen()
            delete_model()
        elif choice == '0':
            break
        else:
            print("无效选择，请重新输入。\n")

if __name__ == "__main__":
    main()