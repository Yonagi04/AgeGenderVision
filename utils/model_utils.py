import os
import json
import datetime
import hashlib
import shutil

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_INFO_FILE = os.path.join(BASE_DIR, 'data', 'models.json')

def refresh_model_list():
    if not os.path.exists(MODELS_INFO_FILE):
        return []
    with open(MODELS_INFO_FILE, 'r', encoding='utf-8') as f:
        info = json.load(f)
    return list(info.keys())

def get_model_type(model_name):
    if os.path.exists(MODELS_INFO_FILE):
        with open(MODELS_INFO_FILE, 'r', encoding='utf-8') as f:
            info = json.load(f)
        t = info.get(model_name, {}).get("model_type")
        if t:
            return t
    return 'resnet18'

def get_model_dir(model_name):
    if os.path.exists(MODELS_INFO_FILE):
        with open(MODELS_INFO_FILE, 'r', encoding='utf-8') as f:
            info = json.load(f)
        d = info.get(model_name, {}).get("model_dir")
        if d:
            return d
    return None

def check_pth_file(pth_path):
    import torch
    try:
        torch.load(pth_path, map_location='cpu', weights_only=True)
        return True
    except Exception:
        return False

def load_state_dict_from_file(pth_path):
    import torch
    try:
        data = torch.load(pth_path, map_location="cpu", weights_only=True)
    except Exception as e:
        data = torch.load(pth_path, map_location="cpu")
    if isinstance(data, dict) and "state_dict" in data:
        return data["state_dict"]
    return data

def try_load_model(state_dict, model_fn):
    model = model_fn()
    try:
        model.load_state_dict(state_dict, strict=True)
        return True, 0
    except RuntimeError:
        return False, 1

def get_resnet_type(pth_path: str):
    from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152

    state_dict = load_state_dict_from_file(pth_path)

    candidates = {
        "resnet18": resnet18,
        "resnet34": resnet34,
        "resnet50": resnet50,
        "resnet101": resnet101,
        "resnet152": resnet152,
    }

    best_match = None
    min_mismatch = float('inf')

    for name, fn in candidates.items():
        success, mismatch = try_load_model(state_dict, fn)
        if success:
            return name
        elif mismatch < min_mismatch:
            min_mismatch = mismatch
            best_match = name

    return best_match

# 创建模型唯一存储路径
def gen_model_dir(model_type, model_name, save_dir="models"):
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    unique_str = f"{model_name}-{model_type}-{timestamp}"
    unique_id = hashlib.md5(unique_str.encode()).hexdigest()[:6]
    if model_name.endswith('.pth'):
        model_name = model_name[:-4]
    elif model_name.endswith('.pt'):
        model_name = model_name[:-3]
    folder_name = f"{date_str}-{model_name}-{unique_id}"
    folder_path = os.path.join(save_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

# 保存上传模型
def save_model(model_type, model_name, model):
    try:
        model_dir = gen_model_dir(model_type, model_name)
        model_save_path = os.path.join(model_dir, model_name)
        shutil.copy(model, model_save_path)
        meta = {
            "model_name": model_name,
            "model_type": model_type,
            "created_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": "UTKFace",
            "epochs": "",
            "batch_size": "",
            "img_size": "",
            "eval_result": {
                "val_age_loss": "",
                "val_gender_loss": "",
                "val_acc": "",
                "age_scatter_image": "",
                "gender_confusion_image": ""
            }
        }
        meta_path = os.path.join(model_dir, "meta.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        if os.path.exists(MODELS_INFO_FILE):
            with open(MODELS_INFO_FILE, 'r', encoding='utf-8') as f:
                info = json.load(f)
        else:
            info = {}
        info[model_name] = {
            "model_name": model_name,
            "model_type": model_type,
            "model_dir": model_dir,
            "created_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(MODELS_INFO_FILE, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        return False