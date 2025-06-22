import os
import json

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