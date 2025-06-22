import os
import json

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_INFO_FILE = os.path.join(BASE_DIR, 'data', 'models.json')
MODELS_DIR = BASE_DIR

def refresh_model_list():
    return [f for f in os.listdir(MODELS_DIR) if f.endswith('.pth')]

def get_model_type(model_path):
    if os.path.exists(MODELS_INFO_FILE):
        with open(MODELS_INFO_FILE, 'r', encoding='utf-8') as f:
            info = json.load(f)
        t = info.get(model_path)
        if t:
            return t
    return 'resnet18'