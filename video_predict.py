import os
import sys
import cv2
import argparse
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageFont, ImageDraw
import numpy as np
import traceback
import datetime
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='age_gender_multitask_resnet18.pth', help='模型路径')
parser.add_argument('--model_type', type=str, default='resnet18', help='模型类型', choices=['resnet18', 'resnet34', 'resnet50'])
args = parser.parse_args()
model_path = args.model_path
model_type = args.model_type

LOG_FILE = 'error_log.log'
YOLO_PATH = 'yolov8m-face.pt'
if __name__ == "__main__" and os.environ.get("DEVELOPER_MODE") is None:
    DEVELOPER_MODE = True
else:
    DEVELOPER_MODE = os.environ.get("DEVELOPER_MODE", "0") == "1"

def save_error_log(e):
    if DEVELOPER_MODE:
        err_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(LOG_FILE, 'a', encoding='utf-8') as f:
            f.write(f"\n[{err_time}] {repr(e)}\n")
            f.write(traceback.format_exc())
            f.write("\n")
        print(f"发生错误，已记录到 {LOG_FILE}")
    sys.exit(1)

def cv2_add_chinese_text(img, text, position, font_size=20, color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype("/data/simhei.ttf", font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

class MultiTaskResNet(nn.Module):
    def __init__(self, model_type='resnet18'):
        super().__init__()
        if model_type == 'resnet18':
            self.backbone = models.resnet18(weights=None)
        elif model_type == 'resnet34':
            self.backbone = models.resnet34(weights=None)
        elif model_type == 'resnet50':
            self.backbone = models.resnet50(weights=None)
        else:
            raise ValueError(f"不支持的模型类型: {model_type}")
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.age_head = nn.Linear(in_features, 1)
        self.gender_head = nn.Linear(in_features, 2)

    def forward(self, x):
        feat = self.backbone(x)
        age = self.age_head(feat)
        gender = self.gender_head(feat)
        return age, gender

transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

def predict(img, model, device):
    model.eval()
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_age, pred_gender = model(img)
        pred_age = pred_age.item()
        pred_gender = pred_gender.softmax(1).argmax(1).item()
    return pred_age, pred_gender

def main():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MultiTaskResNet(model_type=model_type).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))

        cap = cv2.VideoCapture(0)
        
        yolo_device = 0 if torch.cuda.is_available() else 'cpu'
        yolo_face = YOLO(YOLO_PATH, verbose=False)
        yolo_face.to(yolo_device)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = yolo_face(frame, device=yolo_device, verbose=False)
            for box in results[0].boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = map(int, box[:4])
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                if x2 <= x1 or y2 <= y1:
                    continue
                face_img = frame[y1:y2, x1:x2]
                if face_img.size == 0:
                    continue
                pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))

                pred_age, pred_gender = predict(pil_img, model, device)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                gender_text = "男" if pred_gender == 0 else "女"
                frame = cv2_add_chinese_text(frame, f'年龄: {pred_age:.2f}, 性别: {gender_text}', (x1, y1-10), 20, (255, 0, 0))

            cv2.imshow('Result(Press Q to leave)', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):  # 按'q'退出
                break

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        save_error_log(e)


if __name__ == "__main__":
    main()
    