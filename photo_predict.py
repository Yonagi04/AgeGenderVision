import os
import torch
import argparse
import datetime
import sys
import traceback
import tkinter as tk
import torch.nn as nn
import cv2
import numpy as np
from tkinter import filedialog
from torchvision import transforms, models
from PIL import Image, ImageFont, ImageDraw

parser = argparse.ArgumentParser()
parser.add_argument('--model_path', type=str, default='age_gender_multitask_resnet18.pth')
parser.add_argument('--model_type', type=str, default='resnet18', help='模型类型', choices=['resnet18', 'resnet34', 'resnet50'])
parser.add_argument('--img_path', type=str, default='', help='图片路径')
args = parser.parse_args()
model_path = args.model_path
model_type = args.model_type
img_path = args.img_path

LOG_FILE = 'error_log.log'
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
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.age_head = nn.Linear(num_ftrs, 1)
        self.gender_head = nn.Linear(num_ftrs, 2)

    def forward(self, x):
        feat = self.backbone(x)
        age = self.age_head(feat).squeeze(1)
        gender = self.gender_head(feat)
        return age, gender

def cv2_add_chinese_text(img, text, position, font_size=24, color=(255, 0, 0)):
    # OpenCV BGR to PIL RGB
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 使用Windows自带字体
    font_path = "/data/simhei.ttf" if os.name == 'nt' else "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
    font = ImageFont.truetype(font_path, font_size)
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    # PIL RGB to OpenCV BGR
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def main():
    try:
        global img_path
        if not img_path:
            root = tk.Tk()
            root.withdraw()
            img_path = filedialog.askopenfilename(
                title="选择图片文件",
                filetypes=[("图片文件", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")]
            )
            root.destroy()
        if not img_path:
            print("未选择图片文件，已取消。")
            input("按任意键退出。")
            return
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = MultiTaskResNet(model_type=model_type).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        model.eval()

        # 读取图片
        img_cv = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img_cv is None:
            print("无法读取图片文件。")
            input('按任意键退出...')
            exit()

        # 人脸检测
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        results = []
        if len(faces) == 0:
            # 未检测到人脸，直接对整张图预测
            img_pil = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            input_tensor = transform(img_pil).unsqueeze(0).to(device)
            with torch.no_grad():
                pred_age, pred_gender = model(input_tensor)
                age = pred_age.item()
                gender_idx = pred_gender.argmax(dim=1).item()
                gender_text = '男' if gender_idx == 1 else '女'
                results.append({'box': None, 'age': age, 'gender': gender_text})
        else:
            for (x, y, w, h) in faces:
                face_img = img_cv[y:y+h, x:x+w]
                img_pil = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
                input_tensor = transform(img_pil).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred_age, pred_gender = model(input_tensor)
                    age = pred_age.item()
                    gender_idx = pred_gender.argmax(dim=1).item()
                    gender_text = '男' if gender_idx == 1 else '女'
                    results.append({'box': (x, y, w, h), 'age': age, 'gender': gender_text})

        # 控制台输出
        for i, res in enumerate(results):
            if res['box']:
                print(f"人脸{i+1}: 预测年龄: {res['age']:.2f}，预测性别: {res['gender']}")
            else:
                print(f"整图: 预测年龄: {res['age']:.2f}，预测性别: {res['gender']}")

        max_width, max_height = 1024, 768
        h, w = img_cv.shape[:2]
        scale = min(max_width / w, max_height / h, 1.0)
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            img_cv = cv2.resize(img_cv, (new_w, new_h), interpolation=cv2.INTER_AREA)

        base_font_size = 24
        font_size = max(int(base_font_size * scale * 2), 16)

        for res in results:
            if res['box']:
                x, y, w_box, h_box = res['box']
                x = int(x * scale)
                y = int(y * scale)
                w_box = int(w_box * scale)
                h_box = int(h_box * scale)
                cv2.rectangle(img_cv, (x, y), (x + w_box, y + h_box), (0, 255, 0), 2)
                img_cv = cv2_add_chinese_text(
                    img_cv,
                    f"年龄:{res['age']:.1f} 性别:{res['gender']}",
                    (x, y - 30 if y - 30 > 0 else y + 5),
                    font_size = font_size,
                    color = (255, 0, 0)
                )
            else:
                img_cv = cv2_add_chinese_text(
                    img_cv,
                    f"年龄:{res['age']:.1f} 性别:{res['gender']}",
                    (10, 30),
                    font_size=font_size,
                    color=(255, 0, 0)
                )

        cv2.namedWindow('Result', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Result', img_cv)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception as e:
        save_error_log(e)


if __name__ == "__main__":
    main()
    if sys.stdin.isatty() and not os.environ.get("IS_SUBPROCESS"):
        input("按任意键退出...")