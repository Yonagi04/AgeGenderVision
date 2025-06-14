import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageFont, ImageDraw
import numpy as np

def cv2_add_chinese_text(img, text, position, font_size=20, color=(255, 255, 255)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype("/data/simhei.ttf", font_size)
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

class MultiTaskResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.age_head = nn.Linear(in_features, 1)
        self.gender_head = nn.Linear(in_features, 2)

    def forward(self, x):
        feat = self.backbone(x)
        age = self.age_head(feat)
        gender = self.gender_head(feat)
        return age, gender

def predict(img, model, device):
    model.eval()
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_age, pred_gender = model(img)
        pred_age = pred_age.item()
        pred_gender = pred_gender.softmax(1).argmax(1).item()
    return pred_age, pred_gender

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiTaskResNet18().to(device)
model.load_state_dict(torch.load('age_gender_multitask_resnet18.pth'))

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]
        pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))

        pred_age, pred_gender = predict(pil_img, model, device)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        gender_text = "男" if pred_gender == 0 else "女"
        frame = cv2_add_chinese_text(frame, f'年龄: {pred_age:.2f}, 性别: {gender_text}', (x, y-10), 20, (255, 0, 0))

    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按'q'退出
        break

cap.release()
cv2.destroyAllWindows()
