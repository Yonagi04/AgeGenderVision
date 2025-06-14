import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image, ImageFont, ImageDraw
import numpy as np

def cv2_add_chinese_text(img, text, position, font_size=20, color=(255, 255, 255)):
    """在图像上添加中文文本"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype("/data/simhei.ttf", font_size)  # 使用黑体字体
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# 1. 定义多任务模型
class MultiTaskResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.age_head = nn.Linear(in_features, 1)         # 年龄回归
        self.gender_head = nn.Linear(in_features, 2)      # 性别分类

    def forward(self, x):
        feat = self.backbone(x)
        age = self.age_head(feat)
        gender = self.gender_head(feat)
        return age, gender

# 2. 定义推理函数
def predict(img, model, device):
    model.eval()  # 设置模型为评估模式
    img = transform(img).unsqueeze(0).to(device)  # 预处理并转移到设备
    with torch.no_grad():  # 不计算梯度
        pred_age, pred_gender = model(img)  # 进行预测
        pred_age = pred_age.item()  # 获取年龄预测值
        pred_gender = pred_gender.softmax(1).argmax(1).item()  # 获取性别预测值
    return pred_age, pred_gender

# 3. 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiTaskResNet18().to(device)
model.load_state_dict(torch.load('age_gender_multitask_resnet18.pth'))  # 加载模型权重

# 4. 定义数据预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# 5. 打开摄像头
cap = cv2.VideoCapture(0)

# 6. 加载人脸检测模型（使用OpenCV的Haar特征分类器）
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()  # 读取摄像头画面
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 转为灰度图
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)  # 检测人脸

    for (x, y, w, h) in faces:
        # 提取人脸区域
        face_img = frame[y:y+h, x:x+w]
        pil_img = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))  # 转为PIL格式

        # 进行预测
        pred_age, pred_gender = predict(pil_img, model, device)

        # 绘制人脸框和预测结果
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)  # 绘制人脸框
        gender_text = "男" if pred_gender == 0 else "女"
        frame = cv2_add_chinese_text(frame, f'年龄: {pred_age:.2f}, 性别: {gender_text}', (x, y-10), 20, (255, 0, 0))

    cv2.imshow('Video', frame)  # 显示视频画面

    if cv2.waitKey(1) & 0xFF == ord('q'):  # 按'q'退出
        break

# 释放摄像头和窗口
cap.release()
cv2.destroyAllWindows()
