import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

# 数据集定义
class UTKFaceMultiTaskDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.imgs = []
        self.ages = []
        self.genders = []
        for f in os.listdir(root_dir):
            if f.lower().endswith('.jpg'):
                try:
                    age = int(f.split('_')[0])
                    gender = int(f.split('_')[1])
                    if gender in [0, 1]:
                        self.imgs.append(f)
                        self.ages.append(age)
                        self.genders.append(gender)
                except Exception as e:
                    continue  # 跳过解析异常的文件

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        age = self.ages[idx]
        gender = self.genders[idx]
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor([age], dtype=torch.float32), torch.tensor(gender, dtype=torch.long)

# 训练参数
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
IMG_SIZE = 224
DATA_DIR = 'data/UTKFace/cleaned'
MODEL_PATH = 'age_gender_multitask_resnet18.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])
dataset = UTKFaceMultiTaskDataset(DATA_DIR, transform)

val_ratio = 0.1
val_size = int(len(dataset) * val_ratio)
train_size = len(dataset) - val_size
train_set, val_set = random_split(dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

class MultiTaskResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.age_head = nn.Linear(in_features, 1)         # 回归
        self.gender_head = nn.Linear(in_features, 2)      # 分类
    def forward(self, x):
        feat = self.backbone(x)
        age = self.age_head(feat)
        gender = self.gender_head(feat)
        return age, gender

model = MultiTaskResNet18().to(device)

age_criterion = nn.MSELoss()
gender_criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

def evaluate(loader):
    model.eval()
    total_age_loss = 0
    total_gender_loss = 0
    correct_gender = 0
    with torch.no_grad():
        for imgs, ages, genders in loader:
            imgs, ages, genders = imgs.to(device), ages.to(device), genders.to(device)
            pred_ages, pred_genders = model(imgs)
            age_loss = age_criterion(pred_ages, ages)
            gender_loss = gender_criterion(pred_genders, genders)
            total_age_loss += age_loss.item() * imgs.size(0)
            total_gender_loss += gender_loss.item() * imgs.size(0)
            pred_gender_label = pred_genders.argmax(1)
            correct_gender += (pred_gender_label == genders).sum().item()
    avg_age_loss = total_age_loss / len(loader.dataset)
    avg_gender_loss = total_gender_loss / len(loader.dataset)
    gender_acc = correct_gender / len(loader.dataset)
    return avg_age_loss, avg_gender_loss, gender_acc

for epoch in range(EPOCHS):
    model.train()
    running_age_loss = 0
    running_gender_loss = 0
    correct_gender = 0
    for imgs, ages, genders in tqdm(train_loader, desc=f'Epoch {epoch+1}/{EPOCHS}'):
        imgs, ages, genders = imgs.to(device), ages.to(device), genders.to(device)
        optimizer.zero_grad()
        pred_ages, pred_genders = model(imgs)
        age_loss = age_criterion(pred_ages, ages)
        gender_loss = gender_criterion(pred_genders, genders)
        loss = age_loss + gender_loss
        loss.backward()
        optimizer.step()
        running_age_loss += age_loss.item() * imgs.size(0)
        running_gender_loss += gender_loss.item() * imgs.size(0)
        pred_gender_label = pred_genders.argmax(1)
        correct_gender += (pred_gender_label == genders).sum().item()
    train_age_loss = running_age_loss / len(train_loader.dataset)
    train_gender_loss = running_gender_loss / len(train_loader.dataset)
    train_gender_acc = correct_gender / len(train_loader.dataset)
    val_age_loss, val_gender_loss, val_gender_acc = evaluate(val_loader)
    print(f'Epoch {epoch+1}: Train AgeLoss={train_age_loss:.4f}, Train GenderLoss={train_gender_loss:.4f}, Train GenderAcc={train_gender_acc:.4f}')
    print(f'            Val AgeLoss={val_age_loss:.4f}, Val GenderLoss={val_gender_loss:.4f}, Val GenderAcc={val_gender_acc:.4f}')


torch.save(model.state_dict(), MODEL_PATH)
print(f'Model saved to {MODEL_PATH}')

def predict(img_path):
    model.eval()
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_age, pred_gender = model(img)
        pred_age = pred_age.item()
        pred_gender = pred_gender.softmax(1).argmax(1).item()
    return pred_age, pred_gender

# 示例：预测一张图片的年龄和性别
example_img = os.listdir(DATA_DIR)[0]
example_path = os.path.join(DATA_DIR, example_img)
pred_age, pred_gender = predict(example_path)
print(f'Example image: {example_img}, Predicted age: {pred_age:.2f}, Predicted gender: {pred_gender} (0=男, 1=女)') 