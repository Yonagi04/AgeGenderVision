import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import keyboard
import traceback
import datetime
import time
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

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

# 数据集定义
class UTKFaceMultiTaskDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.imgs = []
        self.ages = []
        self.genders = []
        img_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')
        for f in os.listdir(data_dir):
            if f.lower().endswith(img_exts):
                try:
                    age = int(f.split('_')[0])
                    gender = int(f.split('_')[1])
                    if gender in [0, 1]:
                        self.imgs.append(f)
                        self.ages.append(age)
                        self.genders.append(gender)
                except Exception as e:
                    print(f"跳过异常文件: {f}，原因: {e}")
                    continue  # 跳过解析异常的文件
        # 文件名格式: age_gender_race_date.
        # 例如: 23_1_0_20170116174525125.jpg
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.data_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        age = self.ages[idx]
        gender = self.genders[idx]
        return img, torch.tensor(age, dtype=torch.float32), torch.tensor(gender, dtype=torch.long)

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

def evaluate(model, loader, device, age_criterion, gender_criterion):
    model.eval()
    total_age_loss, total_gender_loss, total, correct = 0, 0, 0, 0
    with torch.no_grad():
        for imgs, ages, genders in loader:
            imgs, ages, genders = imgs.to(device), ages.to(device), genders.to(device)
            pred_age, pred_gender = model(imgs)
            age_loss = age_criterion(pred_age, ages)
            gender_loss = gender_criterion(pred_gender, genders)
            total_age_loss += age_loss.item() * imgs.size(0)
            total_gender_loss += gender_loss.item() * imgs.size(0)
            pred_gender_label = pred_gender.argmax(dim=1)
            correct += (pred_gender_label == genders).sum().item()
            total += imgs.size(0)
    avg_age_loss = total_age_loss / total
    avg_gender_loss = total_gender_loss / total
    acc = correct / total
    return avg_age_loss, avg_gender_loss, acc

def predict(img_path, model, device, transform):
    model.eval()
    img = Image.open(img_path).convert('RGB')
    img = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred_age, pred_gender = model(img)
        age = pred_age.item()
        gender_idx = pred_gender.argmax(dim=1).item()
    return age, gender_idx

def main():
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--epochs', type=int, default=10)
        parser.add_argument('--lr', type=float, default=1e-3)
        parser.add_argument('--img_size', type=int, default=224)
        parser.add_argument('--data_dir', type=str, default='data/UTKFace/cleaned')
        parser.add_argument('--model_type', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50'], help='选择模型类型')
        parser.add_argument('--model_path', type=str, default='age_gender_multitask_resnet18.pth')
        args = parser.parse_args()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('使用设备:', device)

        transform = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        dataset = UTKFaceMultiTaskDataset(args.data_dir, transform)

        if len(dataset) == 0:
            print(f"数据集目录 {args.data_dir} 为空或不存在图片。")
            return

        val_ratio = 0.1
        val_size = int(len(dataset) * val_ratio)
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(
            dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)

        model = MultiTaskResNet(model_type=args.model_type).to(device)
        age_criterion = nn.MSELoss()
        gender_criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=args.lr)

        print("训练开始，在一轮训练结束时可按 Q 键退出训练。")
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            for imgs, ages, genders in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", ncols=80):
                if keyboard.is_pressed('q'):
                    print("\n检测到 Q 键，提前结束训练。")
                    torch.save(model.state_dict(), args.model_path)
                    print(f'模型已保存到 {args.model_path}，模型可能不完善，请特别注意。')
                    return
                imgs, ages, genders = imgs.to(device), ages.to(device), genders.to(device)
                optimizer.zero_grad()
                pred_age, pred_gender = model(imgs)
                age_loss = age_criterion(pred_age, ages)
                gender_loss = gender_criterion(pred_gender, genders)
                loss = age_loss + gender_loss
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * imgs.size(0)
            avg_loss = total_loss / len(train_set)
            val_age_loss, val_gender_loss, val_acc = evaluate(model, val_loader, device, age_criterion, gender_criterion)
            print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f} | Val Age Loss={val_age_loss:.4f} | Val Gender Loss={val_gender_loss:.4f} | Val Gender Acc={val_acc:.4f}")
            time.sleep(1)

        torch.save(model.state_dict(), args.model_path)
        print(f'Model saved to {args.model_path}')
    except Exception as e:
        save_error_log(e)

if __name__ == "__main__":
    main()