import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import keyboard
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

# 数据集定义
class UTKFaceMultiTaskDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.img_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        # 文件名格式: age_gender_race_date.jpg
        # 例如: 23_1_0_20170116174525125.jpg
    def __len__(self):
        return len(self.img_files)
    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_path = os.path.join(self.data_dir, img_name)
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        # 解析标签
        try:
            age, gender, *_ = img_name.split('_')
            age = float(age)
            gender = int(gender)
        except Exception:
            age = 0.0
            gender = 0
        return img, torch.tensor(age, dtype=torch.float32), torch.tensor(gender, dtype=torch.long)

class MultiTaskResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet18(weights=None)
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--data_dir', type=str, default='data/UTKFace/cleaned')
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

    model = MultiTaskResNet18().to(device)
    age_criterion = nn.MSELoss()
    gender_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    print("训练开始，按 Q 键随时退出。")
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

    torch.save(model.state_dict(), args.model_path)
    print(f'Model saved to {args.model_path}')

    # 示例：预测一张图片的年龄和性别
    example_imgs = [f for f in os.listdir(args.data_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if example_imgs:
        example_path = os.path.join(args.data_dir, example_imgs[0])
        pred_age, pred_gender = predict(example_path, model, device, transform)
        print(f'Example image: {example_imgs[0]}, Predicted age: {pred_age:.2f}, Predicted gender: {pred_gender} (0=男, 1=女)')

if __name__ == "__main__":
    main()