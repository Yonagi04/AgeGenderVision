import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import keyboard
import traceback
import datetime
import time
import sys
import matplotlib
import signal
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
import matplotlib.pyplot as plt
import hashlib
import json
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm


LOG_FILE = 'error_log.log'
MODEL_DIR_FLAG = 'data/last_model_dir.txt'
STOP_FLAG_FILE = os.path.abspath("stop.flag")
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

# 创建唯一存储路径
def gen_model_dir(model_type, model_path, save_dir="models"):
    now = datetime.datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    unique_str = f"{model_path}-{model_type}-{timestamp}"
    unique_id = hashlib.md5(unique_str.encode()).hexdigest()[:6]
    if model_path.endswith('.pth'):
        model_path = model_path[:-4]
    folder_name = f"{date_str}-{model_path}-{model_type}-{unique_id}"
    folder_path = os.path.join(save_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path

# 存储模型
def save_model(model_type, model_path, model, loader, device, epochs, batch_size, img_size, val_age_loss, val_gender_loss, val_acc):
    try:
        model_dir = gen_model_dir(model_type, model_path)
        model_save_path = os.path.join(model_dir, model_path)
        torch.save(model.state_dict(), model_save_path)
        save_img(model, model_dir, loader, device)
        meta = {
            "model_name": model_path,
            "model_type": model_type,
            "created_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset": "UTKFace",
            "epochs": epochs,
            "batch_size": batch_size,
            "img_size": img_size,
            "eval_result": {
                "val_age_loss": float(val_age_loss),
                "val_gender_loss": float(val_gender_loss),
                "val_acc": float(val_acc),
                "age_scatter_image": "age_scatter.png",
                "gender_confusion_image": "gender_confusion.png"
            }
        }
        meta_path = os.path.join(model_dir, "meta.json")
        with open(meta_path, "w", encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"模型及元信息保存到 {model_dir}")
        with open(MODEL_DIR_FLAG, 'w', encoding='utf-8') as f:
            f.write(model_dir)
    except Exception as e:
        save_error_log(e)

def save_img(model, model_dir, loader, device):
    try:
        model.eval()
        y_true_age, y_pred_age = [], []
        y_true_gender, y_pred_gender = [], []
        with torch.no_grad():
            for imgs, ages, genders in loader:
                imgs = imgs.to(device)
                ages = ages.cpu().numpy()
                genders = genders.cpu().numpy()
                pred_age, pred_gender = model(imgs)
                pred_age = pred_age.cpu().numpy()
                pred_gender_label = pred_gender.argmax(dim=1).cpu().numpy()
                y_true_age.extend(ages.tolist())
                y_pred_age.extend(pred_age.tolist())
                y_true_gender.extend(genders.tolist())
                y_pred_gender.extend(pred_gender_label.tolist())

        plt.figure(figsize=(5, 5))
        plt.scatter(y_true_age, y_pred_age, alpha=0.5)
        plt.xlabel("真实年龄")
        plt.ylabel("预测年龄")
        plt.title("年龄回归散点图")
        plt.plot([min(y_true_age), max(y_true_age)], [min(y_true_age), max(y_true_age)], 'r--')
        age_fig_path = os.path.join(model_dir, "age_scatter.png")
        plt.savefig(age_fig_path)
        plt.close()

        cm = confusion_matrix(y_true_gender, y_pred_gender, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Male", "Female"])
        disp.plot(cmap=plt.cm.Blues)
        plt.title("性别混淆矩阵")
        gender_fig_path = os.path.join(model_dir, "gender_confusion.png")
        plt.savefig(gender_fig_path)
        plt.close()
    except Exception as e:
        save_error_log(e)

def dummy_handler(signum, frame):
    print("非命令行环境下，屏蔽Ctrl+C（KeyboardInterrupt），请通过UI停止训练。")

is_tty = sys.stdout.isatty()
if not is_tty:
    signal.signal(signal.SIGINT, dummy_handler)

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
        is_tty = sys.stdout.isatty()
        for epoch in range(args.epochs):
            model.train()
            total_loss = 0
            if is_tty:
                pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
            else:
                pbar = tqdm(
                    train_loader,
                    desc=f"Epoch {epoch+1}/{args.epochs}",
                    ncols=80,
                    file=sys.stdout,
                    mininterval=0,
                    miniters=1,
                    dynamic_ncols=True,
                    leave=True,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
                )
            for imgs, ages, genders in pbar:
                if is_tty and keyboard.is_pressed('q'):
                    print("\n检测到 Q 键，提前结束训练。")
                    val_age_loss, val_gender_loss, val_acc = evaluate(model, val_loader, device, age_criterion, gender_criterion)
                    save_model(args.model_type, args.model_path, model, val_loader, device, 
                               args.epochs, args.batch_size, args.img_size, val_age_loss, 
                               val_gender_loss, val_acc)
                    print('模型可能不完善，请特别注意。')
                    return
                if os.path.exists(STOP_FLAG_FILE):
                    print("\n检测到停止标志，提前结束训练。")
                    val_age_loss, val_gender_loss, val_acc = evaluate(model, val_loader, device, age_criterion, gender_criterion)
                    save_model(args.model_type, args.model_path, model, val_loader, device,
                               args.epochs, args.batch_size, args.img_size, val_age_loss,
                               val_gender_loss, val_acc)
                    print('模型可能不完善，请特别注意。')
                    os.remove(STOP_FLAG_FILE)
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
                if not is_tty:
                    sys.stdout.flush()
            avg_loss = total_loss / len(train_set)
            val_age_loss, val_gender_loss, val_acc = evaluate(model, val_loader, device, age_criterion, gender_criterion)
            print(f"Epoch {epoch+1}: Train Loss={avg_loss:.4f} | Val Age Loss={val_age_loss:.4f} | Val Gender Loss={val_gender_loss:.4f} | Val Gender Acc={val_acc:.4f}")
            time.sleep(1)

        save_model(args.model_type, args.model_path, model, val_loader, device, 
                               args.epochs, args.batch_size, args.img_size, val_age_loss, 
                               val_gender_loss, val_acc)
        print(f'Model saved to {args.model_path}')
    except Exception as e:
        save_error_log(e)

if __name__ == "__main__":
    main()