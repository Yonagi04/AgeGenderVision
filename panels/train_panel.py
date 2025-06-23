from PyQt5.QtWidgets import (
    QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QTextEdit, QComboBox, QFormLayout, QProgressBar
)
import os
import sys
import json
import datetime
from threads.train_thread import TrainThread

MODELS_INFO_FILE = 'data/models.json'
MODEL_DIR_FLAG = 'data/last_model_dir.txt'
STOP_FLAG_FILE = os.path.abspath("stop.flag")

class TrainPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContentsMargins(20, 20, 20, 20)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.batch_size = QLineEdit("64")
        self.epochs = QLineEdit("10")
        self.lr = QLineEdit("0.001")
        self.img_size = QLineEdit("224")
        self.data_dir = QLineEdit("data/UTKFace/cleaned")
        self.model_type = QComboBox()
        self.model_type.addItems(['resnet18', 'resnet34', 'resnet50'])
        self.model_path = QLineEdit("age_gender_multitask_resnet18.pth")
        form.addRow("Batch size", self.batch_size)
        form.addRow("Epochs", self.epochs)
        form.addRow("Learning rate", self.lr)
        form.addRow("Image size", self.img_size)
        form.addRow("数据集目录", self.data_dir)
        form.addRow("模型类型", self.model_type)
        form.addRow("模型保存路径", self.model_path)
        layout.addLayout(form)
        self.btn_train = QPushButton("开始训练")
        self.btn_stop = QPushButton("停止训练")
        self.btn_stop.setEnabled(False)
        btn_h = QHBoxLayout()
        btn_h.addWidget(self.btn_train)
        btn_h.addWidget(self.btn_stop)
        layout.addLayout(btn_h)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)
        self.tqdm_label = QLabel("训练进度：0/0")
        self.tqdm_bar = QProgressBar()
        self.tqdm_bar.setRange(0, 100)
        layout.addWidget(self.tqdm_label)
        layout.addWidget(self.tqdm_bar)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        layout.addStretch()
        self.btn_train.clicked.connect(self.start_train)
        self.btn_stop.clicked.connect(self.stop_train)
        self.train_thread = None
        self.is_running = False

    def start_train(self):
        try:
            self.log_text.clear()
            batch_size = int(self.batch_size.text())
            epochs = int(self.epochs.text())
            lr = float(self.lr.text())
            img_size = int(self.img_size.text())
            data_dir = self.data_dir.text().strip()
            model_type = self.model_type.currentText()
            model_path = self.model_path.text().strip()
            if batch_size <= 0 or epochs <= 0 or lr <= 0 or img_size <= 0:
                raise ValueError("Batch size、Epochs、Learning rate、Image size 必须为正数")
            if not os.path.isdir(data_dir):
                raise ValueError("数据集目录不存在")
            if not model_path.lower().endswith('.pth'):
                model_path += '.pth'
            if not model_path:
                raise ValueError("模型保存路径不能为空")
        except Exception as e:
            self.log_text.append(f"参数错误：{e}")
            return
        if os.path.exists(STOP_FLAG_FILE):
            os.remove(STOP_FLAG_FILE)
        if os.path.exists(MODEL_DIR_FLAG):
            os.remove(MODEL_DIR_FLAG)
        self.tqdm_label.setText("训练进度：0/0")
        self.tqdm_bar.setValue(0)
        cmd = [
            sys.executable,
            "train_age_gender_multitask.py",
            "--batch_size", batch_size,
            "--epochs", epochs,
            "--lr", lr,
            "--img_size", img_size,
            "--data_dir", data_dir,
            "--model_type", model_type,
            "--model_path", model_path
        ]
        env = os.environ.copy()
        self.log_text.clear()
        self.progress.setValue(0)
        self.btn_train.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.is_running = True
        self.train_thread = TrainThread(cmd, env)
        self.train_thread.log_signal.connect(self.log_text.append)
        self.train_thread.progress_signal.connect(self.progress.setValue)
        def update_tqdm(current, total):
            self.tqdm_label.setText(f"训练进度：{current}/{total}")
            if total > 0:
                percent = int(current / total * 100)
                self.tqdm_bar.setValue(percent)
            else:
                self.tqdm_bar.setValue(0)
        self.train_thread.tqdm_signal.connect(update_tqdm)
        def on_finish(error):
            self.btn_train.setEnabled(True)
            self.btn_stop.setEnabled(False)
            self.is_running = False
            if error:
                self.log_text.append(f"训练失败：{error}")
            else:
                self.log_text.append("训练完成！")
                last_model_dir = None
                if os.path.exists(MODEL_DIR_FLAG):
                    with open(MODEL_DIR_FLAG, 'r', encoding='utf-8') as f:
                        last_model_dir = f.read().strip()
                if not last_model_dir:
                    self.log_text.append("模型位置定位失败，但是模型已经保存成功。请检查 data/last_model_dir.txt 文件是否存在且内容正确，并手动把模型信息写入到 data/models.json")
                else:
                    if os.path.exists(os.path.join(last_model_dir, model_path)):
                        try:
                            if os.path.exists(MODELS_INFO_FILE):
                                with open(MODELS_INFO_FILE, 'r', encoding='utf-8') as f:
                                    info = json.load(f)
                            else:
                                info = {}
                            info[model_path] = {
                                "model_name": model_path,
                                "model_type": model_type,
                                "model_dir": last_model_dir,
                                "created_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            }
                            with open(MODELS_INFO_FILE, 'w', encoding='utf-8') as f:
                                json.dump(info, f, ensure_ascii=False, indent=2)
                        except Exception as e:
                            self.log_text.append(f"模型信息写入失败：{e}")
                    else:
                        self.log_text.append("模型保存失败！请排查原因。")
        self.train_thread.finished.connect(on_finish)
        self.train_thread.start()

    def stop_train(self):
        if self.train_thread:
            self.train_thread.stop()
            self.log_text.append("已请求停止训练...")