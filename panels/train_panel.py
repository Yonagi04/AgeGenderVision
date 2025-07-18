from PyQt5.QtWidgets import (
    QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QTextEdit, QComboBox, QFormLayout, QProgressBar, QGridLayout,
    QSpacerItem, QSizePolicy, QMessageBox
)
import os
import sys
import json
import datetime
from threads.train_thread import TrainThread
from widgets.switch import Switch
from widgets.message_box import MessageBox

MODELS_INFO_FILE = 'data/models.json'
MODEL_DIR_FLAG = 'data/last_model_dir.txt'
STOP_FLAG_FILE = os.path.abspath("stop.flag")

class TrainPanel(QWidget):
    def __init__(self, parent=None, theme='light'):
        super().__init__(parent)
        self.theme = theme
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
        form.addRow("模型名称", self.model_path)
        form.addRow("模型类型", self.model_type)
        form.addRow("数据集目录", self.data_dir)
        form.addRow("Batch size", self.batch_size)
        form.addRow("Epochs", self.epochs)
        form.addRow("Learning rate", self.lr)
        form.addRow("Image size", self.img_size)
        layout.addLayout(form)
        
        self.switch_early = Switch()
        grid_early = QGridLayout()
        grid_early.addWidget(QLabel("Early Stopping"), 0, 0)
        grid_early.addItem(QSpacerItem(10, 0, QSizePolicy.Fixed, QSizePolicy.Fixed), 0, 1)  # 只加一点横向间隔
        grid_early.addWidget(self.switch_early, 0, 2)

        layout.addLayout(grid_early)

        self.switch_lr = Switch()
        grid_lr = QGridLayout()
        grid_lr.addWidget(QLabel("自适应Learning rate调整"), 0, 0)
        grid_lr.addItem(QSpacerItem(10, 0, QSizePolicy.Fixed, QSizePolicy.Fixed), 0, 1)  # 只加一点横向间隔
        grid_lr.addWidget(self.switch_lr, 0, 2)

        layout.addLayout(grid_lr)

        self.btn_train = QPushButton("开始训练")
        self.btn_stop = QPushButton("停止训练")
        self.btn_stop.setEnabled(False)
        btn_h = QHBoxLayout()
        btn_h.addWidget(self.btn_train)
        btn_h.addWidget(self.btn_stop)
        layout.addLayout(btn_h)
        self.tqdm_label = QLabel("训练进度：0/0")
        self.tqdm_bar = QProgressBar()
        self.tqdm_bar.setRange(0, 100)
        layout.addWidget(self.tqdm_label)
        layout.addWidget(self.tqdm_bar)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        self.btn_show_chart = QPushButton("查看训练数据图表")
        layout.addWidget(self.btn_show_chart)
        layout.addStretch()
        self.btn_train.clicked.connect(self.start_train)
        self.btn_stop.clicked.connect(self.stop_train)
        self.btn_show_chart.clicked.connect(self.show_chart_panel)
        self.train_thread = None
        self.is_running = False
        self.metric_history = {
            "train_loss": [],
            "val_age_loss": [],
            "val_gender_loss": [],
            "val_gender_acc": []
        }

    def get_current_theme(self):
        theme = "light"
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, "current_theme"):
                theme = parent.current_theme
                break
            parent = parent.parent()
        return theme

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
            is_lr = self.switch_lr.isChecked()
            is_early = self.switch_early.isChecked()
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
        if os.path.exists(MODELS_INFO_FILE):
            try:
                with open(MODELS_INFO_FILE, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                if info:
                    for name, meta in info.items():
                        if name == model_path:
                            self.log_text.append(f"模型 {model_path} 已存在，可能会覆盖原有模型。")
                            return
            except Exception as e:
                self.log_text.append(f"加载模型信息失败: {e}")

        if os.path.exists(STOP_FLAG_FILE):
            os.remove(STOP_FLAG_FILE)
        if os.path.exists(MODEL_DIR_FLAG):
            os.remove(MODEL_DIR_FLAG)
        self.tqdm_label.setText("训练进度：0/0")
        self.tqdm_bar.setValue(0)
        cmd = [
            sys.executable,
            "train_age_gender_multitask.py",
            "--batch_size", str(batch_size),
            "--epochs", str(epochs),
            "--lr", str(lr),
            "--img_size", str(img_size),
            "--data_dir", data_dir,
            "--model_type", model_type,
            "--model_path", model_path,
            "--is_early_stopping", str(is_early),
            "--reduce_lr", str(is_lr)
        ]
        env = os.environ.copy()
        self.log_text.clear()
        self.clear_plot()
        # self.progress.setValue(0)
        self.btn_train.setEnabled(False)
        self.batch_size.setEnabled(False)
        self.model_path.setEnabled(False)
        self.model_type.setEnabled(False)
        self.data_dir.setEnabled(False)
        self.lr.setEnabled(False)
        self.img_size.setEnabled(False)
        self.switch_early.setCheckable(False)
        self.switch_lr.setCheckable(False)
        self.btn_stop.setEnabled(True)
        self.is_running = True
        self.train_thread = TrainThread(cmd, env)
        self.train_thread.log_signal.connect(self.log_text.append)
        self.train_thread.metrics_signal.connect(self.update_metrics_plot)
        # self.train_thread.progress_signal.connect(self.progress.setValue)
        def update_tqdm(current, total):
            self.tqdm_label.setText(f"训练进度：{current}/{total}")
            if total > 0:
                percent = int(current / total * 100)
                self.tqdm_bar.setValue(percent)
            else:
                self.tqdm_bar.setValue(0)
        self.train_thread.tqdm_signal.connect(update_tqdm)
        def on_finish(error):
            self.theme = self.get_current_theme()
            self.btn_train.setEnabled(True)
            self.batch_size.setEnabled(True)
            self.model_path.setEnabled(True)
            self.model_type.setEnabled(True)
            self.data_dir.setEnabled(True)
            self.lr.setEnabled(True)
            self.img_size.setEnabled(True)
            self.switch_early.setCheckable(True)
            self.switch_lr.setCheckable(True)
            self.btn_stop.setEnabled(False)
            self.is_running = False
            if error:
                self.log_text.append(f"训练失败：{error}")
                msg_box = MessageBox(
                    parent=self,
                    title='错误',
                    text=f'训练失败: {error}',
                    theme=self.theme,
                    icon = QMessageBox.Critical
                )
                msg_box.exec_()
            else:
                self.log_text.append("训练完成！")
                msg_box = MessageBox(
                    parent=self,
                    text='训练完成',
                    theme=self.theme
                )
                msg_box.show()
                last_model_dir = None
                if os.path.exists(MODEL_DIR_FLAG):
                    with open(MODEL_DIR_FLAG, 'r', encoding='utf-8') as f:
                        last_model_dir = f.read().strip()
                if not last_model_dir:
                    self.log_text.append("模型位置定位失败，但是模型已经保存成功。请检查 data/last_model_dir.txt 文件是否存在且内容正确，并手动把模型信息写入到 data/models.json")
                else:
                    if os.path.exists(os.path.join(last_model_dir, model_path)):
                        try:
                            meta_path = os.path.join(last_model_dir, 'meta.json')
                            if os.path.exists(meta_path):
                                with open(meta_path, 'r', encoding='utf-8') as f:
                                    meta = json.load(f)
                            tags = meta['tags'] if meta else []
                            eval_result = meta['eval_result'] if meta else {
                                "val_age_loss": "",
                                "val_gender_loss": "",
                                "val_acc": "",
                                "age_scatter_image": "",
                                "gender_confusion_image": ""
                            }
                            if os.path.exists(MODELS_INFO_FILE):
                                with open(MODELS_INFO_FILE, 'r', encoding='utf-8') as f:
                                    info = json.load(f)
                            else:
                                info = {}
                            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                            info[model_path] = {
                                "model_name": model_path,
                                "model_type": model_type,
                                "model_dir": last_model_dir,
                                "epochs": epochs,
                                "batch_size": batch_size,
                                "img_size": img_size,
                                "created_time": timestamp,
                                "update_time": timestamp,
                                "description": "Created by AgeGenderVision",
                                "tags": tags,
                                "eval_result": eval_result
                            }
                            with open(MODELS_INFO_FILE, 'w', encoding='utf-8') as f:
                                json.dump(info, f, ensure_ascii=False, indent=2)
                            if meta:
                                del meta['tags']
                            with open(meta_path, 'w', encoding='utf-8') as f:
                                json.dump(meta, f, ensure_ascii=False, indent=2)
                            os.remove(MODEL_DIR_FLAG)
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

    def update_metrics_plot(self, metrics):
        for k in self.metric_history:
            self.metric_history[k].append(metrics[k])
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, 'train_chart_panel'):
                parent.train_chart_panel.update_metrics_plot(self.metric_history)
                break
            parent = parent.parent()
        
    def show_chart_panel(self):
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, 'show_train_chart_panel'):
                parent.show_train_chart_panel(self.metric_history)
                break
            parent = parent.parent()

    def clear_plot(self):
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, 'train_chart_panel'):
                parent.train_chart_panel.clear_plot()
                break
            parent = parent.parent()
 
