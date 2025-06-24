import os
import sys
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QTextEdit, QComboBox, QFileDialog
)
from utils.model_utils import refresh_model_list, get_model_type, get_model_dir
from threads.predict_thread import PThread

class PredictCameraPanel(QWidget):
    def __init__(self, parent=None, theme='light'):
        super().__init__(parent)
        self.theme = theme
        self.setContentsMargins(20, 20, 20, 20)
        layout = QVBoxLayout(self)
        self.model_combo = QComboBox()
        self.refresh_models()
        layout.addWidget(QLabel("选择模型"))
        layout.addWidget(self.model_combo)
        self.btn_predict = QPushButton("开始视频预测")
        layout.addWidget(self.btn_predict)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        layout.addWidget(self.result_text)
        layout.addStretch()
        self.btn_predict.clicked.connect(self.predict)
        self.predict_thread = None
        self.is_running = False

    def refresh_models(self):
        self.model_combo.clear()
        models = refresh_model_list()
        self.model_combo.addItems(models)

    def predict(self):
        self.result_text.clear()
        model_name = self.model_combo.currentText()
        if not model_name:
            self.result_text.append("请先训练模型")
            return
        model_type = get_model_type(model_name)
        fold_path = get_model_dir(model_name)
        if not fold_path or not os.path.exists(fold_path):
            self.result_text.append("请选择有效的模型")
            return
        model_path = os.path.join(fold_path, model_name)
        env = os.environ.copy()
        cmd = [
            sys.executable,
            "camera_predict.py",
            "--model_path", model_path,
            "--model_type", model_type
        ]
        self.result_text.append("正在摄像头采集预测，请稍候...")
        self.btn_predict.setEnabled(False)
        self.is_running = True
        self.predict_thread = PThread(cmd, env, capture_output=False)
        def on_finish(result, error):
            self.btn_predict.setEnabled(True)
            self.is_running = False
            if error:
                self.result_text.append(f"摄像头采集预测失败：{error}")
            else:
                self.result_text.append("摄像头采集预测已完成！")
        self.predict_thread.finished.connect(on_finish)
        self.predict_thread.start()