import os
import sys
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QTextEdit, QComboBox, QFileDialog
)
from utils.model_utils import refresh_model_list, get_model_type, get_model_dir
from threads.predict_thread import PThread

class PredictImagePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContentsMargins(20, 20, 20, 20)
        layout = QVBoxLayout(self)
        self.model_combo = QComboBox()
        self.refresh_models()
        self.img_path = QLineEdit()
        self.img_path.setPlaceholderText("请选择图片")
        btn_img = QPushButton("选择图片")
        btn_img.clicked.connect(self.select_image)
        h_img = QHBoxLayout()
        h_img.addWidget(self.img_path)
        h_img.addWidget(btn_img)
        layout.addWidget(QLabel("选择模型"))
        layout.addWidget(self.model_combo)
        layout.addWidget(QLabel("选择图片"))
        layout.addLayout(h_img)
        self.btn_predict = QPushButton("开始预测")
        layout.addWidget(self.btn_predict)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        layout.addWidget(self.result_text)
        layout.addStretch()
        self.btn_predict.clicked.connect(self.predict)
        self.predict_thread = None

    def refresh_models(self):
        self.model_combo.clear()
        models = refresh_model_list()
        self.model_combo.addItems(models)

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)")
        if path:
            self.img_path.setText(path)

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
        img_path = self.img_path.text().strip()
        if not img_path or not os.path.exists(img_path):
            self.result_text.append("请选择有效的图片")
            return
        env = os.environ.copy()
        env["IS_SUBPROCESS"] = "1"
        cmd = f"{sys.executable} photo_predict.py --model_path \"{model_path}\" --model_type \"{model_type}\" --img_path \"{img_path}\""
        self.result_text.append("正在预测，请稍候...")
        self.btn_predict.setEnabled(False)
        self.predict_thread = PThread(cmd, env, capture_output=True)
        def on_finish(result, error):
            self.btn_predict.setEnabled(True)
            if error:
                self.result_text.append(f"图片预测失败：{error}")
            else:
                self.result_text.append(f"预测结果：\n{result.strip()}")
        self.predict_thread.finished.connect(on_finish)
        self.predict_thread.start()