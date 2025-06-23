import os
import sys
import datetime
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QTextEdit, QFileDialog, QCheckBox
)
from utils.model_utils import refresh_model_list, get_model_type, get_model_dir
from threads.predict_thread import PThread

RESULT_LOG_FILE = 'result_log.log'

class PredictMultiImagePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContentsMargins(20, 20, 20, 20)
        layout = QVBoxLayout(self)
        self.model_checks = []
        self.models = []
        self.refresh_models()
        self.img_path = QLineEdit()
        self.img_path.setPlaceholderText("请选择图片")
        self.img_path.setReadOnly(True)
        btn_img = QPushButton("选择图片")
        btn_img.clicked.connect(self.select_image)
        h_img = QHBoxLayout()
        h_img.addWidget(self.img_path)
        h_img.addWidget(btn_img)
        layout.addWidget(QLabel("选择模型（可多选）"))
        self.models_layout = QVBoxLayout()
        for cb in self.model_checks:
            self.models_layout.addWidget(cb)
        layout.addLayout(self.models_layout)
        layout.addLayout(h_img)
        self.btn_predict = QPushButton("开始多模型预测")
        layout.addWidget(self.btn_predict)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        layout.addWidget(self.result_text)
        layout.addStretch()
        self.btn_predict.clicked.connect(self.predict)
        self.predict_threads = []
        self.is_running = False
    
    def refresh_models(self):
        self.models = refresh_model_list()
        self.model_checks = []
        if hasattr(self, "models_layout"):
            while self.models_layout.count():
                item = self.models_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
        for m in self.models:
            cb = QCheckBox(m)
            self.model_checks.append(cb)
            if hasattr(self, "models_layout"):
                self.models_layout.addWidget(cb)

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)")
        if path:
            self.img_path.setText(path)

    def predict(self):
        self.result_text.clear()
        selected = [cb.text() for cb in self.model_checks if cb.isChecked()]
        if not selected:
            self.result_text.append("请至少选择一个模型进行比较")
            return
        img_path = self.img_path.text().strip()
        if not img_path or not os.path.exists(img_path):
            self.result_text.append("请选择有效的图片")
            return
        model_types = [get_model_type(m) for m in selected]
        model_path = []
        for m in selected:
            fold_path = get_model_dir(m)
            if not fold_path or not os.path.exists(fold_path):
                self.result_text.append("请选择有效的模型")
                return
            path = os.path.join(fold_path, m)
            model_path.append(path)
        self.result_text.append("正在进行多模型预测，请稍候...")
        self.btn_predict.setEnabled(False)
        self.is_running = True
        self.multi_results = []
        self.multi_predict_threads = []
        self.finished_count = 0
        def on_finish(idx):
            def inner(result, error):
                self.finished_count += 1
                if error:
                    self.multi_results.append(f"模型: {selected[idx]} ({model_types[idx]})\n预测失败：{error}\n{'-'*30}")
                else:
                    self.multi_results.append(f"模型: {selected[idx]} ({model_types[idx]})\n{result.strip()}\n{'-'*30}")
                    # 写入日志
                    with open(RESULT_LOG_FILE, 'a', encoding='utf-8') as f:
                        f.write(
                            f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n"
                            f"预测模型: {selected[idx]}\n"
                            f"模型类型: {model_types[idx]}\n"
                            f"预测图片: {img_path}\n"
                            f"预测结果:\n{result}\n"
                            f"{'-'*40}\n"
                        )
                if self.finished_count == len(selected):
                    self.result_text.append("\n".join(self.multi_results))
                    self.btn_predict.setEnabled(True)
                    self.is_running = False
            return inner
        for idx, m in enumerate(model_path):
            env = os.environ.copy()
            env["IS_SUBPROCESS"] = "1"
            cmd = [
                sys.executable,
                "photo_predict.py",
                "--model_path", m,
                "--model_type", model_types[idx],
                "--img_path", img_path
            ]
            thread = PThread(cmd, env, capture_output=True)
            thread.finished.connect(on_finish(idx))
            self.multi_predict_threads.append(thread)
            thread.start()