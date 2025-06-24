import os
import sys
import cv2
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QTextEdit, QComboBox, QFileDialog
)
from PyQt5.QtCore import QTimer, Qt, QSize
from PyQt5.QtGui import QImage, QPixmap, QIcon
from utils.model_utils import refresh_model_list, get_model_type, get_model_dir
from threads.predict_thread import PThread

class PredictVideoPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContentsMargins(20, 20, 20, 20)
        layout = QVBoxLayout(self)
        self.model_combo = QComboBox()
        self.refresh_models()
        self.video_path = QLineEdit()
        self.video_path.setPlaceholderText("请选择视频")
        self.video_path.setReadOnly(True)
        self.btn_video = QPushButton("选择视频")
        self.btn_video.clicked.connect(self.select_video)
        self.output_path = QLineEdit()
        self.output_path.setPlaceholderText("导出文件位置")
        self.output_path.setReadOnly(True)
        self.btn_output = QPushButton("保存为")
        self.btn_output.clicked.connect(self.select_output)
        h_video = QHBoxLayout()
        h_video.addWidget(self.video_path)
        h_video.addWidget(self.btn_video)
        h_video.addWidget(self.output_path)
        h_video.addWidget(self.btn_output)
        layout.addWidget(QLabel("选择模型"))
        layout.addWidget(self.model_combo)
        layout.addLayout(h_video)
        self.btn_predict = QPushButton("开始预测")
        self.btn_predict.clicked.connect(self.predict)
        layout.addWidget(self.btn_predict)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        layout.addWidget(self.result_text)
        self.video_preview = QLabel("处理后的视频将在此处播放")
        self.video_preview.setFixedHeight(500)
        self.video_preview.setStyleSheet("background-color: #000; color: #ccc; font-size: 16px;")
        self.video_preview.setAlignment(Qt.AlignCenter)
        self.timer = QTimer()
        self.timer.timeout.connect(self.play_frame)
        self.cap = None
        control_layout = QHBoxLayout()
        self.btn_pause = QPushButton()
        self.btn_pause.setIcon(QIcon("assets/svg/play_light.svg"))
        self.btn_pause.setIconSize(QSize(32, 32))
        self.btn_pause.setFixedSize(40, 40)
        self.btn_pause.setStyleSheet("border:none; background:transparent;")
        self.btn_forward = QPushButton()
        self.btn_forward.setIcon(QIcon("assets/svg/forward_light.svg"))
        self.btn_forward.setIconSize(QSize(32, 32))
        self.btn_forward.setFixedSize(40, 40)
        self.btn_forward.setStyleSheet("border:none; background:transparent;")
        self.btn_backward = QPushButton()
        self.btn_backward.setIcon(QIcon("assets/svg/backward_light.svg"))
        self.btn_backward.setIconSize(QSize(32, 32))
        self.btn_backward.setFixedSize(40, 40)
        self.btn_backward.setStyleSheet("border:none; background:transparent;")
        self.btn_pause.setEnabled(False)
        self.btn_backward.setEnabled(False)
        self.btn_forward.setEnabled(False)
        self.btn_pause.clicked.connect(self.toggle_pause)
        self.btn_forward.clicked.connect(self.seek_forward)
        self.btn_backward.clicked.connect(self.seek_backward)
        control_layout.addWidget(self.btn_backward)
        control_layout.addWidget(self.btn_pause)
        control_layout.addWidget(self.btn_forward)
        layout.addWidget(self.video_preview)
        layout.addLayout(control_layout)
        layout.addStretch()
        self.predict_thread = None
        self.is_paused = True
        self.is_running = False

    def refresh_models(self):
        self.model_combo.clear()
        models = refresh_model_list()
        self.model_combo.addItems(models)

    def select_video(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Videos (*.mp4 *.avi *.mov *.mkv *.flv *.wmv *.webm)")
        if path:
            self.video_path.setText(path)

    def select_output(self):
        path, _ = QFileDialog.getSaveFileName(self, "保存预测视频为", filter="Videos (*.mp4 *.avi *.mov)")
        if path:
            if not path.lower().endswith(('.mp4', '.avi', '.mov')):
                path += '.mp4'
            self.output_path.setText(path)

    def toggle_pause(self):
        if not self.cap:
            return
        self.is_paused = not self.is_paused
        theme = "light"
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, "currrent_theme"):
                theme = parent.current_theme
                break
            parent = parent.parent()
        if self.is_paused:
            if theme == "dark":
                self.btn_pause.setIcon(QIcon("assets/svg/play_dark.svg"))
            else:
                self.btn_pause.setIcon(QIcon("assets/svg/play_light.svg"))
        else:
            if theme == "dark":
                self.btn_pause.setIcon(QIcon("assets/svg/pause_dark.svg"))
            else:
                self.btn_pause.setIcon(QIcon("assets/svg/pause_light.svg"))

    def seek_forward(self):
        if self.cap:
            current = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            self.cap.set(cv2.CAP_PROP_POS_MSEC, current + 5000)

    def seek_backward(self):
        if self.cap:
            current = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            self.cap.set(cv2.CAP_PROP_POS_MSEC, max(0, current - 5000))

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
        video_path = self.video_path.text().strip()
        if not video_path or not os.path.exists(video_path):
            self.result_text.append("请选择有效的视频")
            return
        output_path = self.output_path.text().strip()
        env = os.environ.copy()
        cmd = [
            sys.executable,
            "video_predict.py",
            "--model_path", model_path,
            "--model_type", model_type,
            "--video_path", video_path,
            "--output_path", output_path
        ]
        self.result_text.append("正在视频预测，请稍候...")
        self.btn_predict.setEnabled(False)
        self.is_running = True
        self.predict_thread = PThread(cmd, env, capture_output=True)
        def on_finish(result, error):
            self.btn_predict.setEnabled(True)
            self.is_running = False
            if error:
                self.result_text.append(f"视频预测失败: {error}")
            else:
                self.result_text.append("视频预测已完成，即将开始播放...")
                self.play_video(output_path)
        self.predict_thread.finished.connect(on_finish)
        self.predict_thread.start()

    def play_video(self, video_path):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(video_path)
        self.toggle_pause()
        self.btn_pause.setEnabled(True)
        self.btn_forward.setEnabled(True)
        self.btn_backward.setEnabled(True)
        self.timer.start(30)

    def play_frame(self):
        if self.cap and not self.is_paused:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.video_preview.setPixmap(QPixmap.fromImage(qt_image).scaled(
                    self.video_preview.size(),
                    Qt.KeepAspectRatioByExpanding,
                    Qt.SmoothTransformation
                ))
            else:
                self.cap.release()
                self.cap = None
                self.timer.stop()
                self.btn_pause.setEnabled(False)
                self.btn_forward.setEnabled(False)
                self.btn_backward.setEnabled(False)
                self.toggle_pause()
                self.result_text.append("视频播放结束")
