from PyQt5.QtWidgets import QDialog, QLabel, QPushButton, QHBoxLayout, QVBoxLayout
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import os

class ImageViewer(QDialog):
    def __init__(self, parent=None, image_paths=None, start_index=0, theme="light"):
        super().__init__(parent)
        self.setWindowTitle("图片查看器")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.resize(800, 600)

        self.image_paths = image_paths or []
        self.current_index = start_index if 0 <= start_index < len(self.image_paths) else 0
        self.theme = theme

        main_layout = QVBoxLayout(self)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumSize(400, 300)
        main_layout.addWidget(self.image_label, stretch=1)

        self.name_label = QLabel()
        self.name_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.name_label)

        btn_layout = QHBoxLayout()
        self.prev_btn = QPushButton("上一张")
        self.prev_btn.clicked.connect(self.show_prev_image)
        btn_layout.addWidget(self.prev_btn)

        self.next_btn = QPushButton("下一张")
        self.next_btn.clicked.connect(self.show_next_image)
        btn_layout.addWidget(self.next_btn)

        # self.close_btn = QPushButton("关闭")
        # self.close_btn.clicked.connect(self.accept)
        # btn_layout.addWidget(self.close_btn)

        main_layout.addLayout(btn_layout)

        self.apply_theme(theme)
        self.update_image()

    def apply_theme(self, theme):
        if theme == "light":
            qss = """
            QDialog {
                background-color: #ffffff;
                color: #222222;
                font-size: 16px;
            }
            QLabel {
                color: #222222;
            }
            QPushButton {
                background-color: #f4f4f5;
                color: #222222;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                padding: 6px 16px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #dbeafe;
                color: #2563eb;
                border: 1px solid #2563eb;
            }
            QPushButton:pressed {
                background-color: #2563eb;
                color: #ffffff;
                border: 1px solid #2563eb;
            }
            """
        else:
            qss = """
            QDialog {
                background-color: #23272e;
                color: #eaeaea;
                font-size: 16px;
            }
            QLabel {
                color: #eaeaea;
            }
            QPushButton {
                background-color: #23272e;
                color: #eaeaea;
                border: 1px solid #555;
                border-radius: 8px;
                padding: 6px 16px;
                min-width: 80px;
            }
            QPushButton:hover {
                background-color: #374151;
                color: #93c5fd;
                border: 1px solid #93c5fd;
            }
            QPushButton:pressed {
                background-color: #2563eb;
                color: #ffffff;
                border: 1px solid #2563eb;
            }
            """
        self.setStyleSheet(qss)

    def update_image(self):
        if not self.image_paths:
            self.image_label.setText("无图片可显示")
            self.name_label.setText("")
            self.prev_btn.setEnabled(False)
            self.next_btn.setEnabled(False)
            return

        image_path = self.image_paths[self.current_index]
        pixmap = QPixmap(image_path)
        if pixmap.isNull():
            self.image_label.setText("无法加载图片")
        else:
            scaled_pixmap = pixmap.scaled(
                self.image_label.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            self.image_label.setPixmap(scaled_pixmap)

        self.name_label.setText(f"{os.path.basename(image_path)} ({self.current_index + 1}/{len(self.image_paths)})")

        self.prev_btn.setEnabled(self.current_index > 0)
        self.next_btn.setEnabled(self.current_index < len(self.image_paths) - 1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_image()

    def show_prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.update_image()

    def show_next_image(self):
        if self.current_index < len(self.image_paths) - 1:
            self.current_index += 1
            self.update_image()
