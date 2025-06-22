import os
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QVBoxLayout, QLabel, QTextEdit
)
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QSize

RESULT_LOG_FILE = 'result_log.log'
ERROR_LOG_FILE = 'error_log.log'

class LogPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContentsMargins(20, 20, 20, 20)
        layout = QVBoxLayout(self)
        self.run_log_text = QTextEdit()
        self.run_log_text.setReadOnly(True)
        self.run_log_text.setFixedHeight(300)
        layout.addWidget(QLabel("运行结果日志"))
        layout.addWidget(self.run_log_text)
        self.run_log_refresh_btn = QPushButton()
        self.run_log_refresh_btn.setIcon(QIcon("assets/svg/refresh.svg"))
        self.run_log_refresh_btn.setIconSize(QSize(28, 28))
        self.run_log_refresh_btn.setFixedSize(36, 36)
        self.run_log_refresh_btn.setStyleSheet("border:none; background:transparent;")
        layout.addWidget(self.run_log_refresh_btn)
        self.error_log_text = QTextEdit()
        self.error_log_text.setReadOnly(True)
        self.error_log_text.setFixedHeight(300)
        layout.addWidget(QLabel("错误日志"))
        layout.addWidget(self.error_log_text)
        self.error_log_refresh_btn = QPushButton()
        self.error_log_refresh_btn.setIcon(QIcon("assets/svg/refresh.svg"))
        self.error_log_refresh_btn.setIconSize(QSize(28, 28))
        self.error_log_refresh_btn.setFixedSize(36, 36)
        self.error_log_refresh_btn.setStyleSheet("border:none; background:transparent;")
        layout.addWidget(self.error_log_refresh_btn)
        layout.addStretch()
        self.run_log_refresh_btn.clicked.connect(self.refresh_run_log)
        self.error_log_refresh_btn.clicked.connect(self.refresh_error_log)
        self.refresh_run_log()

    def refresh_run_log(self):
        if os.path.exists(RESULT_LOG_FILE):
            with open(RESULT_LOG_FILE, 'r', encoding='utf-8') as f:
                self.run_log_text.setText(f.read())
        else:
            self.run_log_text.setText("暂无日志。")

    def refresh_error_log(self):
        if os.path.exists(ERROR_LOG_FILE):
            with open(ERROR_LOG_FILE, 'r', encoding='utf-8') as f:
                self.error_log_text.setText(f.read())
        else:
            self.error_log_text.setText("暂无日志。")