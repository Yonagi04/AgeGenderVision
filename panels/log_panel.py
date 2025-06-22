import os
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QVBoxLayout, QLabel, QTextEdit
)

RESULT_LOG_FILE = 'result_log.log'

class LogPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContentsMargins(20, 20, 20, 20)
        layout = QVBoxLayout(self)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(QLabel("结果日志"))
        layout.addWidget(self.log_text)
        self.refresh_btn = QPushButton("刷新")
        layout.addWidget(self.refresh_btn)
        layout.addStretch()
        self.refresh_btn.clicked.connect(self.refresh)
        self.refresh()

    def refresh(self):
        if os.path.exists(RESULT_LOG_FILE):
            with open(RESULT_LOG_FILE, 'r', encoding='utf-8') as f:
                self.log_text.setText(f.read())
        else:
            self.log_text.setText("暂无日志。")