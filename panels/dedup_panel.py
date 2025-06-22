import os
import sys
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QVBoxLayout, QTextEdit
)
from threads.predict_thread import PThread

class DedupPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContentsMargins(20, 20, 20, 20)
        layout = QVBoxLayout(self)
        self.btn_dedup = QPushButton("自动去重数据集")
        layout.addWidget(self.btn_dedup)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        layout.addWidget(self.result_text)
        layout.addStretch()
        self.btn_dedup.clicked.connect(self.dedup)
        self.dedup_thread = None

    def dedup(self):
        self.result_text.clear()
        env = os.environ.copy()
        cmd = f"{sys.executable} check_and_deduplicate_utkface.py"
        self.result_text.append("正在去重，请稍候...")
        self.btn_dedup.setEnabled(False)
        self.dedup_thread = PThread(cmd, env, capture_output=False)
        def on_finish(result, error):
            self.btn_dedup.setEnabled(True)
            if error:
                self.result_text.append(f"数据集去重失败：{error}")
            else:
                self.result_text.append("数据集去重已完成。")
        self.dedup_thread.finished.connect(on_finish)
        self.dedup_thread.start()