import os
import sys
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QVBoxLayout, QTextEdit, QHBoxLayout
)
from threads.predict_thread import PThread

STOP_FLAG_FILE = os.path.abspath("stop.flag")

class DedupPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContentsMargins(20, 20, 20, 20)
        layout = QVBoxLayout(self)
        hbox = QHBoxLayout()
        self.btn_dedup = QPushButton("开始去重")
        self.btn_stop_dedup = QPushButton("停止去重")
        hbox.addWidget(self.btn_dedup)
        hbox.addWidget(self.btn_stop_dedup)
        layout.addLayout(hbox)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        layout.addWidget(self.result_text)
        layout.addStretch()
        self.btn_dedup.clicked.connect(self.dedup)
        self.btn_stop_dedup.clicked.connect(self.stop_dedup)
        self.btn_stop_dedup.setEnabled(False)
        self.dedup_thread = None
        self.is_running = False

    def dedup(self):
        if os.path.exists(STOP_FLAG_FILE):
            os.remove(STOP_FLAG_FILE)
        self.result_text.clear()
        env = os.environ.copy()
        cmd = [
            sys.executable,
            "check_and_deduplicate_utkface.py"
        ]
        self.result_text.append("正在去重，请稍候...")
        self.btn_dedup.setEnabled(False)
        self.btn_stop_dedup.setEnabled(True)
        self.is_running = True
        self.dedup_thread = PThread(cmd, env, capture_output=False)
        def on_finish(result, error):
            self.btn_dedup.setEnabled(True)
            self.btn_stop_dedup.setEnabled(False)
            self.is_running = False
            if error:
                self.result_text.append(f"数据集去重失败：{error}")
            else:
                self.result_text.append("数据集去重已完成。")
        self.dedup_thread.finished.connect(on_finish)
        self.dedup_thread.start()

    def stop_dedup(self):
        with open(STOP_FLAG_FILE, "w") as f:
            f.write("stop")