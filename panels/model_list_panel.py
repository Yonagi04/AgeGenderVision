from PyQt5.QtWidgets import (
    QWidget, QPushButton, QVBoxLayout, QLabel, QTextEdit
)
from utils.model_utils import refresh_model_list

class ModelListPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContentsMargins(20, 20, 20, 20)
        layout = QVBoxLayout(self)
        self.model_list = QTextEdit()
        self.model_list.setReadOnly(True)
        layout.addWidget(QLabel("已训练模型列表"))
        layout.addWidget(self.model_list)
        self.refresh_btn = QPushButton("刷新")
        layout.addWidget(self.refresh_btn)
        layout.addStretch()
        self.refresh_btn.clicked.connect(self.refresh)
        self.refresh()

    def refresh(self):
        models = refresh_model_list()
        if not models:
            self.model_list.setText("暂无模型文件")
        else:
            self.model_list.setText('\n'.join(models))