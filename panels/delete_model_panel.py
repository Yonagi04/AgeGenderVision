import os
import json
from PyQt5.QtWidgets import (
    QWidget, QPushButton, QVBoxLayout, QLabel, QLineEdit,
    QTextEdit, QComboBox
)
from utils.model_utils import refresh_model_list

MODELS_INFO_FILE = 'data/models.json'

class DeleteModelPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContentsMargins(20, 20, 20, 20)
        layout = QVBoxLayout(self)
        self.model_combo = QComboBox()
        self.refresh_models()
        layout.addWidget(QLabel("选择要删除的模型"))
        layout.addWidget(self.model_combo)
        self.confirm_edit = QLineEdit()
        self.confirm_edit.setPlaceholderText("再次输入模型名以确认删除")
        layout.addWidget(self.confirm_edit)
        self.btn_delete = QPushButton("删除模型")
        layout.addWidget(self.btn_delete)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        layout.addWidget(self.result_text)
        layout.addStretch()
        self.btn_delete.clicked.connect(self.delete_model)

    def refresh_models(self):
        self.model_combo.clear()
        models = refresh_model_list()
        self.model_combo.addItems(models)

    def delete_model(self):
        self.result_text.clear()
        model = self.model_combo.currentText()
        confirm = self.confirm_edit.text().strip()
        if not model:
            self.result_text.append("暂无模型文件")
            return
        if confirm != model:
            self.result_text.append("模型名称输入错误，未删除任何文件。")
            return
        try:
            os.remove(model)
            if os.path.exists(MODELS_INFO_FILE):
                with open(MODELS_INFO_FILE, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                if model in info:
                    del info[model]
                    with open(MODELS_INFO_FILE, 'w', encoding='utf-8') as f:
                        json.dump(info, f, ensure_ascii=False, indent=2)
            self.result_text.append(f"模型 {model} 已成功删除。")
            self.confirm_edit.clear()
            self.refresh_models()
        except Exception as e:
            self.result_text.append(f"删除失败：{e}")