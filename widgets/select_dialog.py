from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QLabel, QComboBox, QDialogButtonBox
)
from PyQt5.QtCore import Qt

class SelectDialog(QDialog):
    def __init__(self, parent=None, title='选择', text='请选择：', options=None, default_index=0, theme='light'):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.theme = theme
        self.selected = None
        self.setMinimumWidth(300)
        self.init_ui(text, options or [], default_index)
        self.apply_theme(theme)

    def init_ui(self, text, options, default_index):
        layout = QVBoxLayout(self)
        self.label = QLabel(text)
        self.combo = QComboBox()
        self.combo.addItems(options)
        if 0 <= default_index < len(options):
            self.combo.setCurrentIndex(default_index)

        layout.addWidget(self.label)
        layout.addWidget(self.combo)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

    def accept(self):
        self.selected = self.combo.currentText()
        super().accept()

    def get_value(self):
        return self.selected

    def apply_theme(self, theme):
        if theme == "light":
            qss = """
            QDialog {
                background-color: #ffffff;
                color: #222222;
                font-size: 16px;
                border-radius: 12px;
                border: 1px solid #ccc;
            }
            QLabel {
                color: #222222;
            }
            QComboBox {
                background-color: #f9f9f9;
                color: #222222;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                padding: 6px 10px;
            }
            QComboBox QAbstractItemView {
                background-color: #ffffff;
                selection-background-color: #e0f2fe;
                selection-color: #1e40af;
            }
            QDialogButtonBox QPushButton {
                background-color: #f4f4f5;
                color: #222222;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                padding: 6px 16px;
            }
            QDialogButtonBox QPushButton:hover {
                background-color: #dbeafe;
                color: #2563eb;
                border: 1px solid #2563eb;
            }
            QDialogButtonBox QPushButton:pressed {
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
                border-radius: 12px;
                border: 1px solid #444;
            }
            QLabel {
                color: #eaeaea;
            }
            QComboBox {
                background-color: #2c2f36;
                color: #eaeaea;
                border: 1px solid #555;
                border-radius: 8px;
                padding: 6px 10px;
            }
            QComboBox QAbstractItemView {
                background-color: #2c2f36;
                selection-background-color: #374151;
                selection-color: #93c5fd;
            }
            QDialogButtonBox QPushButton {
                background-color: #23272e;
                color: #eaeaea;
                border: 1px solid #555;
                border-radius: 8px;
                padding: 6px 16px;
            }
            QDialogButtonBox QPushButton:hover {
                background-color: #374151;
                color: #93c5fd;
                border: 1px solid #93c5fd;
            }
            QDialogButtonBox QPushButton:pressed {
                background-color: #2563eb;
                color: #ffffff;
                border: 1px solid #2563eb;
            }
            """
        self.setStyleSheet(qss)
