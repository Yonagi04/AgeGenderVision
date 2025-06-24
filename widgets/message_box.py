from PyQt5.QtWidgets import QMessageBox, QPushButton
from PyQt5.QtCore import Qt

class MessageBox(QMessageBox):
    def __init__(self, parent=None, title="提示", text="内容", theme="light", icon=QMessageBox.Information, addButton=True):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setText(text)
        self.setIcon(icon)
        if addButton:
            self.setStandardButtons(QMessageBox.Ok)
        self.apply_theme(theme)
        self.setMinimumWidth(300)

    def apply_theme(self, theme: str):
        if theme == "light":
            qss = """
            QMessageBox {
                background-color: #ffffff;
                color: #222222;
                font-size: 16px;
                border-radius: 12px;
                border: 1px solid #ccc;
            }

            QMessageBox QLabel,
            QMessageBox QTextEdit,
            QMessageBox QPlainTextEdit,
            QMessageBox QTextBrowser {
                color: #222222;
                background-color: transparent;
            }

            QMessageBox QPushButton {
                background-color: #f4f4f5;
                color: #222222;
                border: 1px solid #d1d5db;
                border-radius: 8px;
                padding: 6px 16px;
            }
            QMessageBox QPushButton:hover {
                background-color: #dbeafe;
                color: #2563eb;
                border: 1px solid #2563eb;
            }
            QMessageBox QPushButton:pressed {
                background-color: #2563eb;
                color: #ffffff;
                border: 1px solid #2563eb;
            }
            """
        else:
            qss = """
            QMessageBox {
                background-color: #23272e;
                color: #eaeaea;
                font-size: 16px;
                border-radius: 12px;
                border: 1px solid #444;
            }

            QMessageBox QLabel,
            QMessageBox QTextEdit,
            QMessageBox QPlainTextEdit,
            QMessageBox QTextBrowser {
                color: #eaeaea;
                background-color: transparent;
            }

            QMessageBox QPushButton {
                background-color: #23272e;
                color: #eaeaea;
                border: 1px solid #555;
                border-radius: 8px;
                padding: 6px 16px;
            }
            QMessageBox QPushButton:hover {
                background-color: #374151;
                color: #93c5fd;
                border: 1px solid #93c5fd;
            }
            QMessageBox QPushButton:pressed {
                background-color: #2563eb;
                color: #ffffff;
                border: 1px solid #2563eb;
            }
            """
        self.setStyleSheet(qss)

    def add_buttons(self, buttons: dict):
        """buttons: {text: role}, returns: {text: QPushButton}"""
        self.custom_buttons = {}
        for label, role in buttons.items():
            btn = self.addButton(label, role)
            self.custom_buttons[label] = btn
        return self.custom_buttons

    def get_clicked_button(self):
        if hasattr(self, 'custom_buttons'):
            for text, btn in self.custom_buttons.items():
                if self.clickedButton() == btn:
                    return text
        return None
