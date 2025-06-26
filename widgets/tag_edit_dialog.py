from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit, QColorDialog,
    QWidget, QScrollArea, QGridLayout
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
import copy
from widgets.flow_layout import FlowLayout

class TagEditDialog(QDialog):
    def __init__(self, tags=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("标签编辑")
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setMinimumWidth(400)
        self.tags = copy.deepcopy(tags) if tags else []
        self.new_tags = []
        self.quick_tags = [
            {"text": "实验模型", "color": "#42a5f5"},
            {"text": "正式模型", "color": "#66bb6a"},
            {"text": "临时测试", "color": "#ffa726"}
        ]

        self.layout = QVBoxLayout(self)
        self.tag_area = QScrollArea()
        self.tag_container = QWidget()
        self.tag_layout = QGridLayout(self.tag_container)
        self.tag_area.setWidgetResizable(True)
        self.tag_area.setWidget(self.tag_container)

        self.layout.addWidget(QLabel("快捷标签:"))
        quick_tag_widget = QWidget()
        quick_tag_layout = FlowLayout(spacing=6)
        for tag in self.quick_tags:
            label = QLabel(tag["text"])
            color = tag["color"]
            text_color = self.get_contrast_font_color(color)
            label.setStyleSheet(f"""
                QLabel {{
                    background-color: {color};
                    color: {text_color};
                    border-radius: 6px;
                    padding: 2px 6px;
                    font-size: 14px;
                    min-height: 30px;
                    max-height: 40px
                }}
            """)
            label.setFixedHeight(24)
            label.setCursor(Qt.PointingHandCursor)
            label.mousePressEvent = self.make_quick_tag_handler(tag)
            quick_tag_layout.addWidget(label)
        quick_tag_widget.setLayout(quick_tag_layout)
        self.layout.addWidget(quick_tag_widget)

        self.layout.addWidget(QLabel("已有标签:"))
        self.layout.addWidget(self.tag_area)

        self.input_layout = QHBoxLayout()
        self.tag_input = QLineEdit()
        self.tag_input.setPlaceholderText("输入标签文字")
        self.color_btn = QPushButton("选择颜色")
        self.color_btn.clicked.connect(self.select_color)
        self.selected_color = "#5c6bc0"
        self.input_layout.addWidget(self.tag_input)
        self.input_layout.addWidget(self.color_btn)

        self.layout.addLayout(self.input_layout)

        self.btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("添加标签")
        self.ok_btn = QPushButton("确定")
        self.cancel_btn = QPushButton("取消")

        self.add_btn.clicked.connect(self.add_tag)
        self.ok_btn.clicked.connect(self.accept)
        self.cancel_btn.clicked.connect(self.reject)

        self.btn_layout.addWidget(self.add_btn)
        self.btn_layout.addStretch()
        self.btn_layout.addWidget(self.ok_btn)
        self.btn_layout.addWidget(self.cancel_btn)

        self.layout.addLayout(self.btn_layout)

        self.refresh_tags()

    def select_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.selected_color = color.name()

    def add_tag(self):
        text = self.tag_input.text().strip()
        if not text:
            return
        tag = {"text": text, "color": self.selected_color}
        self.tags.append(tag)
        self.tag_input.clear()
        self.refresh_tags()

    def remove_tag(self, tag):
        self.tags.remove(tag)
        self.refresh_tags()

    def refresh_tags(self):
        for i in reversed(range(self.tag_layout.count())):
            widget = self.tag_layout.itemAt(i).widget()
            if widget:
                widget.setParent(None)

        for idx, tag in enumerate(self.tags):
            tag_widget = self.create_tag_widget(tag)
            self.tag_layout.addWidget(tag_widget, idx // 3, idx % 3)

    def create_tag_widget(self, tag):
        color = tag.get("color", "#aaa")
        text = tag.get("text", "")
        widget = QWidget()
        layout = QHBoxLayout(widget)
        label = QLabel(text)
        label.setStyleSheet(f"background-color: {color}; color: white; border-radius: 4px; padding: 2px 6px;")
        remove_btn = QPushButton("×")
        remove_btn.setFixedWidth(20)
        remove_btn.clicked.connect(lambda: self.remove_tag(tag))
        remove_btn.setStyleSheet("border: none; color: red; font-weight: bold;")
        layout.addWidget(label)
        layout.addWidget(remove_btn)
        layout.setContentsMargins(0, 0, 0, 0)
        return widget

    def get_tags(self):
        return self.tags

    def make_quick_tag_handler(self, tag):
        def handler(event):
            if tag not in self.tags:
                self.tags.append(tag.copy())
                self.refresh_tags()
        return handler
    
    def get_contrast_font_color(self, bg_color: str):
        bg_color = bg_color.lstrip("#")
        r, g, b = int(bg_color[0:2], 16), int(bg_color[2:4], 16), int(bg_color[4:6], 16)
        luminance = (0.299 * r + 0.587 * g + 0.114 * b)
        return "#000000" if luminance > 186 else "#ffffff"