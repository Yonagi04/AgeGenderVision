from PyQt5.QtWidgets import QWidget, QLabel
from widgets.flow_layout import FlowLayout

def get_contrast_font_color(bg_color: str):
    bg_color = bg_color.lstrip("#")
    r, g, b = int(bg_color[0:2], 16), int(bg_color[2:4], 16), int(bg_color[4:6], 16)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b)
    return "#000000" if luminance > 186 else "#ffffff"
        
def create_tag_container(tags: list) -> QWidget:
    if not tags:
        label_no_tag = QLabel("无标签")
        label_no_tag.setStyleSheet("color: gray; font-size: 18px;")
        return label_no_tag
    flow = FlowLayout(spacing=6)
    for tag in tags:
        label = QLabel(tag["text"])
        color = tag["color"]
        text_color = get_contrast_font_color(color)
        label.setStyleSheet(f"""
            QLabel {{
                background-color: {color};
                color: {text_color};
                border-radius: 6px;
                padding: 2px 6px;
                font-size: 16px;
                min-height: 30px;
                max-height: 40px;
            }}
        """)
        label.setFixedHeight(24)
        flow.addWidget(label)
    tag_container = QWidget()
    tag_container.setLayout(flow)
    tag_container.setStyleSheet("background-color: transparent;")
    return tag_container