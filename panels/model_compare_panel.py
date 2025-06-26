from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QGroupBox, QPushButton, QSpacerItem, QSizePolicy, QMessageBox
)
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QIcon
from widgets.message_box import MessageBox
from utils.ui_utils import create_tag_container
from services.model_service import ModelService
from convention.result_code import ResultCode

class ModelComparePanel(QWidget):
    def __init__(self, parent=None, theme='light'):
        super().__init__(parent)
        self.theme = theme
        self.models_info = None

        result = ModelService.load_model_info()
        if not result.success or result.code == ResultCode.NO_DATA:
            self.models_info = {}
        else:
            self.models_info = result.data

        self.model_names = list(self.models_info.keys())

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        title = QLabel("模型对比")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 32px; font-weight: bold; margin-bottom: 20px;")
        main_layout.addWidget(title)

        select_layout = QHBoxLayout()
        select_layout.addWidget(QLabel("选择模型A:"))
        self.combo_a = QComboBox()
        self.combo_a.addItems(self.model_names)
        select_layout.addWidget(self.combo_a)

        select_layout.addSpacing(40)

        select_layout.addWidget(QLabel("选择模型B:"))
        self.combo_b = QComboBox()
        self.combo_b.addItems(self.model_names)
        select_layout.addWidget(self.combo_b)

        self.btn_refresh = QPushButton()
        if self.theme == 'light':
            self.btn_refresh.setIcon(QIcon("assets/svg/refresh_light.svg"))
        else:
            self.btn_refresh.setIcon(QIcon("assets/svg/refresh_dark.svg"))
        self.btn_refresh.setIconSize(QSize(28, 28))
        self.btn_refresh.setFixedSize(36, 36)
        self.btn_refresh.setStyleSheet("border:none; background:transparent;")
        select_layout.addWidget(self.btn_refresh)

        main_layout.addLayout(select_layout)

        self.compare_layout = QHBoxLayout()
        self.card_a = QGroupBox("模型 A")
        self.card_b = QGroupBox("模型 B")

        self.card_a_layout = QVBoxLayout(self.card_a)
        self.card_b_layout = QVBoxLayout(self.card_b)

        self.compare_layout.addWidget(self.card_a)
        self.compare_layout.addWidget(self.card_b)
        main_layout.addLayout(self.compare_layout)

        if self.model_names:
            self.combo_a.setCurrentIndex(0)
            self.combo_b.setCurrentIndex(min(1, len(self.model_names)-1))
            self.do_compare()
        
        self.combo_a.currentIndexChanged.connect(self.do_compare)
        self.combo_b.currentIndexChanged.connect(self.do_compare)
        self.btn_refresh.clicked.connect(self.refresh)

    def do_compare(self):
        model_a = self.combo_a.currentText()
        model_b = self.combo_b.currentText()
        info_a = self.models_info.get(model_a, {})
        info_b = self.models_info.get(model_b, {})

        def update_model_card(layout, info):
            while layout.count() > 0:
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

            tags = info.get("tags", [])
            if tags:
                tag_container = create_tag_container(tags)
                layout.addWidget(tag_container)
            else:
                no_tag_label = QLabel("无标签")
                no_tag_label.setStyleSheet("color: gray; font-size: 18px;")
                layout.addWidget(no_tag_label)

            def make_info_label(title, value):
                label = QLabel(f"<b>{title}：</b>{value}")
                label.setStyleSheet("font-size: 18px; margin-top: 4px;")
                return label

            layout.addWidget(make_info_label("模型名称", info.get("model_name", "-")))
            layout.addWidget(make_info_label("模型类型", info.get("model_type", "-")))
            layout.addWidget(make_info_label("创建时间", info.get("created_time", "-")))

            layout.addWidget(make_info_label("训练轮数", info.get("epochs", "-")))
            layout.addWidget(make_info_label("Batch Size", info.get("batch_size", "-")))
            layout.addWidget(make_info_label("图片尺寸", info.get("img_size", "-")))

            eval_result = info.get("eval_result", {})
            layout.addWidget(make_info_label("性别准确率", f"{eval_result.get('val_acc', '-')}"))
            layout.addWidget(make_info_label("年龄损失", f"{eval_result.get('val_age_loss', '-')}"))
            layout.addWidget(make_info_label("性别损失", f"{eval_result.get('val_gender_loss', '-')}"))

            layout.addSpacerItem(QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding))

        update_model_card(self.card_a.layout(), info_a)
        update_model_card(self.card_b.layout(), info_b)

    def refresh(self):
        try:
            result = ModelService.load_model_info()
            if not result.success or result.code == ResultCode.NO_DATA:
                self.models_info = {}
            else:
                self.models_info = result.data
            self.model_names = list(self.models_info.keys())
            self.combo_a.clear()
            self.combo_b.clear()
            self.combo_a.addItems(self.model_names)
            self.combo_b.addItems(self.model_names)
            self.combo_a.setCurrentIndex(0)
            self.combo_b.setCurrentIndex(min(1, len(self.model_names) - 1))

            self.do_compare()
        except Exception as e:
            msg_box = MessageBox(
                parent=self,
                title="错误",
                text=f"刷新失败: {e}",
                theme=self.theme,
                icon=QMessageBox.Critical
            )
            msg_box.exec_()