from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QStackedWidget
)
from PyQt5.QtCore import QSize
from PyQt5.QtGui import QIcon
from panels.train_panel import TrainPanel
from panels.model_list_panel import ModelListPanel
from panels.predict_image_panel import PredictImagePanel
from panels.predict_multi_image_panel import PredictMultiImagePanel
from panels.predict_video_panel import PredictVideoPanel
from panels.dedup_panel import DedupPanel
from panels.delete_model_panel import DeleteModelPanel
from panels.log_panel import LogPanel
from utils.qss_loader import load_qss

LIGHT_QSS_FILE = 'assets/light.qss'
DARK_QSS_FILE = 'assets/dark.qss'

class MainPanelWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("年龄性别识别系统")
        self.setGeometry(200, 200, 1200, 800)
        self.current_theme = "light"
        main_layout = QHBoxLayout(self)

        menu_layout = QVBoxLayout()
        menu_layout.setContentsMargins(10, 10, 10, 10)
        menu_layout.setSpacing(10)
        self.btn_train = QPushButton("训练模型")
        self.btn_models = QPushButton("模型管理")
        self.btn_predict_img = QPushButton("图片预测")
        self.btn_predict_multi_img = QPushButton("多模型图片预测")
        self.btn_predict_video = QPushButton("视频预测")
        self.btn_dedup = QPushButton("数据集去重")
        self.btn_log = QPushButton("查看日志")
        self.btn_theme = QPushButton()
        self.btn_theme.setIcon(QIcon("assets/svg/moon.svg"))
        self.btn_theme.setIconSize(QSize(28, 28))
        self.btn_theme.setFixedSize(36, 36)
        self.btn_theme.setStyleSheet("border:none; background:transparent;")
        menu_layout.addWidget(self.btn_train)
        menu_layout.addWidget(self.btn_models)
        menu_layout.addWidget(self.btn_predict_img)
        menu_layout.addWidget(self.btn_predict_multi_img)
        menu_layout.addWidget(self.btn_predict_video)
        menu_layout.addWidget(self.btn_dedup)
        menu_layout.addWidget(self.btn_log)
        menu_layout.addStretch()
        menu_layout.addWidget(self.btn_theme)
        for btn in [
            self.btn_train, self.btn_models, self.btn_predict_img, self.btn_predict_multi_img,
            self.btn_predict_video, self.btn_dedup, self.btn_log
        ]:
            btn.setObjectName("menuButton")

        self.stack = QStackedWidget()
        self.train_panel = TrainPanel()
        self.model_list_panel = ModelListPanel()
        self.predict_img_panel = PredictImagePanel()
        self.predict_multi_img_panel = PredictMultiImagePanel()
        self.predict_video_panel = PredictVideoPanel()
        self.dedup_panel = DedupPanel()
        self.log_panel = LogPanel()
        self.stack.addWidget(self.train_panel)
        self.stack.addWidget(self.model_list_panel)
        self.stack.addWidget(self.predict_img_panel)
        self.stack.addWidget(self.predict_multi_img_panel)
        self.stack.addWidget(self.predict_video_panel)
        self.stack.addWidget(self.dedup_panel)
        self.stack.addWidget(self.log_panel)
        main_layout.addLayout(menu_layout, 1)
        main_layout.addWidget(self.stack, 4)

        self.current_panel_idx = 0
        self.btn_train.clicked.connect(lambda: self.switch_panel(0))
        self.btn_models.clicked.connect(lambda: (self.model_list_panel.refresh(), self.switch_panel(1)))
        self.btn_predict_img.clicked.connect(lambda: (self.predict_img_panel.refresh_models(), self.switch_panel(2)))
        self.btn_predict_multi_img.clicked.connect(lambda: (self.predict_img_panel.refresh_models(), self.switch_panel(3)))
        self.btn_predict_video.clicked.connect(lambda: (self.predict_video_panel.refresh_models(), self.switch_panel(4)))
        self.btn_dedup.clicked.connect(lambda: self.switch_panel(5))
        self.btn_log.clicked.connect(lambda: (self.log_panel.refresh_run_log(), self.log_panel.refresh_error_log(), self.switch_panel(6)))
        self.btn_theme.clicked.connect(self.toggle_theme)
        self.switch_panel(0)

    def switch_panel(self, idx):
        if idx != self.current_panel_idx:
            if idx == 0:
                self.train_panel.log_text.clear()
            elif idx == 2:
                self.predict_img_panel.result_text.clear()
            elif idx == 3:
                self.predict_multi_img_panel.result_text.clear()
                self.predict_multi_img_panel.refresh_models()
            elif idx == 4:
                self.predict_video_panel.result_text.clear()
            elif idx == 5:
                self.dedup_panel.result_text.clear()
        self.stack.setCurrentIndex(idx)
        self.current_panel_idx = idx

    def toggle_theme(self):
        app = QApplication.instance()
        if self.current_theme == "light":
            load_qss(app, DARK_QSS_FILE)
            self.btn_theme.setIcon(QIcon("assets/svg/sun.svg"))
            self.current_theme = "dark"
        else:
            load_qss(app, LIGHT_QSS_FILE)
            self.btn_theme.setIcon(QIcon("assets/svg/moon.svg"))
            self.current_theme = "light"