import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QStackedWidget, QSizePolicy, QFileDialog, QMessageBox
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import QSize, QUrl
from PyQt5.QtGui import QIcon, QDesktopServices
from widgets.message_box import MessageBox

class TrainChartPanel(QWidget):
    def __init__(self, parent=None, theme='light'):
        super().__init__(parent)
        self.theme = theme
        layout = QVBoxLayout(self)
        self.chart_btns = []
        charts_names = ['Train Loss', 'Val Age Loss', 'Val Gender Loss', 'Val Gender Acc']
        chart_layout = QHBoxLayout()
        self.btn_back = QPushButton()
        if self.theme == 'light':
            self.btn_back.setIcon(QIcon("assets/svg/back_light.svg"))
        else:
            self.btn_back.setIcon(QIcon("assets/svg/back_dark.svg"))
        self.btn_back.setIconSize(QSize(28, 28))
        self.btn_back.setFixedSize(36, 36)
        self.btn_back.setStyleSheet("border:none; background:transparent;")
        chart_layout.addWidget(self.btn_back)
        for i, name in enumerate(charts_names):
            btn = QPushButton(name)
            btn.clicked.connect(lambda _, idx=i: self.chart_stack.setCurrentIndex(idx))
            self.chart_btns.append(btn)
            chart_layout.addWidget(btn)
        layout.addLayout(chart_layout)

        self.chart_stack = QStackedWidget()
        self.charts = []
        for _ in range(4):
            fig = Figure(figsize=(10, 6))
            canvas = FigureCanvas(fig)
            self.charts.append((fig, canvas))
            self.chart_stack.addWidget(canvas)
        layout.addWidget(self.chart_stack)
        self.chart_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.btn_export = QPushButton("导出图表")
        layout.addWidget(self.btn_export)
        self.btn_clear = QPushButton("清空图表")
        layout.addWidget(self.btn_clear)

        layout.addStretch()
        self.metric_history = {
            "train_loss": [],
            "val_age_loss": [],
            "val_gender_loss": [],
            "val_gender_acc": []
        }
        self.btn_back.clicked.connect(self.go_back)
        self.btn_export.clicked.connect(self.export_plot)
        self.btn_clear.clicked.connect(self.clear_plot)

    def get_current_theme(self):
        theme = "light"
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, "current_theme"):
                theme = parent.current_theme
                break
            parent = parent.parent()
        return theme

    def update_metrics_plot(self, metrics):
        for k in self.metric_history:
            self.metric_history[k] = metrics[k]
        x = list(range(1, len(self.metric_history["train_loss"]) + 1))
        names = ['train_loss', 'val_age_loss', 'val_gender_loss', 'val_gender_acc']
        ylabels = ['Train Loss', 'Val Age Loss', 'Val Gender Loss', 'Val Gender Acc']
        for i, (fig, canvas) in enumerate(self.charts):
            fig.clear()
            ax = fig.add_subplot(111)
            ax.plot(x, self.metric_history[names[i]], marker='o')
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabels[i])
            ax.grid(True)
            canvas.draw()

    def go_back(self):
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, "switch_panel"):
                parent.switch_panel(0)
                break
            parent = parent.parent()

    def export_plot(self):
        self.theme = self.get_current_theme()
        dir_path = QFileDialog.getExistingDirectory(self, "选择导出文件夹")
        if not dir_path:
            return
        names = ['train_loss', 'val_age_loss', 'val_gender_loss', 'val_gender_acc']
        ylabels = ['Train Loss', 'Val Age Loss', 'Val Gender Loss', 'Val Gender Acc']
        for i, (fig, _) in enumerate(self.charts):
            file_path = os.path.join(dir_path, f"{names[i]}.png")
            try:
                fig.savefig(file_path, dpi=150, bbox_inches='tight')
            except Exception as e:
                msg_box = MessageBox(
                    parent=self,
                    title='错误',
                    text=f'导出 {ylabels[i]} 图表失败: {e}',
                    icon=QMessageBox.Critical,
                    theme=self.theme
                )
                msg_box.exec_()
                return
        msg_box = MessageBox(
            parent=self,
            text='导出成功',
            theme=self.theme
        )
        msg_box.exec_()
        QDesktopServices.openUrl(QUrl.fromLocalFile(dir_path))
    
    def clear_plot(self):
        for k in self.metric_history:
            self.metric_history[k] = []
        ylabels = ['Train Loss', 'Val Age Loss', 'Val Gender Loss', 'Val Gender Acc']
        for i, (fig, canvas) in enumerate(self.charts):
            fig.clear()
            ax = fig.add_subplot(111)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabels[i])
            ax.grid(True)
            canvas.draw()