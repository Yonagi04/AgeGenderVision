import sys
import os
import json
import datetime
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit,
    QTextEdit, QComboBox, QFormLayout, QFileDialog, QProgressBar, QCheckBox, QStackedWidget,
    QProgressDialog, QSizePolicy, QSpacerItem, QMessageBox, QDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize
from PyQt5.QtGui import QIcon, QTextCursor

RESULT_LOG_FILE = 'result_log.log'
MODELS_INFO_FILE = 'data/models.json'
STOP_FLAG_FILE = os.path.abspath("stop.flag")
LIGHT_QSS_FILE = 'assets/light.qss'
DARK_QSS_FILE = 'assets/dark.qss'

def refresh_model_list():
    return [f for f in os.listdir('.') if f.endswith('.pth')]

def get_model_type(model_path):
    if os.path.exists(MODELS_INFO_FILE):
        with open(MODELS_INFO_FILE, 'r', encoding='utf-8') as f:
            info = json.load(f)
        t = info.get(model_path)
        if t:
            return t
    return 'resnet18'

def load_qss(app, qss_file):
    with open(qss_file, encoding='utf-8') as f:
        app.setStyleSheet(f.read())

class TrainThread(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished = pyqtSignal(object)

    def __init__(self, cmd, env):
        super().__init__()
        self.cmd = cmd
        self.env = env
        self._process = None

    def run(self):
        import subprocess
        try:
            self._process = subprocess.Popen(
                self.cmd, shell=True, env=self.env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                bufsize=1, universal_newlines=True
            )
            epoch = 0
            total_epochs = None
            for line in self._process.stdout:
                self.log_signal.emit(line)
                if "Epoch" in line:
                    import re
                    m = re.search(r"Epoch\s+(\d+)/(\d+)", line)
                    if m:
                        epoch = int(m.group(1))
                        total_epochs = int(m.group(2))
                        if total_epochs:
                            percent = int(epoch / total_epochs * 100)
                            self.progress_signal.emit(percent)
                if os.path.exists(STOP_FLAG_FILE):
                    self._process.terminate()
                    break
            self._process.wait()
            self.finished.emit(None)
        except Exception as e:
            self.finished.emit(e)

    def stop(self):
        with open(STOP_FLAG_FILE, "w") as f:
            f.write("stop")

class PThread(QThread):
    finished = pyqtSignal(str, object)

    def __init__(self, cmd, env, capture_output=True):
        super().__init__()
        self.cmd = cmd
        self.env = env
        self.capture_output = capture_output

    def run(self):
        import subprocess
        try:
            if self.capture_output:
                result = subprocess.run(self.cmd, shell=True, check=True, capture_output=True, text=True, env=self.env)
                self.finished.emit(result.stdout, None)
            else:
                subprocess.run(self.cmd, shell=True, check=True, env=self.env)
                self.finished.emit("", None)
        except Exception as e:
            self.finished.emit("", e)

class TrainPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContentsMargins(20, 20, 20, 20)
        layout = QVBoxLayout(self)
        form = QFormLayout()
        self.batch_size = QLineEdit("64")
        self.epochs = QLineEdit("10")
        self.lr = QLineEdit("0.001")
        self.img_size = QLineEdit("224")
        self.data_dir = QLineEdit("data/UTKFace/cleaned")
        self.model_type = QComboBox()
        self.model_type.addItems(['resnet18', 'resnet34', 'resnet50'])
        self.model_path = QLineEdit("age_gender_multitask_resnet18.pth")
        form.addRow("Batch size", self.batch_size)
        form.addRow("Epochs", self.epochs)
        form.addRow("Learning rate", self.lr)
        form.addRow("Image size", self.img_size)
        form.addRow("数据集目录", self.data_dir)
        form.addRow("模型类型", self.model_type)
        form.addRow("模型保存路径", self.model_path)
        layout.addLayout(form)
        self.btn_train = QPushButton("开始训练")
        self.btn_stop = QPushButton("停止训练")
        self.btn_stop.setEnabled(False)
        btn_h = QHBoxLayout()
        btn_h.addWidget(self.btn_train)
        btn_h.addWidget(self.btn_stop)
        layout.addLayout(btn_h)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        layout.addWidget(self.progress)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(self.log_text)
        layout.addStretch()
        self.btn_train.clicked.connect(self.start_train)
        self.btn_stop.clicked.connect(self.stop_train)
        self.train_thread = None

    def start_train(self):
        try:
            batch_size = int(self.batch_size.text())
            epochs = int(self.epochs.text())
            lr = float(self.lr.text())
            img_size = int(self.img_size.text())
            data_dir = self.data_dir.text().strip()
            model_type = self.model_type.currentText()
            model_path = self.model_path.text().strip()
            if batch_size <= 0 or epochs <= 0 or lr <= 0 or img_size <= 0:
                raise ValueError("Batch size、Epochs、Learning rate、Image size 必须为正数")
            if not os.path.isdir(data_dir):
                raise ValueError("数据集目录不存在")
            if not model_path.lower().endswith('.pth'):
                model_path += '.pth'
            if not model_path:
                raise ValueError("模型保存路径不能为空")
        except Exception as e:
            self.log_text.append(f"参数错误：{e}")
            return
        if os.path.exists(STOP_FLAG_FILE):
            os.remove(STOP_FLAG_FILE)
        cmd = (f"{sys.executable} train_age_gender_multitask.py "
               f"--batch_size {batch_size} --epochs {epochs} --lr {lr} "
               f"--img_size {img_size} --data_dir \"{data_dir}\" --model_type \"{model_type}\" --model_path \"{model_path}\"")
        env = os.environ.copy()
        self.log_text.clear()
        self.progress.setValue(0)
        self.btn_train.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.train_thread = TrainThread(cmd, env)
        self.train_thread.log_signal.connect(self.log_text.append)
        self.train_thread.progress_signal.connect(self.progress.setValue)
        def on_finish(error):
            self.btn_train.setEnabled(True)
            self.btn_stop.setEnabled(False)
            if error:
                self.log_text.append(f"训练失败：{error}")
            else:
                self.log_text.append("训练完成！")
                if os.path.exists(model_path):
                    try:
                        if os.path.exists(MODELS_INFO_FILE):
                            with open(MODELS_INFO_FILE, 'r', encoding='utf-8') as f:
                                info = json.load(f)
                        else:
                            info = {}
                        info[model_path] = model_type
                        with open(MODELS_INFO_FILE, 'w', encoding='utf-8') as f:
                            json.dump(info, f, ensure_ascii=False, indent=2)
                    except Exception as e:
                        self.log_text.append(f"模型信息写入失败：{e}")
                else:
                    self.log_text.append("模型保存失败！请排查原因。")
        self.train_thread.finished.connect(on_finish)
        self.train_thread.start()

    def stop_train(self):
        if self.train_thread:
            self.train_thread.stop()
            self.log_text.append("已请求停止训练...")

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

class PredictImagePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContentsMargins(20, 20, 20, 20)
        layout = QVBoxLayout(self)
        self.model_combo = QComboBox()
        self.refresh_models()
        self.img_path = QLineEdit()
        self.img_path.setPlaceholderText("请选择图片")
        btn_img = QPushButton("选择图片")
        btn_img.clicked.connect(self.select_image)
        h_img = QHBoxLayout()
        h_img.addWidget(self.img_path)
        h_img.addWidget(btn_img)
        layout.addWidget(QLabel("选择模型"))
        layout.addWidget(self.model_combo)
        layout.addWidget(QLabel("选择图片"))
        layout.addLayout(h_img)
        self.btn_predict = QPushButton("开始预测")
        layout.addWidget(self.btn_predict)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        layout.addWidget(self.result_text)
        layout.addStretch()
        self.btn_predict.clicked.connect(self.predict)
        self.predict_thread = None

    def refresh_models(self):
        self.model_combo.clear()
        models = refresh_model_list()
        self.model_combo.addItems(models)

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)")
        if path:
            self.img_path.setText(path)

    def predict(self):
        model_path = self.model_combo.currentText()
        if not model_path:
            self.result_text.append("请先训练模型")
            return
        model_type = get_model_type(model_path)
        img_path = self.img_path.text().strip()
        if not img_path or not os.path.exists(img_path):
            self.result_text.append("请选择有效的图片")
            return
        env = os.environ.copy()
        env["IS_SUBPROCESS"] = "1"
        cmd = f"{sys.executable} photo_predict.py --model_path \"{model_path}\" --model_type \"{model_type}\" --img_path \"{img_path}\""
        self.result_text.append("正在预测，请稍候...")
        self.btn_predict.setEnabled(False)
        self.predict_thread = PThread(cmd, env, capture_output=True)
        def on_finish(result, error):
            self.btn_predict.setEnabled(True)
            if error:
                self.result_text.append(f"图片预测失败：{error}")
            else:
                self.result_text.append(f"预测结果：\n{result.strip()}")
        self.predict_thread.finished.connect(on_finish)
        self.predict_thread.start()

class PredictMultiImagePanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContentsMargins(20, 20, 20, 20)
        layout = QVBoxLayout(self)
        self.model_checks = []
        self.models = []
        self.refresh_models()
        self.img_path = QLineEdit()
        self.img_path.setPlaceholderText("请选择图片")
        btn_img = QPushButton("选择图片")
        btn_img.clicked.connect(self.select_image)
        h_img = QHBoxLayout()
        h_img.addWidget(self.img_path)
        h_img.addWidget(btn_img)
        layout.addWidget(QLabel("选择模型（可多选）"))
        self.models_layout = QVBoxLayout()
        for cb in self.model_checks:
            self.models_layout.addWidget(cb)
        layout.addLayout(self.models_layout)
        layout.addWidget(QLabel("选择图片"))
        layout.addLayout(h_img)
        self.btn_predict = QPushButton("开始多模型预测")
        layout.addWidget(self.btn_predict)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        layout.addWidget(self.result_text)
        layout.addStretch()
        self.btn_predict.clicked.connect(self.predict)
        self.predict_threads = []
    
    def refresh_models(self):
        # 清空旧的checkbox
        self.models = refresh_model_list()
        self.model_checks = []
        if hasattr(self, "models_layout"):
            while self.models_layout.count():
                item = self.models_layout.takeAt(0)
                widget = item.widget()
                if widget:
                    widget.deleteLater()
        for m in self.models:
            cb = QCheckBox(m)
            self.model_checks.append(cb)
            if hasattr(self, "models_layout"):
                self.models_layout.addWidget(cb)

    def select_image(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)")
        if path:
            self.img_path.setText(path)

    def predict(self):
        selected = [cb.text() for cb in self.model_checks if cb.isChecked()]
        if not selected:
            self.result_text.append("请至少选择一个模型进行比较")
            return
        img_path = self.img_path.text().strip()
        if not img_path or not os.path.exists(img_path):
            self.result_text.append("请选择有效的图片")
            return
        model_types = [get_model_type(m) for m in selected]
        self.result_text.append("正在进行多模型预测，请稍候...")
        self.btn_predict.setEnabled(False)
        self.multi_results = []
        self.multi_predict_threads = []
        self.finished_count = 0
        def on_finish(idx):
            def inner(result, error):
                self.finished_count += 1
                if error:
                    self.multi_results.append(f"模型: {selected[idx]} ({model_types[idx]})\n预测失败：{error}\n{'-'*30}")
                else:
                    self.multi_results.append(f"模型: {selected[idx]} ({model_types[idx]})\n{result.strip()}\n{'-'*30}")
                    # 写入日志
                    with open(RESULT_LOG_FILE, 'a', encoding='utf-8') as f:
                        f.write(
                            f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n"
                            f"预测模型: {selected[idx]}\n"
                            f"模型类型: {model_types[idx]}\n"
                            f"预测图片: {img_path}\n"
                            f"预测结果:\n{result}\n"
                            f"{'-'*40}\n"
                        )
                if self.finished_count == len(selected):
                    self.result_text.append("\n".join(self.multi_results))
                    self.btn_predict.setEnabled(True)
            return inner
        for idx, m in enumerate(selected):
            env = os.environ.copy()
            env["IS_SUBPROCESS"] = "1"
            cmd = f"{sys.executable} photo_predict.py --model_path \"{m}\" --model_type \"{model_types[idx]}\" --img_path \"{img_path}\""
            thread = PThread(cmd, env, capture_output=True)
            thread.finished.connect(on_finish(idx))
            self.multi_predict_threads.append(thread)
            thread.start()

class PredictVideoPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContentsMargins(20, 20, 20, 20)
        layout = QVBoxLayout(self)
        self.model_combo = QComboBox()
        self.refresh_models()
        layout.addWidget(QLabel("选择模型"))
        layout.addWidget(self.model_combo)
        self.btn_predict = QPushButton("开始视频预测")
        layout.addWidget(self.btn_predict)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        layout.addWidget(self.result_text)
        layout.addStretch()
        self.btn_predict.clicked.connect(self.predict)
        self.predict_thread = None

    def refresh_models(self):
        self.model_combo.clear()
        models = refresh_model_list()
        self.model_combo.addItems(models)

    def predict(self):
        model_path = self.model_combo.currentText()
        if not model_path:
            self.result_text.append("请先训练模型")
            return
        model_type = get_model_type(model_path)
        env = os.environ.copy()
        cmd = f"{sys.executable} video_predict.py --model_path \"{model_path}\" --model_type \"{model_type}\""
        self.result_text.append("正在视频预测，请稍候...")
        self.btn_predict.setEnabled(False)
        self.predict_thread = PThread(cmd, env, capture_output=False)
        def on_finish(result, error):
            self.btn_predict.setEnabled(True)
            if error:
                self.result_text.append(f"视频预测失败：{error}")
            else:
                self.result_text.append("视频预测已完成！")
        self.predict_thread.finished.connect(on_finish)
        self.predict_thread.start()

class DedupPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContentsMargins(20, 20, 20, 20)
        layout = QVBoxLayout(self)
        self.btn_dedup = QPushButton("自动去重数据集")
        layout.addWidget(self.btn_dedup)
        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        layout.addWidget(self.result_text)
        layout.addStretch()
        self.btn_dedup.clicked.connect(self.dedup)
        self.dedup_thread = None

    def dedup(self):
        env = os.environ.copy()
        cmd = f"{sys.executable} check_and_deduplicate_utkface.py"
        self.result_text.append("正在去重，请稍候...")
        self.btn_dedup.setEnabled(False)
        self.dedup_thread = PThread(cmd, env, capture_output=False)
        def on_finish(result, error):
            self.btn_dedup.setEnabled(True)
            if error:
                self.result_text.append(f"数据集去重失败：{error}")
            else:
                self.result_text.append("数据集去重已完成。")
        self.dedup_thread.finished.connect(on_finish)
        self.dedup_thread.start()

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

class LogPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setContentsMargins(20, 20, 20, 20)
        layout = QVBoxLayout(self)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        layout.addWidget(QLabel("结果日志"))
        layout.addWidget(self.log_text)
        self.refresh_btn = QPushButton("刷新")
        layout.addWidget(self.refresh_btn)
        layout.addStretch()
        self.refresh_btn.clicked.connect(self.refresh)
        self.refresh()

    def refresh(self):
        if os.path.exists(RESULT_LOG_FILE):
            with open(RESULT_LOG_FILE, 'r', encoding='utf-8') as f:
                self.log_text.setText(f.read())
        else:
            self.log_text.setText("暂无日志。")

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
        self.btn_models = QPushButton("查看模型")
        self.btn_predict_img = QPushButton("图片预测")
        self.btn_predict_multi_img = QPushButton("多模型图片预测")
        self.btn_predict_video = QPushButton("视频预测")
        self.btn_dedup = QPushButton("数据集去重")
        self.btn_delete = QPushButton("删除模型")
        self.btn_log = QPushButton("查看日志")
        self.btn_theme = QPushButton()
        self.btn_theme.setIcon(QIcon("assets/svg/moon.svg"))
        self.btn_theme.setIconSize(QSize(28, 28))
        self.btn_theme.setFixedSize(36, 36)
        self.btn_theme.setStyleSheet("border:none; background:transparent;")
        self.btn_theme.setToolTip("昼夜切换")
        menu_layout.addWidget(self.btn_train)
        menu_layout.addWidget(self.btn_models)
        menu_layout.addWidget(self.btn_predict_img)
        menu_layout.addWidget(self.btn_predict_multi_img)
        menu_layout.addWidget(self.btn_predict_video)
        menu_layout.addWidget(self.btn_dedup)
        menu_layout.addWidget(self.btn_delete)
        menu_layout.addWidget(self.btn_log)
        menu_layout.addStretch()
        menu_layout.addWidget(self.btn_theme)
        for btn in [
            self.btn_train, self.btn_models, self.btn_predict_img, self.btn_predict_multi_img,
            self.btn_predict_video, self.btn_dedup, self.btn_delete, self.btn_log
        ]:
            btn.setObjectName("menuButton")

        self.stack = QStackedWidget()
        self.train_panel = TrainPanel()
        self.model_list_panel = ModelListPanel()
        self.predict_img_panel = PredictImagePanel()
        self.predict_multi_img_panel = PredictMultiImagePanel()
        self.predict_video_panel = PredictVideoPanel()
        self.dedup_panel = DedupPanel()
        self.delete_panel = DeleteModelPanel()
        self.log_panel = LogPanel()
        self.stack.addWidget(self.train_panel)
        self.stack.addWidget(self.model_list_panel)
        self.stack.addWidget(self.predict_img_panel)
        self.stack.addWidget(self.predict_multi_img_panel)
        self.stack.addWidget(self.predict_video_panel)
        self.stack.addWidget(self.dedup_panel)
        self.stack.addWidget(self.delete_panel)
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
        self.btn_delete.clicked.connect(lambda: (self.delete_panel.refresh_models(), self.switch_panel(6)))
        self.btn_log.clicked.connect(lambda: (self.log_panel.refresh(), self.switch_panel(7)))
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
            elif idx == 4:
                self.predict_video_panel.result_text.clear()
            elif idx == 5:
                self.dedup_panel.result_text.clear()
            elif idx == 6:
                self.delete_panel.result_text.clear()
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

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    load_qss(app, LIGHT_QSS_FILE)
    window = MainPanelWindow()
    window.show()
    sys.exit(app.exec_())