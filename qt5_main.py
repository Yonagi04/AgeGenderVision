import sys
import os
import json
import datetime
import subprocess
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QVBoxLayout, QProgressBar, QLabel, QFileDialog,
    QMessageBox, QInputDialog, QTextEdit, QDialog, QCheckBox, QProgressDialog,
    QFormLayout, QLineEdit
)
from PyQt5.QtGui import QTextCursor
from PyQt5.QtCore import Qt, QThread, pyqtSignal

RESULT_LOG_FILE = 'result_log.log'
MODELS_INFO_FILE = 'data/models.json'
STOP_FLAG_FILE = os.path.abspath("stop.flag")

def refresh_model_list():
    return [f for f in os.listdir('.') if f.endswith('.pth')]

def get_model_type(model_path):
    if os.path.exists(MODELS_INFO_FILE):
        with open(MODELS_INFO_FILE, 'r', encoding='utf-8') as f:
            info = json.load(f)
        t = info.get(model_path)
        if t:
            return t
    types = ['resnet18', 'resnet34', 'resnet50']
    t, ok = QInputDialog.getItem(None, "模型类型", "请选择模型类型：", types, 0, False)
    return t if ok else types[0]

class PThread(QThread):
    finished = pyqtSignal(str, object)

    def __init__(self, cmd, env, capture_output=True):
        super().__init__()
        self.cmd = cmd
        self.env = env
        self.capture_output = capture_output

    def run(self):
        try:
            if self.capture_output:
                result = subprocess.run(self.cmd, shell=True, check=True, capture_output=True, text=True, env=self.env)
                self.finished.emit(result.stdout, None)
            else:
                subprocess.run(self.cmd, shell=True, check=True, env=self.env)
                self.finished.emit("", None)
        except Exception as e:
            self.finished.emit("", e)

class TrainThread(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    finished = pyqtSignal(object)

    def __init__(self, cmd, env):
        super().__init__()
        self.cmd = cmd
        self.env = env
        self._process = None
        self._stop_flag = False
    
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
                    self._stop_flag = True
                    self._process.terminate()
                    break
            self._process.wait()
            self.finished.emit(None)
        except Exception as e:
            self.finished.emit(e)

    def stop(self):
        with open("stop.flag", "w") as f:
            f.write("stop")        

class TrainDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("训练模型参数")
        self.labels = ["Batch size", "Epochs", "Learning rate", "Image size", "数据集目录", "模型类型(resnet18/resnet34/resnet50)", "模型保存路径"]
        self.defaults = [64, 10, 1e-3, 224, 'data/UTKFace/cleaned', 'resnet18', 'age_gender_multitask_resnet18.pth']
        self.edits = []
        layout = QFormLayout()
        for label, default in zip(self.labels, self.defaults):
            edit = QLineEdit(str(default))
            layout.addRow(label, edit)
            self.edits.append(edit)
        btn_ok = QPushButton("确定")
        btn_ok.clicked.connect(self.validate_and_accept)
        layout.addRow(btn_ok)
        self.setLayout(layout)
    
    def validate_and_accept(self):
        try:
            batch_size = int(self.edits[0].text())
            epoch = int(self.edits[1].text())
            lr = float(self.edits[2].text())
            img_size = int(self.edits[3].text())
            data_dir = self.edits[4].text().strip()
            model_type = self.edits[5].text().strip()
            model_path = self.edits[6].text().strip()
            if batch_size <= 0 or epoch <= 0 or lr <= 0 or img_size <= 0:
                raise ValueError("Batch size、Epochs、Learning rate、Image size 必须为正数")
            if not os.path.isdir(data_dir):
                raise ValueError("数据集目录不存在")
            if model_type not in ['resnet18', 'resnet34', 'resnet50']:
                raise ValueError("模型类型必须为 resnet18、resnet34 或 resnet50")
            if not model_path.lower().endswith('.pth'):
                model_path += '.pth'
            if not model_path:
                raise ValueError("模型保存路径不能为空")
        except Exception as e:
            msg = QMessageBox(QMessageBox.Warning, "参数错误", f"参数不合法：{e}", parent=self)
            msg.setWindowFlags(msg.windowFlags() & ~Qt.WindowContextHelpButtonHint)
            msg.exec_()
            return
        self.accept()

    def get_params(self):
        values = [e.text().strip() for e in self.edits]
        if not values[6].lower().endswith('.pth'):
            values[6] += '.pth'
        return values
    
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("年龄性别识别系统")
        self.setGeometry(200, 200, 420, 600)
        layout = QVBoxLayout()

        btn_train = QPushButton("训练模型")
        btn_train.clicked.connect(self.train_model)
        layout.addWidget(btn_train)

        btn_view = QPushButton("查看训练模型")
        btn_view.clicked.connect(self.view_models)
        layout.addWidget(btn_view)

        btn_predict = QPushButton("图片预测")
        btn_predict.clicked.connect(self.predict_image)
        layout.addWidget(btn_predict)

        btn_multi = QPushButton("多模型图片比较")
        btn_multi.clicked.connect(self.predict_image_multi)
        layout.addWidget(btn_multi)

        btn_video = QPushButton("视频预测")
        btn_video.clicked.connect(self.predict_video)
        layout.addWidget(btn_video)

        btn_dedup = QPushButton("数据集自动去重")
        btn_dedup.clicked.connect(self.deduplicate_dataset)
        layout.addWidget(btn_dedup)

        btn_delete = QPushButton("删除训练模型")
        btn_delete.clicked.connect(self.delete_model)
        layout.addWidget(btn_delete)

        btn_log = QPushButton("查看结果日志")
        btn_log.clicked.connect(self.view_log)
        layout.addWidget(btn_log)

        btn_exit = QPushButton("退出")
        btn_exit.clicked.connect(self.close)
        layout.addWidget(btn_exit)

        self.setLayout(layout)

    def train_model(self):
        dlg = TrainDialog(self)
        if dlg.exec() == QDialog.Accepted:
            if os.path.exists(STOP_FLAG_FILE):
                os.remove(STOP_FLAG_FILE)
            params = dlg.get_params()
            if not params:
                return
            cmd = (f"{sys.executable} train_age_gender_multitask.py "
                f"--batch_size {params[0]} --epochs {params[1]} --lr {params[2]} "
                f"--img_size {params[3]} --data_dir \"{params[4]}\" --model_type \"{params[5]}\" --model_path \"{params[6]}\"")
            env = os.environ.copy()
            log_dlg = QDialog(self)
            log_dlg.setWindowTitle("训练进度")
            log_dlg.setWindowFlags(log_dlg.windowFlags() & ~Qt.WindowContextHelpButtonHint)
            vbox = QVBoxLayout()
            progress_bar = QProgressBar()
            progress_bar.setRange(0, 100)
            vbox.addWidget(progress_bar)
            tqdm_label = QLabel()
            tqdm_label.setText(" ")
            vbox.addWidget(tqdm_label)
            log_text = QTextEdit()
            log_text.setReadOnly(True)
            vbox.addWidget(log_text)
            btn_stop = QPushButton("停止训练")
            vbox.addWidget(btn_stop)
            log_dlg.setLayout(vbox)
            self.train_thread = TrainThread(cmd, env)
            def log_filter(text):
                if "|" in text and "%" in text:
                    tqdm_label.setText(text.rstrip())
                else:
                    log_text.append(text.rstrip())
                    log_text.moveCursor(QTextCursor.End)
            self.train_thread.log_signal.connect(log_filter)
            self.train_thread.progress_signal.connect(progress_bar.setValue)
            def on_finish(error):
                btn_stop.setEnabled(False)
                if error:
                    msg = QMessageBox(QMessageBox.Critical, "训练失败", f"模型训练失败：{error}", parent=self)
                    msg.setWindowFlags(msg.windowFlags() & ~Qt.WindowContextHelpButtonHint)
                    msg.exec_()
                else:
                    if os.path.exists(params[6]):
                        try:
                            if os.path.exists(MODELS_INFO_FILE):
                                with open(MODELS_INFO_FILE, 'r', encoding='utf-8') as f:
                                    info = json.load(f)
                            else:
                                info = {}
                            info[params[6]] = params[5]
                            with open(MODELS_INFO_FILE, 'w', encoding='utf-8') as f:
                                json.dump(info, f, ensure_ascii=False, indent=2)
                        except Exception as e:
                            msg = QMessageBox(QMessageBox.Warning, "模型信息写入失败", str(e), parent=self)
                            msg.setWindowFlags(msg.windowFlags() & ~Qt.WindowContextHelpButtonHint)
                            msg.exec_()
                        msg = QMessageBox(QMessageBox.Information, "训练完成", "模型训练完成！", parent=self)
                        msg.setWindowFlags(msg.windowFlags() & ~Qt.WindowContextHelpButtonHint)
                        msg.exec_()
                    else:
                        msg = QMessageBox(QMessageBox.Critical, "训练中断", "模型保存失败！请启动命令行开发者模式，排查原因。", parent=self)
                        msg.setWindowFlags(msg.windowFlags() & ~Qt.WindowContextHelpButtonHint)
                        msg.exec_()
                log_dlg.accept()
            self.train_thread.finished.connect(on_finish)
            btn_stop.clicked.connect(self.train_thread.stop)
            self.train_thread.start()
            log_dlg.exec_()
    
    def view_models(self):
        models = refresh_model_list()
        dlg = QDialog(self)
        dlg.setWindowTitle("已训练模型列表")
        dlg.setWindowFlags(dlg.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        vbox = QVBoxLayout()
        if not models:
            vbox.addWidget(QLabel("暂无模型文件"))
        else:
            for m in models:
                vbox.addWidget(QLabel(m))
        dlg.setLayout(vbox)
        dlg.exec_()

    def predict_image(self):
        model_path = self.select_model()
        if not model_path:
            return
        model_type = get_model_type(model_path)
        img_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)")
        if not img_path:
            return
        env = os.environ.copy()
        env["IS_SUBPROCESS"] = "1"
        cmd = f"{sys.executable} photo_predict.py --model_path \"{model_path}\" --model_type \"{model_type}\" --img_path \"{img_path}\""
        progress = QProgressDialog("正在预测，请稍候...", None, 0, 0, self)
        progress.setWindowFlags(progress.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        progress.setWindowTitle("提示")
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setCancelButton(None)
        progress.show()
        QApplication.processEvents()
        self.predict_thread = PThread(cmd, env, capture_output=True)
        def on_finish(result, error):
            progress.close()
            if error:
                msg = QMessageBox(QMessageBox.Critical, "错误", f"图片预测失败：{error}", parent=self)
                msg.setWindowFlags(msg.windowFlags() & ~Qt.WindowContextHelpButtonHint)
                msg.exec_()
            else:
                msg = QMessageBox(QMessageBox.Information, "预测完成", f"图片预测已完成！\n\n{result.strip()}", parent=self)
                msg.setWindowFlags(msg.windowFlags() & ~Qt.WindowContextHelpButtonHint)
                msg.exec_()
        self.predict_thread.finished.connect(on_finish)
        self.predict_thread.start()

    def predict_image_multi(self):
        models = refresh_model_list()
        if not models:
            msg = QMessageBox(QMessageBox.Information, "提示", "请先训练模型", parent=self)
            msg.setWindowFlags(msg.windowFlags() & ~Qt.WindowContextHelpButtonHint)
            msg.exec_()
            return
        dlg = QDialog(self)
        dlg.setWindowFlags(dlg.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        dlg.setWindowTitle("选择多个模型")
        vbox = QVBoxLayout()
        check = []
        for m in models:
            cb = QCheckBox(m)
            vbox.addWidget(cb)
            check.append(cb)
        btn_ok = QPushButton("开始比较")
        vbox.addWidget(btn_ok)
        dlg.setLayout(vbox)
        def on_ok():
            selected = [cb.text() for cb in check if cb.isChecked()]
            if not selected:
                msg = QMessageBox(QMessageBox.Information, "提示", "请至少选择一个模型进行比较", parent=self)
                msg.setWindowFlags(msg.windowFlags() & ~Qt.WindowContextHelpButtonHint)
                msg.exec_()
                return
            if len(selected) > len(models):
                msg = QMessageBox(QMessageBox.Warning, "提示", f"最多只能选择{len(models)}个模型进行比较", parent=self)
                msg.setWindowFlags(msg.windowFlags() & ~Qt.WindowContextHelpButtonHint)
                msg.exec_()
                return
            dlg.accept()
            img_path, _ = QFileDialog.getOpenFileName(self, "选择图片", "", "Images (*.png *.jpg *.jpeg *.bmp *.tiff *.webp)")
            if not img_path:
                return
            model_types = [get_model_type(m) for m in selected]
            progress = QProgressDialog("正在进行多模型预测，请稍候...", None, 0, len(selected), self)
            progress.setWindowFlags(progress.windowFlags() & ~Qt.WindowContextHelpButtonHint)
            progress.setWindowTitle("提示")
            progress.setWindowModality(Qt.ApplicationModal)
            progress.setCancelButton(None)
            progress.show()
            QApplication.processEvents()
            self.multi_results = []
            self.multi_errors = []
            self.multi_predict_threads = []
            def run_next(idx = 0):
                if idx >= len(selected):
                    progress.close()
                    all_results = ""
                    for i, m in enumerate(selected):
                        out = self.multi_results[i] if i < len(self.multi_results) else ""
                        all_results += f"模型: {m} ({model_types[i]})\n{out.strip()}\n{'-'*30}\n"
                    msg = QMessageBox(QMessageBox.Information, "多模型预测完成", f"多模型预测已完成，详细结果如下：\n\n{all_results.strip()}\n\n详细结果请查看 {RESULT_LOG_FILE}", parent=self)
                    msg.setWindowFlags(msg.windowFlags() & ~Qt.WindowContextHelpButtonHint)
                    msg.exec_()
                    return
                m = selected[idx]
                env = os.environ.copy()
                env["IS_SUBPROCESS"] = "1"
                cmd = f"{sys.executable} photo_predict.py --model_path \"{m}\" --model_type \"{model_types[idx]}\" --img_path \"{img_path}\""
                predict_thread = PThread(cmd, env, capture_output=True)
                def on_finish(result, error):
                    progress.setValue(idx + 1)
                    if error:
                        self.multi_results.append(f"预测失败：{error}")
                    else:
                        self.multi_results.append(result)
                        with open(RESULT_LOG_FILE, 'a', encoding='utf-8') as f:
                            f.write(
                                f"\n[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}]\n"
                                f"预测模型: {m}\n"
                                f"模型类型: {model_types[idx]}\n"
                                f"预测图片: {img_path}\n"
                                f"预测结果:\n{result}\n"
                                f"{'-'*40}\n"
                            )
                    run_next(idx + 1)
                self.multi_predict_threads.append(predict_thread)
                predict_thread.finished.connect(on_finish)
                predict_thread.start()
            run_next(0)
        btn_ok.clicked.connect(on_ok)
        dlg.exec_()

    def predict_video(self):
        model_path = self.select_model()
        if not model_path:
            return
        model_type = get_model_type(model_path)
        env = os.environ.copy()
        cmd = f"{sys.executable} video_predict.py --model_path \"{model_path}\" --model_type \"{model_type}\""
        progress = QProgressDialog("正在视频预测，请稍候...", None, 0, 0, self)
        progress.setWindowFlags(progress.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        progress.setWindowTitle("提示")
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setCancelButton(None)
        progress.show()
        QApplication.processEvents()
        self.predict_thread = PThread(cmd, env, capture_output=False)
        def on_finish(result, error):
            progress.close()
            if error:
                msg = QMessageBox(QMessageBox.Critical, "错误", f"视频预测失败：{error}", parent=self)
                msg.setWindowFlags(msg.windowFlags() & ~Qt.WindowContextHelpButtonHint)
                msg.exec_()
            else:
                msg = QMessageBox(QMessageBox.Information, "完成", "视频预测已完成！", parent=self)
                msg.setWindowFlags(msg.windowFlags() & ~Qt.WindowContextHelpButtonHint)
                msg.exec_()
        self.predict_thread.finished.connect(on_finish)
        self.predict_thread.start()
            
    def deduplicate_dataset(self):
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle("确认去重")
        msg.setText("是否自动去重数据集？\n注意：此操作会删除重复图片，且不可恢复！")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setWindowFlags(msg.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        reply = msg.exec_()
        if reply == QMessageBox.Yes:
            env = os.environ.copy()
            cmd = f"{sys.executable} check_and_deduplicate_utkface.py"
            progress = QProgressDialog("正在去重，请稍候...", None, 0, 0, self)
            progress.setWindowFlags(progress.windowFlags() & ~Qt.WindowContextHelpButtonHint)
            progress.setWindowTitle("提示")
            progress.setWindowModality(Qt.ApplicationModal)
            progress.setCancelButton(None)
            progress.show()
            QApplication.processEvents()
            self.dedup_thread = PThread(cmd, env, capture_output=False)
            def on_finish(result, error):
                progress.close()
                if error:
                    msg = QMessageBox(QMessageBox.Critical, "错误", f"数据集去重失败：{error}", parent=self)
                    msg.setWindowFlags(msg.windowFlags() & ~Qt.WindowContextHelpButtonHint)
                    msg.exec_()
                else:
                    msg = QMessageBox(QMessageBox.Information, "完成", "数据集去重已完成。", parent=self)
                    msg.setWindowFlags(msg.windowFlags() & ~Qt.WindowContextHelpButtonHint)
                    msg.exec_()
            self.dedup_thread.finished.connect(on_finish)
            self.dedup_thread.start()

    def delete_model(self):
        models = refresh_model_list()
        if not models:
            msg = QMessageBox(QMessageBox.Information, "提示", "暂无模型文件", parent=self)
            msg.setWindowFlags(msg.windowFlags() & ~Qt.WindowContextHelpButtonHint)
            msg.exec_()
            return
        model, ok = QInputDialog.getItem(self, "删除模型", "请选择要删除的模型：", models, 0, False)
        if not ok or not model:
            return
        confirm, ok = QInputDialog.getText(self, "确认删除", f"请再次输入模型文件名以确认删除：")
        if not ok or confirm != model:
            msg = QMessageBox(QMessageBox.Information, "提示", "模型名称输入错误，未删除任何文件。", parent=self)
            msg.setWindowFlags(msg.windowFlags() & ~Qt.WindowContextHelpButtonHint)
            msg.exec_()
            return
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle("确认")
        msg.setText(f"即将永久删除模型 {model}，确定删除？")
        msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msg.setWindowFlags(msg.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        reply = msg.exec_()
        if reply == QMessageBox.Yes:
            os.remove(model)
            try:
                if os.path.exists(MODELS_INFO_FILE):
                    with open(MODELS_INFO_FILE, 'r', encoding='utf-8') as f:
                        info = json.load(f)
                    if model in info:
                        del info[model]
                        with open(MODELS_INFO_FILE, 'w', encoding='utf-8') as f:
                            json.dump(info, f, ensure_ascii=False, indent=2)
            except Exception as e:
                msg = QMessageBox(QMessageBox.Warning, "模型信息更新失败", str(e), parent=self)
                msg.setWindowFlags(msg.windowFlags() & ~Qt.WindowContextHelpButtonHint)
                msg.exec_()
            msg = QMessageBox(QMessageBox.Information, "完成", f"模型 {model} 已成功删除。", parent=self)
            msg.setWindowFlags(msg.windowFlags() & ~Qt.WindowContextHelpButtonHint)
            msg.exec_()
        
    def view_log(self):
        dlg = QDialog(self)
        dlg.setWindowFlags(dlg.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        dlg.setWindowTitle("结果日志")
        vbox = QVBoxLayout()
        txt = QTextEdit()
        txt.setReadOnly(True)
        if os.path.exists(RESULT_LOG_FILE):
            with open(RESULT_LOG_FILE, 'r', encoding='utf-8') as f:
                txt.setText(f.read())
        else:
            txt.setText("暂无日志。")
        vbox.addWidget(txt)
        dlg.setLayout(vbox)
        dlg.resize(800, 600)
        dlg.exec_()

    def select_model(self):
        models = refresh_model_list()
        if not models:
            msg = QMessageBox(QMessageBox.Information, "提示", "暂无模型文件", parent=self)
            msg.setWindowFlags(msg.windowFlags() & ~Qt.WindowContextHelpButtonHint)
            msg.exec_()
            return None
        model, ok = QInputDialog.getItem(self, "选择模型", "请选择模型：", models, 0, False)
        return model if ok else None
    
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())