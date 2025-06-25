from PyQt5.QtWidgets import (
    QWidget, QPushButton, QVBoxLayout, QLabel, QScrollArea, QGroupBox, 
    QHBoxLayout, QLineEdit, QFileDialog, QMessageBox, QMenu, QDialog
)
from PyQt5.QtCore import QUrl, QPropertyAnimation, QEasingCurve, QSize, Qt, QTimer
from PyQt5.QtGui import QDesktopServices, QIcon
import json
import csv
import os
import shutil
import zipfile
from threads.model_import_thread import ModelImportThread
from widgets.message_box import MessageBox
from widgets.input_dialog import InputDialog
from widgets.select_dialog import SelectDialog
from services.model_service import ModelService
from convention.result_code import ResultCode

MODELS_INFO_FILE = os.path.join("data", "models.json")

class ModelListPanel(QWidget):
    def __init__(self, parent=None, theme='light'):
        super().__init__(parent)
        self.theme = theme
        self.setContentsMargins(20, 20, 20, 20)
        self.layout = QVBoxLayout(self)

        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.inner = QWidget()
        self.inner_layout = QVBoxLayout(self.inner)
        self.scroll.setWidget(self.inner)
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("输入模型名称、类型、创建时间、备注以搜索")
        self.btn_search = QPushButton("搜索")

        self.btn_refresh = QPushButton()
        if self.theme == 'light':
            self.btn_refresh.setIcon(QIcon("assets/svg/refresh_light.svg"))
        else:
            self.btn_refresh.setIcon(QIcon("assets/svg/refresh_dark.svg"))
        self.btn_refresh.setIconSize(QSize(28, 28))
        self.btn_refresh.setFixedSize(36, 36)
        self.btn_refresh.setStyleSheet("border:none; background:transparent;")

        self.btn_download = QPushButton()
        if self.theme == 'light':
            self.btn_download.setIcon(QIcon("assets/svg/download_light.svg"))
        else:
            self.btn_download.setIcon(QIcon("assets/svg/download_dark.svg"))
        self.btn_download.setIconSize(QSize(22, 22))
        self.btn_download.setFixedSize(36, 36)
        self.btn_download.setStyleSheet("border:none; background:transparent;")

        self.btn_upload = QPushButton()
        if self.theme == 'light':
            self.btn_upload.setIcon(QIcon("assets/svg/upload_light.svg"))
        else:
            self.btn_upload.setIcon(QIcon("assets/svg/upload_dark.svg"))
        self.btn_upload.setIconSize(QSize(20, 20))
        self.btn_upload.setFixedSize(36, 36)
        self.btn_upload.setStyleSheet("border:none; background:transparent;")

        self.hbox = QHBoxLayout()
        self.hbox.addWidget(self.search_box)
        self.hbox.addWidget(self.btn_search)
        self.hbox.addWidget(self.btn_refresh)
        self.hbox.addWidget(self.btn_upload)
        self.hbox.addWidget(self.btn_download)
        self.layout.addLayout(self.hbox)
        self.layout.addWidget(self.scroll)

        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_search.clicked.connect(self.search)
        self.btn_download.clicked.connect(self.download)
        self.btn_upload.clicked.connect(self.upload)

        self._pending_delete = None

    def get_current_theme(self):
        theme = "light"
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, "current_theme"):
                theme = parent.current_theme
                break
            parent = parent.parent()
        return theme

    def refresh(self):
        self.theme = self.get_current_theme()
        for i in reversed(range(self.inner_layout.count())):
            widget = self.inner_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        model_info = ModelService.load_model_info()
        if not model_info.success:
            msg_box = MessageBox(
                parent=self,
                title="错误",
                text=model_info.message,
                icon=QMessageBox.Critical,
                theme=self.theme
            )
            msg_box.exec_()
            return
        if model_info.code == ResultCode.NO_DATA:
            msg_box = MessageBox(
                parent=self,
                title="提示",
                text=model_info.message,
                icon=QMessageBox.Information,
                theme = self.theme
            )
            msg_box.exec_()
            return
        
        for model_name, meta in model_info.data.items():
            model_type = meta.get("model_type", "未知")
            model_dir = meta.get("model_dir", "未知")
            created_time = meta.get("created_time", "未知")
            update_time = meta.get("update_time", "未知")
            description = meta.get("description", "未知")
            group = QGroupBox()
            vbox = QVBoxLayout(group)
            vbox.addWidget(QLabel(f"模型名称: {model_name}"))
            vbox.addWidget(QLabel(f"模型类型: {model_type}"))
            vbox.addWidget(QLabel(f"存放位置: {model_dir}"))
            vbox.addWidget(QLabel(f"创建时间: {created_time}"))
            vbox.addWidget(QLabel(f"更新时间: {update_time}"))
            vbox.addWidget(QLabel(f"备注: {description}"))
            hbox = QHBoxLayout()
            btn_open = QPushButton("打开目录")
            btn_open.clicked.connect(lambda _, d=model_dir: QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(d))))
            hbox.addWidget(btn_open)
            btn_meta = QPushButton("查看模型元信息")
            meta_json_path = os.path.abspath(os.path.join(model_dir, 'meta.json'))
            if not os.path.exists(meta_json_path):
                btn_meta.setEnabled(False)
                btn_meta.setToolTip("没有模型元信息")
            else:
                btn_meta.setEnabled(True)
            btn_meta.clicked.connect(lambda _, d=model_dir: QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(os.path.join(d, 'meta.json')))))
            hbox.addWidget(btn_meta)
            btn_output = QPushButton("导出模型包")
            if model_dir == '.':
                btn_output.setEnabled(False)
                btn_output.setToolTip("无法导出，因为模型存储在主文件夹")
            else:
                btn_output.setEnabled(True)
            btn_output.clicked.connect(lambda _, m=model_name, d=model_dir: self.output_model(m, d))
            hbox.addWidget(btn_output)
            btn_more = QPushButton("更多")
            menu = QMenu()
            menu.addAction("重命名模型", lambda m=model_name: self.rename_model(m))
            menu.addAction("设置备注", lambda m=model_name,d=description: self.set_description(m, d))
            menu.addAction("修改模型类型", lambda m=model_name, t=model_type: self.update_model_type(m, t))
            menu.addAction("删除模型", lambda m=model_name, d=model_dir: self.delete_model(m, d))
            btn_more.setMenu(menu)
            hbox.addWidget(btn_more)

            vbox.addLayout(hbox)
            self.inner_layout.addWidget(group)
        self.inner_layout.addStretch()

    def search(self):
        self.theme = self.get_current_theme()
        keyword = self.search_box.text().strip().lower()
        if not keyword:
            self.refresh()
            return
        
        for i in reversed(range(self.inner_layout.count())):
            widget = self.inner_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        model_info = ModelService.search_model(keyword)
        if not model_info.success:
            msg_box = MessageBox(
                parent=self,
                title="错误",
                text=model_info.message,
                icon=QMessageBox.critical,
                theme=self.theme
            )
            msg_box.exec_()
            return
        if model_info.code == ResultCode.NO_DATA:
            msg_box = MessageBox(
                parent=self,
                title="提示",
                text=model_info.message,
                icon=QMessageBox.Information,
                theme=self.theme
            )
            msg_box.exec_()
            self.refresh()
            return
        for model_name, meta in model_info.data.items():
            model_type = meta.get("model_type", "未知")
            created_time = meta.get("created_time", "未知")
            update_time = meta.get("update_time", "未知")
            model_dir = meta.get("model_dir", "未知")
            description = meta.get("description", "未知")
            
            group = QGroupBox()
            vbox = QVBoxLayout(group)
            vbox.addWidget(QLabel(f"模型名称: {model_name}"))
            vbox.addWidget(QLabel(f"模型类型: {model_type}"))
            vbox.addWidget(QLabel(f"存放位置: {model_dir}"))
            vbox.addWidget(QLabel(f"创建时间: {created_time}"))
            vbox.addWidget(QLabel(f"修改时间: {update_time}"))
            vbox.addWidget(QLabel(f"备注: {description}"))

            hbox = QHBoxLayout()
            btn_open = QPushButton("打开目录")
            btn_open.clicked.connect(lambda _, d=model_dir: QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(d))))
            hbox.addWidget(btn_open)
            btn_meta = QPushButton("查看模型元信息")
            meta_json_path = os.path.abspath(os.path.join(model_dir, 'meta.json'))
            if not os.path.exists(meta_json_path):
                btn_meta.setEnabled(False)
                btn_meta.setToolTip("没有模型元信息")
            else:
                btn_meta.setEnabled(True)
            btn_meta.clicked.connect(lambda _, d=model_dir: QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(os.path.join(d, 'meta.json')))))
            hbox.addWidget(btn_meta)
            btn_output = QPushButton("导出模型包")
            if model_dir == '.':
                btn_output.setEnabled(False)
                btn_output.setToolTip("无法导出，因为模型存储在主文件夹")
            else:
                btn_output.setEnabled(True)
            btn_output.clicked.connect(lambda m=model_name, d=model_dir: self.output_model(m, d))
            hbox.addWidget(btn_output)
            btn_more = QPushButton("更多")
            menu = QMenu()
            menu.addAction("重命名模型", lambda m=model_name: self.rename_model(m))
            menu.addAction("设置备注", lambda m=model_name, d=description: self.set_description(m, d))
            menu.addAction("删除模型", lambda m=model_name, d=model_dir: self.delete_model(m, d))
            btn_more.setMenu(menu)
            hbox.addWidget(btn_more)
            vbox.addLayout(hbox)
            self.inner_layout.addWidget(group)
        self.inner_layout.addStretch()
    
    def download(self):
        try:
            self.theme = self.get_current_theme()
            ask_box = MessageBox(
                parent=self,
                title="请选择保存格式",
                text="请选择保存格式（CSV / JSON）",
                icon=QMessageBox.Information,
                theme=self.theme,
                addButton=False
            )
            ask_box.add_buttons({
                "CSV": QMessageBox.AcceptRole,
                "JSON": QMessageBox.AcceptRole,
                "取消": QMessageBox.RejectRole
            })
            ask_box.exec_()
            user_choice = ask_box.get_clicked_button()
            if user_choice not in ['CSV', 'JSON']:
                return
            result = ModelService.download_model_info(user_choice)
            if not result.success:
                msg_box = MessageBox(
                    parent=self,
                    title="错误",
                    text=result.message,
                    icon=QMessageBox.Critical,
                    theme=self.theme
                )
                msg_box.exec_()
                return
            if result.code == ResultCode.NO_DATA:
                msg_box = MessageBox(
                    parent=self,
                    text=result.message,
                    theme=self.theme
                )
                msg_box.exec_()
                return
            export_path = result.data
            if user_choice == 'CSV':
                msg_box = MessageBox(
                    parent=self,
                    title="下载模型信息成功",
                    text="模型信息已导出为 CSV 格式",
                    icon=QMessageBox.Information,
                    theme=self.theme
                )
                msg_box.exec_()
                QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(export_path)))
            else:
                msg_box = MessageBox(
                    parent=self,
                    title="下载模型信息成功",
                    text="模型信息已导出为 JSON 格式",
                    icon=QMessageBox.Information,
                    theme=self.theme
                )
                msg_box.exec_()
                QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(export_path)))
        except Exception as e:
            msg_box = MessageBox(
                parent=self,
                title="错误",
                text= f"导出过程出错: {e}",
                icon=QMessageBox.Critical,
                theme=self.theme
            )
            msg_box.exec_()
            
    def upload(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型进行上传", "", "PyTorch 模型文件 (*.pth *.pt)")
        if not path:
            return
        allowed_types = ['resnet18', 'resnet34', 'resnet50']
        self.btn_upload.setEnabled(False)
        self.thread = ModelImportThread(path, allowed_types)
        self.thread.finished.connect(self.on_model_import_finished)
        self.thread.start()

    def on_model_import_finished(self, success, msg):
        self.theme = self.get_current_theme()
        self.btn_upload.setEnabled(True)
        if success:
            self.refresh()
        self.theme = self.get_current_theme()
        msg_box = MessageBox(
            parent=self,
            title="提示" if success else "错误",
            text=msg,
            icon=QMessageBox.Information if success else QMessageBox.Critical,
            theme=self.theme
        )
        msg_box.exec_()

    def output_model(self, model_name, model_dir):
        self.theme = self.get_current_theme()
        result = ModelService.output_model(model_name, model_dir)
        if not result.success:
            msg_box = MessageBox(
                parent=self,
                title="错误",
                text= result.message,
                theme=self.theme,
                icon=QMessageBox.Critical
            )
            msg_box.exec_()
        zip_path = result.data
        msg_box = MessageBox(
            parent=self,
            title="提示",
            text="导出成功！",
            theme=self.theme,
            icon=QMessageBox.Information
        )
        msg_box.exec_()
        QDesktopServices.openUrl(QUrl.fromLocalFile(zip_path))
        
    def rename_model(self, old_model_name):
        self.theme = self.get_current_theme()
        dialog = InputDialog(
            self, 
            title="重命名模型", 
            text="请输入新的模型名称：", 
            default_text=old_model_name, 
            theme=self.theme
        )
        if dialog.exec_() != QDialog.Accepted:
            return
        new_model_name = dialog.get_value()
        if not new_model_name:
            msg_box = MessageBox(
                parent = self,
                text = "模型名称为空",
                theme = self.theme
            )
            msg_box.exec_()
            return
        if new_model_name == old_model_name:
            msg_box = MessageBox(
                parent = self,
                text = "模型名称未修改",
                theme = self.theme
            )
            msg_box.exec_()
            return
        if not new_model_name.endswith('.pth'):
            new_model_name += '.pth'
        result = ModelService.rename_model(old_model_name, new_model_name)
        if not result.success:
            msg_box = MessageBox(
                parent=self,
                title="错误",
                text=result.message,
                icon=QMessageBox.Critical,
                theme=self.theme
            )
            msg_box.exec_()
            return
        msg_box = MessageBox(
            parent=self,
            text=result.message,
            theme=self.theme
        )
        msg_box.exec_()
        self.refresh()

    def set_description(self, model_name, description):
        self.theme = self.get_current_theme()
        dialog = InputDialog(
            self,
            title='设置备注',
            text='请设置备注',
            default_text=description,
            theme=self.theme
        )
        if dialog.exec_() != QDialog.Accepted:
            return
        description = dialog.get_value()
        result = ModelService.set_model_description(model_name, description)
        if not result.success:
            msg_box = MessageBox(
                parent=self,
                title="错误",
                text=result.message,
                icon=QMessageBox.Critical,
                theme=self.theme
            )
            msg_box.exec_()
            return
        msg_box = MessageBox(
            parent=self,
            title="提示",
            text=result.message,
            icon=QMessageBox.Information,
            theme=self.theme
        )
        msg_box.exec_()
        self.refresh()

    def update_model_type(self, model_name, old_model_type):
        self.theme = self.get_current_theme()
        type_options = ['resnet18', 'resnet34', 'resnet50']
        dialog = SelectDialog(
            parent=self,
            title="选择",
            text="请选择新的模型：",
            options=type_options,
            default_index=type_options.index(old_model_type),
            theme=self.theme
        )
        if dialog.exec_() != QDialog.Accepted:
            return
        new_model_type = dialog.get_value()
        if new_model_type == old_model_type:
            return
        msg_box = MessageBox(
            parent=self,
            text=f"即将更改模型类型，原模型类型为：{old_model_type}，新模型类型为：{new_model_type}。修改模型类型后可能会导致模型无法加载，确定要继续吗？",
            theme=self.theme,
            addButton=False
        )
        msg_box.add_buttons({
            "确定": QMessageBox.AcceptRole,
            "取消": QMessageBox.RejectRole
        })
        msg_box.exec_()
        if msg_box.get_clicked_button() == "取消":
            return
        result = ModelService.update_model_type(model_name, new_model_type)
        if not result.success:
            msg_box = MessageBox(
                parent=self,
                title="错误",
                text=result.message,
                icon=QMessageBox.Critical,
                theme=self.theme
            )
            msg_box.exec_()
            return
        msg_box = MessageBox(
            parent=self,
            title="提示",
            text=result.message,
            icon=QMessageBox.Information,
            theme=self.theme
        )
        msg_box.exec_()
        self.refresh()        

    def delete_model(self, model_name, model_dir):
        self.theme = self.get_current_theme()
        dialog = InputDialog(
            self, 
            title="删除模型", 
            text=f"请输入需要删除模型的名称: {model_name}",  
            theme=self.theme
        )
        if dialog.exec_() != QDialog.Accepted:
            return
        delete_name = dialog.get_value()
        if delete_name != model_name:
            msg_box = MessageBox(
                parent=self,
                title="提示",
                text="输入的模型名称错误，未执行删除",
                theme=self.theme
            )
            msg_box.exec_()
            return
        result = ModelService.delete_model(model_name, model_dir)
        if not result.success:
            msg_box = MessageBox(
                parent=self,
                title="错误",
                text=result.message,
                icon=QMessageBox.Critical,
                theme=self.theme
            )
            msg_box.exec_()
            return
        msg_box = MessageBox(
            parent=self,
            title="提示",
            text="模型删除成功",
            icon=QMessageBox.Information,
            theme=self.theme
        )
        msg_box.exec_()
        self.refresh()