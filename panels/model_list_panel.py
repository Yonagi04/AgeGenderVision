from PyQt5.QtWidgets import (
    QWidget, QPushButton, QVBoxLayout, QLabel, QScrollArea, QGroupBox, QHBoxLayout, QLineEdit, QFileDialog, QMessageBox
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

MODELS_INFO_FILE = os.path.join("data", "models.json")

class ModelListPanel(QWidget):
    def __init__(self, parent=None, theme='light'):
        super().__init__(parent)
        self.theme = theme
        self.setContentsMargins(20, 20, 20, 20)
        self.layout = QVBoxLayout(self)

        self.delete_overlay = QWidget(self)
        self.delete_overlay.setObjectName("deleteConfirmOverlay")
        overlay_layout = QHBoxLayout(self.delete_overlay)
        self.delete_label = QLabel("")
        self.delete_input = QLineEdit()
        self.delete_input.setPlaceholderText("请输入模型名称以确认删除")
        self.btn_confirm = QPushButton("确认删除")
        self.btn_cancel = QPushButton("取消")
        overlay_layout.addWidget(self.delete_label)
        overlay_layout.addWidget(self.delete_input)
        overlay_layout.addWidget(self.btn_confirm)
        overlay_layout.addWidget(self.btn_cancel)
        self.delete_overlay.hide()
        self.layout.addWidget(self.delete_overlay)

        self.scroll = QScrollArea(self)
        self.scroll.setWidgetResizable(True)
        self.inner = QWidget()
        self.inner_layout = QVBoxLayout(self.inner)
        self.scroll.setWidget(self.inner)
        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("输入模型名称、类型、创建时间以搜索")
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

        self.btn_confirm.clicked.connect(self._do_delete_model)
        self.btn_cancel.clicked.connect(self._hide_delete_overlay)
        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_search.clicked.connect(self.search)
        self.btn_download.clicked.connect(self.download)
        self.btn_upload.clicked.connect(self.upload)

        self._pending_delete = None

    def refresh(self):
        self.theme = self.get_current_theme()
        for i in reversed(range(self.inner_layout.count())):
            widget = self.inner_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()
        if not os.path.exists(MODELS_INFO_FILE):
            msg_box = MessageBox(
                parent=self,
                title="提示",
                text="暂无模型文件",
                icon=QMessageBox.Information,
                theme=self.theme
            )
            msg_box.exec_()
            return
        with open(MODELS_INFO_FILE, "r", encoding="utf-8") as f:
            info = json.load(f)
        if not info:
            msg_box = MessageBox(
                parent=self,
                title="提示",
                text="暂无模型文件",
                icon=QMessageBox.Information,
                theme=self.theme
            )
            msg_box.exec_()
            return
        for model_name, meta in info.items():
            model_type = meta.get("model_type", "未知")
            model_dir = meta.get("model_dir", "未知")
            created_time = meta.get("created_time", "未知")
            group = QGroupBox()
            vbox = QVBoxLayout(group)
            vbox.addWidget(QLabel(f"模型名称: {model_name}"))
            vbox.addWidget(QLabel(f"模型类型: {model_type}"))
            vbox.addWidget(QLabel(f"存放位置: {model_dir}"))
            vbox.addWidget(QLabel(f"创建时间: {created_time}"))
            hbox = QHBoxLayout()
            btn_open = QPushButton("打开目录")
            btn_open.clicked.connect(lambda _, d=model_dir: QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(d))))
            hbox.addWidget(btn_open)
            btn_delete = QPushButton("删除模型")
            btn_delete.clicked.connect(lambda _, m=model_name, d=model_dir: self.show_delete_overlay(m, d))
            hbox.addWidget(btn_delete)
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

            vbox.addLayout(hbox)
            self.inner_layout.addWidget(group)
        self.inner_layout.addStretch()

    def get_current_theme(self):
        theme = "light"
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, "current_theme"):
                theme = parent.current_theme
                break
            parent = parent.parent()
        return theme

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

        if not os.path.exists(MODELS_INFO_FILE):
            msg_box = MessageBox(
                parent=self,
                title="提示",
                text="暂无模型文件",
                icon=QMessageBox.Information,
                theme=self.theme
            )
            msg_box.exec_()
            return
        
        with open(MODELS_INFO_FILE, 'r', encoding='utf-8') as f:
            info = json.load(f)
        if not info:
            msg_box = MessageBox(
                parent=self,
                title="提示",
                text="暂无模型文件",
                icon=QMessageBox.Information,
                theme=self.theme
            )
            msg_box.exec_()
            return

        found = False
        for model_name, meta in info.items():
            model_type = meta.get("model_type", "未知")
            created_time = meta.get("created_time", "未知")
            model_dir = meta.get("model_dir", "未知")

            match = (
                keyword in model_name.lower() or
                keyword in model_type.lower() or
                keyword in created_time.lower()
            )
            if not match:
                continue
            
            found = True
            group = QGroupBox()
            vbox = QVBoxLayout(group)
            vbox.addWidget(QLabel(f"模型名称: {model_name}"))
            vbox.addWidget(QLabel(f"模型类型: {model_type}"))
            vbox.addWidget(QLabel(f"存放位置: {model_dir}"))
            vbox.addWidget(QLabel(f"创建时间: {created_time}"))

            hbox = QHBoxLayout()
            btn_open = QPushButton("打开目录")
            btn_open.clicked.connect(lambda _, d=model_dir: QDesktopServices.openUrl(QUrl.fromLocalFile(os.path.abspath(d))))
            hbox.addWidget(btn_open)
            btn_delete = QPushButton("删除模型")
            btn_delete.clicked.connect(lambda _, m=model_name, d=model_dir: self.show_delete_overlay(m, d))
            hbox.addWidget(btn_delete)
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
            vbox.addLayout(hbox)
            self.inner_layout.addWidget(group)
        if not found:
            msg_box = MessageBox(
                parent=self,
                title="提示",
                text="未找到匹配的模型",
                icon=QMessageBox.Information,
                theme=self.theme
            )
            msg_box.exec()
            self.refresh()
        self.inner_layout.addStretch()
    
    def download(self):
        try:
            self.theme = self.get_current_theme()
            if not os.path.exists(MODELS_INFO_FILE):
                msg_box = MessageBox(
                    parent=self,
                    title="提示",
                    text="暂无模型信息",
                    icon=QMessageBox.Information,
                    theme=self.theme
                )
                msg_box.exec_()
                return
            with open(MODELS_INFO_FILE, 'r', encoding='utf-8') as f:
                info = json.load(f)
            if not info:
                msg_box = MessageBox(
                    parent=self,
                    title="提示",
                    text="暂无模型信息",
                    icon=QMessageBox.Information,
                    theme=self.theme
                )
                msg_box.exec_()
                return
            
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
            if ask_box.get_clicked_button() not in ['CSV', 'JSON']:
                return
            
            export_dir = "exports"
            os.makedirs(export_dir, exist_ok=True)
            
            rows = []
            all_keys = set()

            def flatten_dict(d, parent_key = ''):
                items = {}
                for k, v in d.items():
                    new_key = f"{parent_key}.{k}" if parent_key else k
                    if isinstance(v, dict):
                        items.update(flatten_dict(v, new_key))
                    else:
                        items[new_key] = v
                return items
            
            for model_name, fallback_meta in info.items():
                model_dir = fallback_meta.get("model_dir", ".")
                meta_json_path = os.path.join(model_dir, "meta.json")
                meta = fallback_meta.copy()

                if os.path.exists(meta_json_path):
                    try:
                        with open(meta_json_path, "r", encoding='utf-8') as f:
                            detailed_meta = json.load(f)
                            meta.update(detailed_meta)
                    except Exception as e:
                        print(f"读取 {meta_json_path} 失败: {e}")
                
                flat_meta = flatten_dict(meta)
                flat_meta["model_name"] = model_name
                all_keys.update(flat_meta.keys())
                rows.append(flat_meta)

            if ask_box.get_clicked_button() == 'CSV':
                export_path = os.path.join(export_dir, "models_export.csv")
                with open(export_path, 'w', newline='', encoding='utf-8-sig') as cf:
                    writer = csv.writer(cf)
                    writer.writerow(all_keys)
                    for row in rows:
                        writer.writerow([row.get(k, "") for k in all_keys])
                
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
                export_path = os.path.join(export_dir, "models_export.json")
                merged_json = {}
                for row in rows:
                    model_name = row.get("model_name", "未知模型")
                    merged_json[model_name] = row
                with open(export_path, 'w', encoding='utf-8') as jf:
                    json.dump(merged_json, jf, indent=2, ensure_ascii=False)

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
                title="下载模型信息失败",
                text=f"模型信息导出失败: {e}",
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

    def output_model(self, model_name, model_dir):
        try:
            self.theme = self.get_current_theme()
            zip_filename = f"{model_name}_export.zip"
            zip_path = os.path.join("exports", zip_filename)
            os.makedirs("exports", exist_ok=True)

            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
                model_path = os.path.join(model_dir, model_name)
                if os.path.exists(model_path):
                    zf.write(model_path, arcname=model_name)

                meta_path = os.path.join(model_dir, "meta.json")
                if os.path.exists(meta_path):
                    zf.write(meta_path, arcname="meta.json")

                for img_name in ['age_scatter.png', 'gender_confusion.png']:
                    img_path = os.path.join(model_dir, img_name)
                    if os.path.exists(img_path):
                        zf.write(img_path, arcname=img_name)
            msg_box = MessageBox(
                parent=self,
                title="提示",
                text="导出成功！",
                theme=self.theme,
                icon=QMessageBox.Information
            )
            msg_box.exec_()
            QDesktopServices.openUrl(QUrl.fromLocalFile(zip_path))
        except Exception as e:
            msg_box = MessageBox(
                parent=self,
                title="提示",
                text=f"导出失败: {e}",
                theme=self.theme,
                icon=QMessageBox.Critical
            )
            msg_box.exec_()

    def on_model_import_finished(self, success, msg):
        self.btn_upload.setEnabled(True)
        if success:
            self.refresh()
        self.theme = self.get_current_theme()
        msg_box = MessageBox(
            parent=self,
            title="上传模型",
            text=msg,
            icon=QMessageBox.Information if success else QMessageBox.Critical,
            theme=self.theme
        )
        msg_box.exec_()
        
    def show_delete_overlay(self, model_name, model_dir):
        self._pending_delete = (model_name, model_dir)
        self.delete_label.setText(f"确认删除模型: {model_name}")
        self.delete_input.clear()
        self.delete_overlay.show()
        self.delete_overlay.raise_()
        self.delete_overlay.setMaximumHeight(0)
        target_height = self.delete_overlay.sizeHint().height()
        if target_height < 50:
            target_height = 50
        anim = QPropertyAnimation(self.delete_overlay, b"maximumHeight")
        anim.setDuration(300)
        anim.setStartValue(0)
        anim.setEndValue(target_height)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.start()
        self._anim = anim

    def _hide_delete_overlay(self):
        anim = QPropertyAnimation(self.delete_overlay, b"maximumHeight")
        anim.setDuration(200)
        anim.setStartValue(self.delete_overlay.height())
        anim.setEndValue(0)
        anim.setEasingCurve(QEasingCurve.InCubic)
        anim.finished.connect(self.delete_overlay.hide)
        anim.start()
        self._anim = anim

    def _do_delete_model(self):
        if not self._pending_delete:
            return
        model_name, model_dir = self._pending_delete
        text = self.delete_input.text().strip()
        if text != model_name:
            self.delete_label.setText("模型名称输入错误，未删除任何文件。")
            return
        try:
            if model_dir == '.':
                os.remove(model_name)
            else:
                shutil.rmtree(model_dir)
            if os.path.exists(MODELS_INFO_FILE):
                with open(MODELS_INFO_FILE, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                if model_name in info:
                    del info[model_name]
                    with open(MODELS_INFO_FILE, 'w', encoding='utf-8') as f:
                        json.dump(info, f, ensure_ascii=False, indent=2)
            self.delete_label.setText(f"模型 {model_name} 已成功删除。")
            self.refresh()
            self._hide_delete_overlay()
        except Exception as e:
            self.delete_label.setText(f"删除失败：{e}")