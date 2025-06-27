from PyQt5.QtWidgets import (
    QWidget, QPushButton, QVBoxLayout, QLabel, QScrollArea, QGroupBox, 
    QHBoxLayout, QLineEdit, QFileDialog, QMessageBox, QMenu, QDialog,
    QComboBox, QAction
)
from PyQt5.QtCore import QUrl, QSize
from PyQt5.QtGui import QDesktopServices, QIcon
import os
import zipfile
import copy
import shutil
from threads.model_import_thread import ModelImportThread
from threads.model_output_onnx_thread import ModelOutputOnnxThread
from widgets.message_box import MessageBox
from widgets.input_dialog import InputDialog
from widgets.select_dialog import SelectDialog
from widgets.netron_viewer import NetronLocalBrowserViewer
from widgets.tag_edit_dialog import TagEditDialog
from widgets.flow_layout import FlowLayout
from services.model_service import ModelService
from convention.result_code import ResultCode
from utils.ui_utils import create_tag_container

MODELS_INFO_FILE = os.path.join("data", "models.json")

class ModelListPanel(QWidget):
    def __init__(self, parent=None, theme='light'):
        super().__init__(parent)
        self.theme = theme
        self.is_descending = True
        self.sort_by = 'created_time'
        self.current_model_info = {}
        self.viewer = None
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
        self.sort_type = QComboBox()
        self.sort_type.addItems(['按创建时间排序', '按修改时间排序', '按模型名称排序'])
        
        self.btn_order = QPushButton()
        if self.theme == 'light':
            self.btn_order.setIcon(QIcon("assets/svg/descend_light.svg"))
        else:
            self.btn_order.setIcon(QIcon("assets/svg/descend_dark.svg"))
        self.btn_order.setIconSize(QSize(28, 28))
        self.btn_order.setFixedSize(36, 36)
        self.btn_order.setStyleSheet("border:none; background:transparent;")

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
        self.hbox.addWidget(self.sort_type)
        self.hbox.addWidget(self.btn_order)
        self.hbox.addWidget(self.btn_refresh)
        self.hbox.addWidget(self.btn_upload)
        self.hbox.addWidget(self.btn_download)
        self.layout.addLayout(self.hbox)
        self.layout.addWidget(self.scroll)

        self.btn_refresh.clicked.connect(self.refresh)
        self.btn_search.clicked.connect(self.search)
        self.btn_download.clicked.connect(self.download)
        self.btn_upload.clicked.connect(self.upload)
        self.sort_type.currentIndexChanged.connect(self.on_sort_type_changed)
        self.btn_order.clicked.connect(self.toggle_sort_order)

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

    def update_model_list(self, info_dict: dict):
        if self.sort_by in ['created_time', 'updated_time']:
            sorted_items = sorted(
                info_dict.items(),
                key=lambda x: x[1].get(self.sort_by, ""),
                reverse=self.is_descending
            )
        elif self.sort_by == 'model_name':
            sorted_items = sorted(info_dict.items(), key=lambda x: x[0].lower(), reverse=self.is_descending)
        else:
            sorted_items = list(info_dict.items())
        
        self.theme = self.get_current_theme()
        for i in reversed(range(self.inner_layout.count())):
            widget = self.inner_layout.itemAt(i).widget()
            if widget:
                widget.deleteLater()

        for model_name, meta in sorted_items:
            model_type = meta.get("model_type", "未知")
            model_dir = meta.get("model_dir", "未知")
            created_time = meta.get("created_time", "未知")
            update_time = meta.get("update_time", "未知")
            description = meta.get("description", "未知")
            tags = meta.get("tags", [])
            group = QGroupBox()
            vbox = QVBoxLayout(group)
            vbox.addWidget(QLabel(f"模型名称: {model_name}"))
            vbox.addWidget(QLabel(f"模型类型: {model_type}"))
            vbox.addWidget(QLabel(f"存放位置: {model_dir}"))
            vbox.addWidget(QLabel(f"创建时间: {created_time}"))
            vbox.addWidget(QLabel(f"更新时间: {update_time}"))
            vbox.addWidget(QLabel(f"备注: {description}"))

            # if tags:
            #     flow = FlowLayout(spacing=6)
            #     for tag in tags:
            #         label = QLabel(tag["text"])
            #         color = tag["color"]
            #         text_color = get_contrast_font_color(color)
            #         label.setStyleSheet(f"""
            #                 QLabel {{
            #                     background-color: {color};
            #                     color: {text_color};
            #                     border-radius: 6px;
            #                     padding: 2px 6px;
            #                     font-size: 16px;
            #                     min-height: 30px;
            #                     max-height: 40px;
            #                 }}
            #             """)
            #         label.setFixedHeight(24)
            #         flow.addWidget(label)

            #     tag_container = QWidget()
            #     tag_container.setLayout(flow)
            #     tag_container.setStyleSheet("background-color: transparent;")
            #     vbox.addWidget(tag_container)
            # else:
            #     label_no_tag = QLabel("无标签")
            #     label_no_tag.setStyleSheet("color: gray; font-size: 18px;")
            #     vbox.addWidget(label_no_tag)
            vbox.addWidget(create_tag_container(tags))

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
            btn_export = QPushButton("导出模型")
            if model_dir == '.':
                btn_export.setEnabled(False)
                btn_export.setToolTip("无法导出，因为模型存储在主文件夹")
            else:
                btn_export.setEnabled(True)
            menu_export = QMenu()
            menu_export.addAction("导出为 PTH", lambda m=model_name, d=model_dir: self.output_model(m, d))
            menu_export.addAction("导出为 ONNX", lambda m=model_name, d=model_dir, t=model_type: self.output_model_onnx(m, d, t))
            btn_export.setMenu(menu_export)

            hbox.addWidget(btn_export)

            btn_more = QPushButton("更多")
            menu = QMenu()
            menu.addAction("重命名模型", lambda m=model_name: self.rename_model(m))
            menu.addAction("设置备注", lambda m=model_name,d=description: self.set_description(m, d))
            menu.addAction("设置标签", lambda m=model_name, t=tags: self.set_tags(m, t))
            menu.addAction("修改模型类型", lambda m=model_name, t=model_type: self.update_model_type(m, t))
            menu.addAction("模型比较", lambda m=model_name: self.open_compare(m))
            menu.addAction("查看模型结构", lambda m=model_name: self.view_model_structure(m))
            menu.addAction("删除模型", lambda m=model_name, d=model_dir: self.delete_model(m, d))
            btn_more.setMenu(menu)
            hbox.addWidget(btn_more)

            vbox.addLayout(hbox)
            self.inner_layout.addWidget(group)
        self.inner_layout.addStretch()

    def refresh(self):
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
        self.current_model_info = model_info.data
        self.update_model_list(self.current_model_info)

    def search(self):
        self.theme = self.get_current_theme()
        keyword = self.search_box.text().strip().lower()
        if not keyword:
            self.refresh()
            return

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
        self.current_model_info = model_info.data
        self.update_model_list(self.current_model_info)
    
    def on_sort_type_changed(self, index):
        mapping = {
            0: "created_time",
            1: "update_time",
            2: "model_name"
        }
        self.sort_by = mapping.get(index, "created_time")
        self.update_model_list(self.current_model_info)
    
    def toggle_sort_order(self):
        self.theme = self.get_current_theme()
        self.is_descending = not self.is_descending
        icon_path = f"assets/svg/{'descend' if self.is_descending else 'ascend'}_{self.theme}.svg"
        self.btn_order.setIcon(QIcon(icon_path))
        self.update_model_list(self.current_model_info)

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
        self.btn_upload.setEnabled(True)
        self.thread = None
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
        ask_box = MessageBox(
            parent=self,
            title="提示",
            text="请选择导出格式（zip/pth）",
            theme=self.theme,
            addButton=False
        )
        ask_box.add_buttons({
            "ZIP": QMessageBox.AcceptRole,
            "PTH": QMessageBox.AcceptRole,
            "取消": QMessageBox.RejectRole
        })
        ask_box.exec_()
        user_choice = ask_box.get_clicked_button()
        if user_choice not in ['ZIP', 'PTH']:
            return
        result = ModelService.output_model(model_name, model_dir, user_choice)
        if not result.success:
            msg_box = MessageBox(
                parent=self,
                title="错误",
                text= result.message,
                theme=self.theme,
                icon=QMessageBox.Critical
            )
            msg_box.exec_()
        path = result.data
        msg_box = MessageBox(
            parent=self,
            title="提示",
            text="导出成功！",
            theme=self.theme,
            icon=QMessageBox.Information
        )
        msg_box.exec_()
        if user_choice == 'ZIP':
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        else:
            QDesktopServices.openUrl(QUrl.fromLocalFile("exports"))
        
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

    def output_model_onnx(self, model_name, model_dir, model_type):
        save_path = os.path.join(model_dir, model_name)
        if not os.path.exists(save_path):
            msg_box = MessageBox(
                parent=self,
                title="错误",
                text="模型文件不存在",
                icon=QMessageBox.Critical,
                theme=self.theme
            )
            msg_box.exec_()
            return
        msg_box = MessageBox(
            parent=self,
            text="正在导出ONNX，需要一些时间，请稍候...",
            theme=self.theme
        )
        msg_box.show()
        self.thread = ModelOutputOnnxThread(model_dir, model_name, model_type)
        self.thread.finished.connect(self.output_model_onnx_finished)
        self.thread.start()

    def output_model_onnx_finished(self, success, msg):
        self.theme = self.get_current_theme()
        model_name = self.thread.model_name
        model_dir = self.thread.model_dir
        onnx_name = os.path.splitext(model_name)[0] + ".onnx"
        onnx_path = os.path.join(model_dir, onnx_name)

        self.thread = None

        if not success or not os.path.exists(onnx_path):
            msg_box = MessageBox(
                parent=self,
                title="错误",
                text="模型转换失败",
                icon=QMessageBox.Critical,
                theme=self.theme
            )
            msg_box.exec_()
            return
        
        try:
            dest_path = os.path.join("exports", onnx_name)
            shutil.copy2(onnx_path, dest_path)
            os.remove(onnx_path)
            msg_box = MessageBox(
                parent=self,
                title="提示",
                text="导出成功",
                icon=QMessageBox.Information,
                theme=self.theme
            )
            msg_box.exec_()
            QDesktopServices.openUrl(QUrl.fromLocalFile("exports"))
        except Exception as e:
            msg_box = MessageBox(
                parent=self,
                title="错误",
                text=f"导出失败: {e}",
                icon=QMessageBox.Critical,
                theme=self.theme
            )
            msg_box.exec_()
        
    def set_tags(self, model_name, tags):
        self.theme = self.get_current_theme()
        old_tags = copy.deepcopy(tags)
        dialog = TagEditDialog(tags=old_tags)
        if dialog.exec_():
            new_tags = dialog.get_tags()
            if new_tags == old_tags:
                return
            result = ModelService.update_model_tags(model_name, new_tags)
            if not result.success:
                msg = MessageBox(
                    parent=self,
                    title="错误",
                    text=result.message,
                    icon=QMessageBox.Critical,
                    theme=self.theme
                )
                msg.exec_()
                return
            if result.code == ResultCode.NO_DATA:
                msg = MessageBox(
                    parent=self,
                    text=result.message,
                    theme=self.theme
                )
                msg.exec_()
                return
            msg = MessageBox(
                parent=self,
                text=result.message,
                theme=self.theme
            )
            msg.exec_()
            if self.search_box.text().strip():
                self.search()
            else:
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

    def view_model_structure(self, model_name):
        self.theme = self.get_current_theme()
        ask_box = MessageBox(
            parent=self,
            text="查看模型结构需要安装 netron 依赖，确定要查看模型结构吗？",
            theme=self.theme,
            addButton=False
        )
        ask_box.add_buttons({
            "确定": QMessageBox.AcceptRole,
            "取消": QMessageBox.RejectRole
        })
        ask_box.exec_()
        user_choice = ask_box.get_clicked_button()
        if user_choice == '取消':
            return
        result = ModelService.get_model_save_path(model_name)
        if not result.success:
            msg_box = MessageBox(
                parent=self,
                text=result.message,
                title='错误',
                theme=self.theme,
                icon=QMessageBox.Critical
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
        save_path = result.data
        self.viewer = NetronLocalBrowserViewer(save_path)
        self.viewer.start()

    def open_compare(self, model_name):
        parent = self.parent()
        while parent is not None:
            if hasattr(parent, "show_compare_panel_with_model"):
                parent.show_compare_panel_with_model(model_name)
                break
            parent = parent.parent()