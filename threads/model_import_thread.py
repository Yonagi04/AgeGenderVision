import os
from PyQt5.QtCore import QThread, pyqtSignal
from utils.model_utils import check_pth_file, get_resnet_type, save_model

class ModelImportThread(QThread):
    finished = pyqtSignal(bool, str)

    def __init__(self, model_path, allowed_types):
        super().__init__()
        self.model_path = model_path
        self.allowed_types = allowed_types

    def run(self):
        try:
            if not check_pth_file(self.model_path):
                self.finished.emit(False, "模型文件不合法或无法加载")
                return
            model_type = get_resnet_type(self.model_path)
            if not model_type in self.allowed_types:
                self.finished.emit(False, "模型类型不合法")
                return
            model_name = os.path.basename(self.model_path)
            result = save_model(model_type, model_name, self.model_path)
            if result:
                self.finished.emit(True, f"模型上传成功，模型名称: {model_name}，模型类型: {model_type}")
            else:
                self.finished.emit(False, "模型上传失败")
        except Exception as e:
            self.finished.emit(False, f"模型上传出现异常: {e}")

