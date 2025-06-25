import os
from PyQt5.QtCore import QThread, pyqtSignal
from utils.model_utils import output_model_onnx

class ModelOutputOnnxThread(QThread):
    finished = pyqtSignal(bool, str)

    def __init__(self, model_dir, model_name, model_type):
        super().__init__()
        self.model_dir = model_dir
        self.model_name = model_name
        self.model_type = model_type

    def run(self):
        result = output_model_onnx(self.model_dir, self.model_name, self.model_type)
        if result:
            self.finished.emit(True, "模型转换成功")
        else:
            self.finished.emit(False, "模型转换失败")