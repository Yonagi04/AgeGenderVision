from PyQt5.QtCore import QThread, pyqtSignal

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
                result = subprocess.run(self.cmd, shell=False, check=True, capture_output=True, text=True, encoding="utf-8", env=self.env)
                self.finished.emit(result.stdout, None)
            else:
                subprocess.run(self.cmd, shell=False, check=True, env=self.env)
                self.finished.emit("", None)
        except Exception as e:
            self.finished.emit("", e)