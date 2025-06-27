from PyQt5.QtCore import QThread, pyqtSignal
import os
import time

STOP_FLAG_FILE = os.path.abspath("stop.flag")

class TrainThread(QThread):
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int)
    tqdm_signal = pyqtSignal(int, int)
    metrics_signal = pyqtSignal(dict)
    finished = pyqtSignal(object)

    def __init__(self, cmd, env):
        super().__init__()
        self.cmd = cmd
        self.env = env
        self._process = None

    def run(self):
        import subprocess
        import re
        try:
            self._process = subprocess.Popen(
                self.cmd, shell=False, env=self.env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                bufsize=1, universal_newlines=True
            )
            epoch = 0
            total_epochs = None
            last_emit_time = 0
            check_counter = 0
            for line in self._process.stdout:
                check_counter += 1
                if ("\r" in line) or ("|" in line and "it/s" in line):
                    tqdm_match = re.search(r'(\d+)\s*/\s*(\d+)', line)
                    if tqdm_match:
                        current = int(tqdm_match.group(1))
                        total = int(tqdm_match.group(2))
                        now = time.time()
                        if now - last_emit_time > 0.1:
                            self.tqdm_signal.emit(current, total)
                            last_emit_time = now
                    continue
                if "Epoch" in line:
                    m = re.search(r"Epoch\s+(\d+)/(\d+)", line)
                    if m:
                        epoch = int(m.group(1))
                        total_epochs = int(m.group(2))
                        if total_epochs:
                            percent = int(epoch / total_epochs * 100)
                            self.progress_signal.emit(percent)
                if "Train Loss=" in line and "Val Age Loss=" in line:
                    m = re.search(
                        r"Train Loss=([0-9.]+)\s*\|\s*Val Age Loss=([0-9.]+)\s*\|\s*Val Gender Loss=([0-9.]+)\s*\|\s*Val Gender Acc=([0-9.]+)",
                        line
                    )
                    if m:
                        metrics = {
                            "train_loss": float(m.group(1)),
                            "val_age_loss": float(m.group(2)),
                            "val_gender_loss": float(m.group(3)),
                            "val_gender_acc": float(m.group(4))
                        }
                        self.metrics_signal.emit(metrics)
                self.log_signal.emit(line)
                if check_counter >= 20:
                    check_counter = 0
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