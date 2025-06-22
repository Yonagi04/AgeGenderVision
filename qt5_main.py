import sys
from PyQt5.QtWidgets import QApplication
from panels.main_panel import MainPanelWindow
from utils.qss_loader import load_qss, LIGHT_QSS_FILE

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    load_qss(app, LIGHT_QSS_FILE)
    window = MainPanelWindow()
    window.show()
    sys.exit(app.exec_())