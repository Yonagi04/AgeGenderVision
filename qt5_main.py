import sys
from PyQt5.QtWidgets import QApplication
from panels.main_panel import MainPanelWindow
from utils.qss_loader import load_qss, LIGHT_QSS_FILE, DARK_QSS_FILE
from utils.sys_utils import detect_system_theme

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    theme = detect_system_theme()
    if theme == "light":
        load_qss(app, LIGHT_QSS_FILE)
    else:
        load_qss(app, DARK_QSS_FILE)
    window = MainPanelWindow(theme=theme)
    window.show()
    sys.exit(app.exec_())