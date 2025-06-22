LIGHT_QSS_FILE = 'assets/light.qss'
DARK_QSS_FILE = 'assets/dark.qss'

def load_qss(app, qss_file):
    with open(qss_file, encoding='utf-8') as f:
        app.setStyleSheet(f.read())