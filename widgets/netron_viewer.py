import netron
import threading
import webbrowser
import os

class NetronLocalBrowserViewer:
    def __init__(self, model_path):
        self.model_path = model_path
        self.url = "http://localhost:8080"

    def start(self):
        thread = threading.Thread(
            target=lambda: netron.start(self.model_path, browse=False)
        )
        thread.daemon = True
        thread.start()
        
        webbrowser.open(self.url)

    def stop(self):
        os._exit(0)