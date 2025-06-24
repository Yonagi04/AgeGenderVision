import sys
import platform
import subprocess

def detect_system_theme():
    system = platform.system()
    try:
        if system == 'Windows':
            import winreg
            registry = winreg.ConnectRegistry(None, winreg.HKEY_CURRENT_USER)
            key_path = r"Software\Microsoft\Windows\CurrentVersion\Themes\Personalize"
            key = winreg.OpenKey(registry, key_path)
            value, _ = winreg.QueryValueEx(key, "AppsUseLightTheme")
            return "light" if value == 1 else "dark"
        elif system == 'Darwin':
            result = subprocess.run(
                ["defaults", "read", "-g", "AppleInterfaceStyle"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True
            )
            return "dark" if "Dark" in result.stdout else "light"
        elif system == 'Linux':
            result = subprocess.run(
                ["gsettings", "get", "org.gnome.desktop.interface", "color-scheme"],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True
            )
            if "dark" in result.stdout.lower():
                return "dark"
            else:
                return "light"
    except Exception as e:
        pass
    return "light"