import sys

try:
    from PyQt6.QtWidgets import QApplication, QWidget
    print("✅ PyQt6 is installed!")
except ImportError:
    print("❌ PyQt6 is NOT installed!")

try:
    import pyqt6_tools
    print("✅ pyqt6-tools is installed!")
except ImportError:
    print("❌ pyqt6-tools is NOT installed!")

print("\nPython Version:", sys.version)
