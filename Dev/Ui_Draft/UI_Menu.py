import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QStackedWidget, QLabel, QMainWindow, \
    QFileDialog
from PyQt6 import uic

global_image = None

class MainWindow(QMainWindow):  # Change QWidget to QMainWindow
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget

        # Load the UI from Qt Designer
        uic.loadUi("MainMenu.ui", self)
        self.setFixedSize(800, 450)  # Set fixed width and height


        # Find buttons from the UI and connect them
        self.button1 = self.findChild(QPushButton, "createNew")  # Match the objectName from UI
        self.button2 = self.findChild(QPushButton, "openSaved")

        self.button1.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        self.button2.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(2))

class Screen1(QMainWindow):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget

        # Load UI
        uic.loadUi("CreateNew.ui", self)
        self.setFixedSize(800, 450)

        # Back button
        self.backButton = self.findChild(QPushButton, "BackToMain")
        if self.backButton:
            self.backButton.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))

        # Image Upload Button
        self.uploadButton = self.findChild(QPushButton, "UploadImage")  # Ensure this name matches in Qt Designer
        if self.uploadButton:
            self.uploadButton.clicked.connect(self.upload_image)

        # Image Label
        self.imageLabel = self.findChild(QLabel, "imageLabel")  # Ensure this name matches in Qt Designer
        if self.imageLabel:
            self.imageLabel.setText("No Image Selected")  # Default text

        # Image Upload Button
        self.NextButton = self.findChild(QPushButton, "Next")  # Ensure this name matches in Qt Designer
        if self.NextButton:
            self.NextButton.clicked.connect((lambda: self.stacked_widget.setCurrentIndex(3)))

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.pixmap = QPixmap(file_path)  # Store pixmap in self to prevent garbage collection
            self.pixmap = self.pixmap.scaled(300, 200, Qt.AspectRatioMode.KeepAspectRatio)  # Correct scaling
            self.imageLabel.setPixmap(self.pixmap)
            self.imageLabel.setScaledContents(False)  # Prevents distortion
            self.NextButton.setEnabled(True)
            print(f"Image loaded: {file_path}")  # Debugging

            global_image = self.pixmap

class Screen3(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("This is Screen 3"))

        back_button = QPushButton("Back to Main")
        back_button.clicked.connect(lambda: stacked_widget.setCurrentIndex(1))
        layout.addWidget(back_button)

        self.setLayout(layout)


class Screen2(QWidget):
    def __init__(self, stacked_widget):
        super().__init__()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("This is Screen 2"))

        back_button = QPushButton("Back to Main")
        back_button.clicked.connect(lambda: stacked_widget.setCurrentIndex(0))
        layout.addWidget(back_button)

        self.setLayout(layout)

app = QApplication(sys.argv)
stacked_widget = QStackedWidget()

main_window = MainWindow(stacked_widget)
screen1 = Screen1(stacked_widget)
screen2 = Screen2(stacked_widget)
screen3 = Screen3(stacked_widget)

stacked_widget.addWidget(main_window)
stacked_widget.addWidget(screen1)
stacked_widget.addWidget(screen2)
stacked_widget.addWidget(screen3)

stacked_widget.setCurrentIndex(0)
stacked_widget.show()
sys.exit(app.exec())
