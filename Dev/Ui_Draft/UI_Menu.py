import sys

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QStackedWidget, QLabel, QMainWindow, \
    QFileDialog
from PyQt6 import uic
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours
import svgwrite


class MainWindow(QMainWindow):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget


        uic.loadUi("MainMenu.ui", self)
        self.setFixedSize(800, 450)


        # Find buttons from the UI and connect them
        self.button1 = self.findChild(QPushButton, "createNew")
        self.button2 = self.findChild(QPushButton, "openSaved")

        self.button1.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        self.button2.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(2))

class Screen1(QMainWindow):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget

        uic.loadUi("CreateNew.ui", self)
        self.setFixedSize(800, 450)

        self.backButton = self.findChild(QPushButton, "BackToMain")
        if self.backButton:
            self.backButton.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))

        self.uploadButton = self.findChild(QPushButton, "UploadImage")
        if self.uploadButton:
            self.uploadButton.clicked.connect(self.upload_image)

        self.imageLabel = self.findChild(QLabel, "imageLabel")
        if self.imageLabel:
            self.imageLabel.setText("No Image Selected")

        self.NextButton = self.findChild(QPushButton, "Next")
        if self.NextButton:
            self.NextButton.clicked.connect((lambda: self.stacked_widget.setCurrentIndex(3)))

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
        if file_path:
            self.global_image = file_path
            self.pixmap = QPixmap(file_path)
            self.pixmap = self.pixmap.scaled(300, 200, Qt.AspectRatioMode.KeepAspectRatio)
            self.imageLabel.setPixmap(self.pixmap)
            self.imageLabel.setScaledContents(False)
            self.NextButton.setEnabled(True)
            print(f"Image loaded: {file_path}")




class Screen3(QMainWindow):
    def __init__(self, stacked_widget, screen1):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.screen1 = screen1

        uic.loadUi("Edges.ui", self)
        self.setFixedSize(800, 450)

        self.backButton = self.findChild(QPushButton, "BackToMain")
        if self.backButton:
            self.backButton.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))

        self.autoFind = self.findChild(QPushButton, "Auto")
        if self.autoFind:
            self.autoFind.clicked.connect(self.vectorise)

        self.imageLabel = self.findChild(QLabel, "imageLabel")
        if self.imageLabel:
            self.imageLabel.setText("No Image Selected")

        self.manualFind = self.findChild(QPushButton, "Manual")
        if self.manualFind:
            self.manualFind.clicked.connect(self.cannyTrackbars)

    def vectorise(self):
        if not hasattr(self.screen1, "global_image") or not self.screen1.global_image:
            print("Error: No image loaded.")
            return

        image_path = self.screen1.global_image
        print(f"Processing image: {image_path}")

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print("Error: Unable to read the image.")
            return

        edges = cv2.Canny(image, 50, 150)
        edge_image_path = "edges_output.png"
        success = cv2.imwrite(edge_image_path, edges)

        if not success:
            print("Error: Failed to save the edge-detected image.")
            return

        print(f"Edge-detected image saved as: {edge_image_path}")

        pixmap = QPixmap(edge_image_path)
        if pixmap.isNull():
            print("Error: Failed to load the saved image.")
            return

        pixmap = pixmap.scaled(300, 200, Qt.AspectRatioMode.KeepAspectRatio)
        self.imageLabel.setPixmap(pixmap)
        self.imageLabel.setScaledContents(True)

        print("Edge-detected image successfully loaded into QLabel.")

    def cannyTrackbars(self):
        print("placeholder")

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
screen3 = Screen3(stacked_widget, screen1)

stacked_widget.addWidget(main_window)
stacked_widget.addWidget(screen1)
stacked_widget.addWidget(screen2)
stacked_widget.addWidget(screen3)

stacked_widget.setCurrentIndex(0)
stacked_widget.show()
sys.exit(app.exec())
