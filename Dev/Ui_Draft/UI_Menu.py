import sys
import pyembroidery as pe
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
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # Create a blank canvas for drawing contours
        contour_image = np.zeros_like(image)
        cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 1)

        edge_image_path = "edges_output.png"
        success = cv2.imwrite(edge_image_path, contour_image)

        if not success:
            print("Error: Failed to save the edge-detected image.")
            return

        print(f"Edge-detected image saved as: {edge_image_path}")

        outfile = "emb"
        smallthresh = 6
        scale_factor = 3.0

        p1 = pe.EmbPattern()
        print(f"THere are {len(contours)} contours found")
        for c in contours:
            print(f"This contour is {len(c)} points long")
            if len(c) < smallthresh:
                print("Really short contour, probably noise")

            else:
                print("Adding the contour of an object to our pattern")
                stitches = [];
                for pt in c:
                    stitches.append([pt[0][0], pt[0][1]])
                p1.add_block(stitches, "blue")
        pe.write_pes(p1, f"{outfile}.pes")
        pe.write_png(p1, f"{outfile}.png")



        pixmap = QPixmap(edge_image_path)
        if pixmap.isNull():
            print("Error: Failed to load the saved image.")
            return

        pixmap = pixmap.scaled(300, 200, Qt.AspectRatioMode.KeepAspectRatio)
        self.imageLabel.setPixmap(pixmap)
        self.imageLabel.setScaledContents(True)

        print("Edge-detected image successfully loaded into QLabel.")

    def cannyTrackbars(self):
        if not hasattr(self.screen1, "global_image") or not self.screen1.global_image:
            print("Error: No image loaded.")
            return

        image_path = self.screen1.global_image
        print(f"Processing image: {image_path}")

        # Read image and preprocess
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Unable to read the image.")
            return

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

        # Ensure sliders exist
        if not hasattr(self, "lowThresh") or not hasattr(self, "highThresh"):
            print("Error: Threshold sliders not found.")
            return

        # Enable sliders
        self.lowThresh.setEnabled(True)
        self.highThresh.setEnabled(True)

        # Connect slider events to update function
        self.lowThresh.valueChanged.connect(lambda: self.update_edges(blurred))
        self.highThresh.valueChanged.connect(lambda: self.update_edges(blurred))

        # Call update once to show initial edges
        self.update_edges(blurred)

    def update_edges(self, blurred):
        low = self.lowThresh.value()
        high = self.highThresh.value()

        edges = cv2.Canny(blurred, low, high)
        edge_image_path = "edges_output.png"

        success = cv2.imwrite(edge_image_path, edges)
        if not success:
            print("Error: Failed to save the edge-detected image.")
            return

        print(f"Edge-detected image saved as: {edge_image_path}")

        # Load and display the image
        pixmap = QPixmap(edge_image_path)
        if pixmap.isNull():
            print("Error: Failed to load the saved image.")
            return

        pixmap = pixmap.scaled(300, 200, Qt.AspectRatioMode.KeepAspectRatio)
        self.imageLabel.setPixmap(pixmap)
        self.imageLabel.setScaledContents(True)

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
