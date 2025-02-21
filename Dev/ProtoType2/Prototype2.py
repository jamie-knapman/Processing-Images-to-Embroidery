import sys
import pyembroidery as pe
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QStackedWidget, QLabel, QMainWindow, \
    QFileDialog, QPlainTextEdit
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


        uic.loadUi("MainMenu1.ui", self)
        self.setFixedSize(800, 450)


        self.button1 = self.findChild(QPushButton, "createNew")
        self.button2 = self.findChild(QPushButton, "openSaved")

        self.button1.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        self.button2.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(2))

class Screen1(QMainWindow):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget

        uic.loadUi("CreateNew1.ui", self)
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
            self.imageLabel.setPixmap(self.pixmap)
            self.imageLabel.setScaledContents(True)
            self.NextButton.setEnabled(True)
            print(f"Image loaded: {file_path}")


class Screen3(QMainWindow):
    def __init__(self, stacked_widget, screen1):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.screen1 = screen1

        uic.loadUi("Edges1.ui", self)
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

        self.smallThreshIn = self.findChild(QPlainTextEdit, "smallThreshIn")
        if not self.smallThreshIn:
            print("Error: smallThreshIn QPlainTextEdit not found.")

        self.nextButton = self.findChild(QPushButton, "next")
        self.nextButton.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(4))

    def get_small_thresh(self):
        """ Retrieves and validates the user's threshold input """
        if not self.smallThreshIn:
            return 6

        text = self.smallThreshIn.toPlainText().strip()
        try:
            return int(text)
        except ValueError:
            try:
                return float(text)
            except ValueError:
                print("Invalid input. Using default threshold (6).")
                return 6

    def vectorise(self):
        if not hasattr(self.screen1, "global_image") or not self.screen1.global_image:
            print("Error: No image loaded.")
            return

        image_path = self.screen1.global_image
        print(f"Processing image: {image_path}")

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        v = np.median(image)
        low = int(max(0, 0.77 * v))
        high = int(min(255, 0.77 * v))

        if image is None:
            print("Error: Unable to read the image.")
            return

        self.nextButton.setEnabled(True)

        blurred = cv2.GaussianBlur(image, (7,7), 0)
        edges = cv2.Canny(blurred, low, high)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        contour_image = np.zeros_like(image)
        cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 1)

        edge_image_path = "edges_output.png"
        success = cv2.imwrite(edge_image_path, contour_image)

        if not success:
            print("Error: Failed to save the edge-detected image.")
            return

        print(f"Edge-detected image saved as: {edge_image_path}")

        outfile = "emb1"
        scale_factor = 3.0
        smallthresh = self.get_small_thresh()

        p1 = pe.EmbPattern()
        print(f"There are {len(contours)} contours found")

        contour_paths = []
        for c in contours:
            if len(c) < smallthresh:
                continue

            stitches = [(pt[0][0] * scale_factor, pt[0][1] * scale_factor) for pt in c]
            contour_paths.append(stitches)

        ordered_stitches = []
        current_pos = (0, 0)

        while contour_paths:
            next_contour = min(contour_paths,
                               key=lambda path: np.linalg.norm(np.array(path[0]) - np.array(current_pos)))
            contour_paths.remove(next_contour)

            if ordered_stitches:
                ordered_stitches.append((next_contour[0][0], next_contour[0][1], "JUMP"))

            ordered_stitches.extend(next_contour)
            current_pos = next_contour[-1]

        for stitch in ordered_stitches:
            if isinstance(stitch, tuple) and len(stitch) == 3 and stitch[2] == "JUMP":
                p1.add_stitch_absolute(pe.JUMP, stitch[0], stitch[1])
            else:
                p1.add_stitch_absolute(pe.STITCH, stitch[0], stitch[1])

        p1.end()

        pe.write_pes(p1, f"{outfile}.pes")
        pe.write_png(p1, f"{outfile}.png")

        pixmap = QPixmap(f"{outfile}.png")
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

        image = cv2.imread(image_path)
        if image is None:
            print("Error: Unable to read the image.")
            return

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)


        if not hasattr(self, "lowThresh") or not hasattr(self, "highThresh"):
            print("Error: Threshold sliders not found.")
            return


        self.lowThresh.setEnabled(True)
        self.highThresh.setEnabled(True)

        self.lowThresh.valueChanged.connect(lambda: self.update_edges(blurred))
        self.highThresh.valueChanged.connect(lambda: self.update_edges(blurred))

        self.update_edges(blurred)

        self.submit = self.findChild(QPushButton, "Submit")
        if self.submit:
            self.submit.clicked.connect(lambda: self.writeManual())
        self.submit.setEnabled(True)

    def update_edges(self, blurred):
        """Updates the edges dynamically based on trackbar values."""
        low = self.lowThresh.value()
        high = self.highThresh.value()

        self.edges = cv2.Canny(blurred, low, high)
        edge_image_path = "edges_output.png"

        success = cv2.imwrite(edge_image_path, self.edges)
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

    def writeManual(self):
        """Processes manually detected edges and converts them into embroidery format."""
        if not hasattr(self, "edges") or self.edges is None:
            print("Error: No edge data available.")
            return

        outfile = "emb1"
        scale_factor = 3.0
        smallthresh = self.get_small_thresh()

        contours, _ = cv2.findContours(self.edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        contour_image = np.zeros_like(self.edges)
        cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 1)

        p1 = pe.EmbPattern()
        print(f"There are {len(contours)} contours found")

        contour_paths = []
        for c in contours:
            if len(c) < smallthresh:
                continue

            stitches = [(pt[0][0] * scale_factor, pt[0][1] * scale_factor) for pt in c]
            contour_paths.append(stitches)

        ordered_stitches = []
        current_pos = (0, 0)

        while contour_paths:
            next_contour = min(contour_paths,
                               key=lambda path: np.linalg.norm(np.array(path[0]) - np.array(current_pos)))
            contour_paths.remove(next_contour)

            if ordered_stitches:
                ordered_stitches.append((next_contour[0][0], next_contour[0][1], "JUMP"))

            ordered_stitches.extend(next_contour)
            current_pos = next_contour[-1]

        for stitch in ordered_stitches:
            if isinstance(stitch, tuple) and len(stitch) == 3 and stitch[2] == "JUMP":
                p1.add_stitch_absolute(pe.JUMP, stitch[0], stitch[1])
            else:
                p1.add_stitch_absolute(pe.STITCH, stitch[0], stitch[1])

        p1.end()

        pe.write_pes(p1, f"{outfile}.pes")
        pe.write_png(p1, f"{outfile}.png")

        pixmap = QPixmap(f"{outfile}.png")
        if pixmap.isNull():
            print("Error: Failed to load the saved image.")
            return

        pixmap = pixmap.scaled(300, 200, Qt.AspectRatioMode.KeepAspectRatio)
        self.imageLabel.setPixmap(pixmap)
        self.imageLabel.setScaledContents(True)

        print("Edge-detected image successfully loaded into QLabel.")



class Screen4(QMainWindow):
    def __init__(self, stacked_widget, screen3):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.screen3 = screen3


        uic.loadUi("Shading.ui", self)
        self.setFixedSize(800, 450)

        # Find buttons from the UI and connect them
        self.auto = self.findChild(QPushButton, "Auto")
        self.auto.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))

        self.stitchDistance = self.findChild(QPlainTextEdit, "stitchDistance")
        if not self.stitchDistance:
            print("Error: smallThreshIn QPlainTextEdit not found.")

        self.imageLabel = self.findChild(QLabel, "imageLabel")
        if self.imageLabel:
            self.imageLabel.setText("No Image Selected")

        outfile = "emb1.png"
        pixmap = QPixmap(outfile)
        if pixmap.isNull():
            print("Error: Failed to load the saved image.")
            return

        pixmap = pixmap.scaled(300, 200, Qt.AspectRatioMode.KeepAspectRatio)
        self.imageLabel.setPixmap(pixmap)
        self.imageLabel.setScaledContents(True)

        print("Image Transferred Successfully")

    def get_stitch_distance(self):
        """ Retrieves and validates the user's threshold input """
        if not self.stitchDistance:
            return 6

        text = self.stitchDistance.toPlainText().strip()
        try:
            return int(text)
        except ValueError:
            try:
                return float(text)
            except ValueError:
                print("Invalid input. Using default threshold (6).")
                return 6


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
screen4 = Screen4(stacked_widget, screen3)

stacked_widget.addWidget(main_window)
stacked_widget.addWidget(screen1)
stacked_widget.addWidget(screen2)
stacked_widget.addWidget(screen3)
stacked_widget.addWidget(screen4)


stacked_widget.setCurrentIndex(0)
stacked_widget.show()
sys.exit(app.exec())
