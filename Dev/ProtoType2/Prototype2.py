import os
import sys
import pyembroidery as pe
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QStackedWidget, QLabel, QMainWindow, \
    QFileDialog, QPlainTextEdit, QCheckBox
from PyQt6 import uic
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import find_contours
import svgwrite

from Dev.shadingTest.Shading import contours_to_embroidery_with_bridging


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
        print(p1)

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
        if not hasattr(self, "edges") or self.edges is None:
            print("Error: No edge data available.")
            return

        outfile = "emb1"
        scale_factor = 3.0
        smallthresh = self.get_small_thresh()

        dilated = cv2.dilate(self.edges, (3,3), iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        contour_image = np.zeros_like(dilated)
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
    def __init__(self, stacked_widget, screen1):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.screen1 = screen1


        uic.loadUi("Shading.ui", self)
        self.setFixedSize(800, 450)

        self.auto = self.findChild(QPushButton, "Auto")
        if self.auto:
            self.auto.clicked.connect(self.segmentColours)

        self.red1 = self.findChild(QCheckBox, "red1")
        self.red2 = self.findChild(QCheckBox, "red2")
        self.yellow = self.findChild(QCheckBox, "yellow")
        self.orange = self.findChild(QCheckBox, "orange")
        self.lGreen = self.findChild(QCheckBox, "lGreen")
        self.mGreen = self.findChild(QCheckBox, "mGreen")
        self.dGreen = self.findChild(QCheckBox, "dGreen")
        self.cyan = self.findChild(QCheckBox, "cyan")
        self.teal = self.findChild(QCheckBox, "teal")
        self.lBlue = self.findChild(QCheckBox, "lBlue")
        self.mBlue = self.findChild(QCheckBox, "mBlue")
        self.dBlue = self.findChild(QCheckBox, "dBlue")
        self.purple = self.findChild(QCheckBox, "purple")
        self.magenta = self.findChild(QCheckBox, "magenta")
        self.black = self.findChild(QCheckBox, "black")
        self.gray = self.findChild(QCheckBox, "gray")
        self.white = self.findChild(QCheckBox, "white")


        self.segmented_images = {}
        self.colourBridge = []

        self.reconstruct = self.findChild(QPushButton, "reconstruct")
        if self.reconstruct:
            self.reconstruct.clicked.connect(self.reconstructImg)

        self.generate = self.findChild(QPushButton, "generate")
        if self.generate:
            self.generate.clicked.connect(self.generateImg)

        self.finalise = self.findChild(QPushButton, "finalise")
        if self.finalise:
            self.finalise.clicked.connect(self.reconstructGenImg)

    def segmentColours(self):
        if not hasattr(self.screen1, "global_image") or not self.screen1.global_image:
            print("Error: No image loaded.")
            return

        imgPath = self.screen1.global_image
        image = cv2.imread(imgPath)
        if image is None:
            print("Error: Unable to read the image.")
            return


        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        color_ranges = {
            "red": [(0, 100, 50), (10, 255, 255)],
            "red2": [(170, 100, 50), (180, 255, 255)],
            "yellow": [(20, 100, 100), (40, 255, 255)],
            "orange": [(10, 100, 100), (25, 255, 255)],
            "light_green": [(40, 50, 100), (70, 255, 255)],
            "medium_green": [(70, 50, 50), (90, 255, 200)],
            "dark_green": [(90, 50, 50), (110, 255, 150)],
            "cyan": [(90, 50, 50), (110, 255, 255)],
            "teal": [(80, 50, 100), (100, 255, 255)],
            "light_blue": [(90, 50, 150), (110, 255, 255)],
            "medium_blue": [(110, 50, 100), (130, 255, 255)],
            "dark_blue": [(130, 50, 50), (150, 255, 200)],
            "purple": [(130, 50, 50), (160, 255, 255)],
            "magenta": [(130, 50, 50), (160, 255, 255)],
            "black": [(0, 0, 0), (180, 255, 50)],
            "gray": [(0, 0, 50), (180, 50, 200)],
            "white": [(0, 0, 200), (180, 55, 255)],
        }


        PIXEL_THRESHOLD = 500

        mainColours = []

        for name, (lower, upper) in color_ranges.items():
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")


            mask = cv2.inRange(hsv, lower, upper)
            result = cv2.bitwise_and(image, image, mask=mask)


            non_black_pixels = cv2.countNonZero(mask)

            if non_black_pixels >= PIXEL_THRESHOLD:
                output_path = f"{name}.png"
                cv2.imwrite(output_path, result)
                print(f"Saved: {output_path} ({non_black_pixels} pixels)")
                mainColours.append(name)
                self.segmented_images[name] = result
            else:
                print(f"Skipped: {name} (Only {non_black_pixels} pixels)")

        red1 = cv2.imread("red.png")
        red2 = cv2.imread("red2.png")

        if red1 is None or red2 is None:
            print("Error: One or both red images could not be loaded.")
            return
        if red1 is not None and red2 is not None:
            if red1.shape != red2.shape:
                red2 = cv2.resize(red2, (red1.shape[1], red1.shape[0]))
            merged_red = cv2.addWeighted(red1, 1, red2, 1, 0)

            red_mask = cv2.inRange(hsv, np.array([0, 100, 50]), np.array([10, 255, 255])) + \
                       cv2.inRange(hsv, np.array([170, 100, 50]), np.array([180, 255, 255]))

            if cv2.countNonZero(red_mask) >= PIXEL_THRESHOLD:
                cv2.imwrite("red_combined.png", merged_red)
                print("Saved: red_combined.png")
            else:
                print("Skipped: red_combined (Too few pixels)")

        print("Segmentation complete. Check the output images.")

        for i in range(len(mainColours)):
            if mainColours[i] == "red":
                self.red1.setEnabled(True)
            if mainColours[i] == "red2":
                self.red2.setEnabled(True)
            if mainColours[i] == "yellow":
                self.yellow.setEnabled(True)
            if mainColours[i] == "orange":
                self.orange.setEnabled(True)
            if mainColours[i] == "light_green":
                self.lGreen.setEnabled(True)
            if mainColours[i] == "medium_green":
                self.mGreen.setEnabled(True)
            if mainColours[i] == "dark_green":
                self.dGreen.setEnabled(True)
            if mainColours[i] == "cyan":
                self.cyan.setEnabled(True)
            if mainColours[i] == "teal":
                self.teal.setEnabled(True)
            if mainColours[i] == "light_blue":
                self.lBlue.setEnabled(True)
            if mainColours[i] == "medium_blue":
                self.mBlue.setEnabled(True)
            if mainColours[i] == "dark_blue":
                self.dBlue.setEnabled(True)
            if mainColours[i] == "purple":
                self.purple.setEnabled(True)
            if mainColours[i] == "magenta":
                self.magenta.setEnabled(True)
            if mainColours[i] == "black":
                self.black.setEnabled(True)
            if mainColours[i] == "gray":
                self.gray.setEnabled(True)
            if mainColours[i] == "white":
                self.white.setEnabled(True)


    def reconstructImg(self):
        imgPath = self.screen1.global_image
        original = cv2.imread(imgPath)
        if original is None:
            print("Error: Unable to read the original image.")
            return

        reconstructed = np.zeros_like(original)

        selected_colors = []

        checkboxes = {
            "red": self.red1, "red2": self.red2, "yellow": self.yellow, "orange": self.orange,
            "light_green": self.lGreen, "medium_green": self.mGreen, "dark_green": self.dGreen,
            "cyan": self.cyan, "teal": self.teal, "light_blue": self.lBlue, "medium_blue": self.mBlue,
            "dark_blue": self.dBlue, "purple": self.purple, "magenta": self.magenta,
            "black": self.black, "gray": self.gray, "white": self.white,
        }

        for name, checkbox in checkboxes.items():
            if checkbox.isChecked():
                selected_colors.append(name)

        print("Reconstructing with:", selected_colors)

        for name in selected_colors:
            if name in self.segmented_images:
                seg_img = self.segmented_images[name]

                if seg_img.shape[:2] != reconstructed.shape[:2]:
                    seg_img = cv2.resize(seg_img, (reconstructed.shape[1], reconstructed.shape[0]))

                reconstructed = cv2.bitwise_or(reconstructed, seg_img)

        cv2.imwrite("reconstructed.png", reconstructed)
        print("Reconstructed image saved as reconstructed.png")





    def generateImg(self):
        imgPath = self.screen1.global_image
        original = cv2.imread(imgPath)
        if original is None:
            print("Error: Unable to read the original image.")
            return

        selected_colors = []

        checkboxes = {
            "red": self.red1, "red2": self.red2, "yellow": self.yellow, "orange": self.orange,
            "light_green": self.lGreen, "medium_green": self.mGreen, "dark_green": self.dGreen,
            "cyan": self.cyan, "teal": self.teal, "light_blue": self.lBlue, "medium_blue": self.mBlue,
            "dark_blue": self.dBlue, "purple": self.purple, "magenta": self.magenta,
            "black": self.black, "gray": self.gray, "white": self.white,
        }

        for name, checkbox in checkboxes.items():
            if checkbox.isChecked():
                selected_colors.append(name)

        self.colourBridge = selected_colors

        for name in selected_colors:
            img_file = f"{name}.png"
            print(img_file)
            if not os.path.exists(img_file):
                print(f"Warning: Image {img_file} not found. Skipping.")
                continue

            try:
                pattern = self.contours_to_embroidery_with_bridging(img_file, 4, 3.0)
                if pattern is None:
                    print(f"Warning: No valid pattern for {img_file}. Skipping.")
                    continue

                pe.write_pes(pattern, f"{name}Bridge.pes")
                pe.write_png(pattern, f"{name}Bridge.png")
            except Exception as e:
                print(f"Critical Error processing {img_file}: {e}")
                continue

    def contours_to_embroidery_with_bridging(self, image_path, bridge_spacing, scale_factor):
        print(f"Processing {image_path}")

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print("Error: Unable to load image.")
            return None

        blurred = cv2.GaussianBlur(image, (7, 7), 0)
        edges = cv2.Canny(blurred, 50, 150)

        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print("Error: No contours detected.")
            return None
        print(f"Contours found: {len(contours)}")

        pattern = pe.EmbPattern()

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            print(f"Processing contour at (x={x}, y={y}, w={w}, h={h})")

            outline_stitches = [(pt[0][0] * scale_factor, pt[0][1] * scale_factor) for pt in contour]
            pattern.add_block(outline_stitches)

            previous_end = None
            for row in range(y, y + h, bridge_spacing):
                scanline_points = []

                for col in range(x, x + w):
                    if cv2.pointPolygonTest(contour, (col, row), False) >= 0:
                        scanline_points.append((col, row))

                segment_start = None
                for i in range(len(scanline_points) - 1):
                    if segment_start is None:
                        segment_start = scanline_points[i]

                    if scanline_points[i + 1][0] - scanline_points[i][0] > 1:
                        segment_end = scanline_points[i]


                        if segment_start and segment_end and segment_start != segment_end:
                            first_scaled = (segment_start[0] * scale_factor, segment_start[1] * scale_factor)
                            last_scaled = (segment_end[0] * scale_factor, segment_end[1] * scale_factor)

                            if previous_end:
                                pattern.add_stitch_absolute(pe.JUMP, previous_end[0], previous_end[1])

                            pattern.add_stitch_absolute(pe.STITCH, first_scaled[0], first_scaled[1])
                            pattern.add_stitch_absolute(pe.STITCH, last_scaled[0], last_scaled[1])

                            previous_end = last_scaled

                        segment_start = None

                if segment_start:
                    segment_end = scanline_points[-1]
                    first_scaled = (segment_start[0] * scale_factor, segment_start[1] * scale_factor)
                    last_scaled = (segment_end[0] * scale_factor, segment_end[1] * scale_factor)

                    if previous_end:
                        pattern.add_stitch_absolute(pe.JUMP, previous_end[0], previous_end[1])

                    pattern.add_stitch_absolute(pe.STITCH, first_scaled[0], first_scaled[1])
                    pattern.add_stitch_absolute(pe.STITCH, last_scaled[0], last_scaled[1])

                    previous_end = last_scaled

        pattern.end()
        print("Embroidery pattern generated and saved.")

        return pattern

    def reconstructGenImg(self):
        print("placehold")



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
screen4 = Screen4(stacked_widget, screen1)

stacked_widget.addWidget(main_window)
stacked_widget.addWidget(screen1)
stacked_widget.addWidget(screen2)
stacked_widget.addWidget(screen3)
stacked_widget.addWidget(screen4)


stacked_widget.setCurrentIndex(0)
stacked_widget.show()
sys.exit(app.exec())