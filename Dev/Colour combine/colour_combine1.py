import os
import sys

import networkx as nx
import pyembroidery as pe
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QStackedWidget, QLabel, QMainWindow, \
    QFileDialog, QPlainTextEdit, QCheckBox
from PyQt6 import uic
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
from skimage.measure import find_contours
import svgwrite
from scipy.spatial import KDTree
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix


# Class for the main opening window for the program, the user will be able to make a new design or load an old one here
class MainWindow(QMainWindow):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget

        # Load .ui file
        uic.loadUi("MainMenu2.ui", self)
        self.setFixedSize(1200, 700)

        # Buttons for creating new and opening a saved Design
        self.button1 = self.findChild(QPushButton, "createNew")
        self.button2 = self.findChild(QPushButton, "openSaved")
        # Button actions
        self.button1.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        self.button2.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(2))


# Class for the upload image window for the program, the user will be able to upload their images here or go back to main
class Screen1(QMainWindow):
    def __init__(self, stacked_widget):
        super().__init__()
        self.stacked_widget = stacked_widget

        # Load the .ui for this window
        uic.loadUi("CreateNew2.ui", self)
        self.setFixedSize(1200, 700)

        # Define buttons in the window
        self.backButton = self.findChild(QPushButton, "BackToMain")
        self.uploadButton = self.findChild(QPushButton, "UploadImage")
        self.imageLabel = self.findChild(QLabel, "imageLabel")
        self.NextButton = self.findChild(QPushButton, "Next")

        # If X feature exists then allow click condition
        if self.backButton:
            # Bacl to main
            self.backButton.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(0))
        if self.uploadButton:
            # Upload image
            self.uploadButton.clicked.connect(self.upload_image)
        if self.imageLabel:
            # Image Label for frame
            self.imageLabel.setText("No Image Selected")
        if self.NextButton:
            # Proceed to next window (currently disabled in this state)
            self.NextButton.clicked.connect((lambda: self.stacked_widget.setCurrentIndex(3)))

    # Upload image function
    def upload_image(self):
        # Open file explorer and let user upload file path
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")

        # If filepath exists then set it "globally" and load it into the image frame in the UI then allow next to be pressed
        if file_path:
            self.global_image = file_path
            self.pixmap = QPixmap(file_path)
            self.imageLabel.setPixmap(self.pixmap)
            self.imageLabel.setScaledContents(True)
            self.NextButton.setEnabled(True)
            print(f"Image loaded: {file_path}")


# Class for defining the outline of the image, user can do this manually or automatically.
class Screen3(QMainWindow):
    def __init__(self, stacked_widget, screen1):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.screen1 = screen1

        # Load from .ui file
        uic.loadUi("Edges2.ui", self)
        self.setFixedSize(1200, 700)

        # Define UI features
        self.backButton = self.findChild(QPushButton, "BackToMain")
        self.autoFind = self.findChild(QPushButton, "Auto")
        self.imageLabel = self.findChild(QLabel, "imageLabel")
        self.manualFind = self.findChild(QPushButton, "Manual")
        self.smallThreshIn = self.findChild(QPlainTextEdit, "smallThreshIn")
        self.nextButton = self.findChild(QPushButton, "next")

        # If X feature exists then process input logic
        if self.backButton:
            # Back to prev
            self.backButton.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(1))
        if self.autoFind:
            # Auto find edges
            self.autoFind.clicked.connect(self.vectorise)
        if self.imageLabel:
            # Image frame and label
            self.imageLabel.setText("No Image Selected")
        if self.manualFind:
            # Manually find edges
            self.manualFind.clicked.connect(self.cannyTrackbars)
        if not self.smallThreshIn:
            # This is the text box in the UI, it defines the threshold that small contours will be ignored
            print("Error: smallThreshIn QPlainTextEdit not found.")
        if self.nextButton:
            # Go to Next screen, currently disabled
            self.nextButton.clicked.connect(lambda: self.stacked_widget.setCurrentIndex(4))

    # Get the text from the text box for thresholding small contours
    def get_small_thresh(self):
        # Default to 6
        if not self.smallThreshIn:
            return 6

        # Get text and assign threshold, if invalid then default to 6
        text = self.smallThreshIn.toPlainText().strip()
        try:
            return int(text)
        except ValueError:
            try:
                return float(text)
            except ValueError:
                print("Invalid input. Using default threshold (6).")
                return 6

    # Auto find edges function
    def vectorise(self):
        # Prevents crashes
        try:
            # Obtain threshold, define a scale factor and max stitch length
            smallthresh = self.get_small_thresh()
            scale_factor = 2.0
            max_stitch_length = 10.0

            # Obtain image through screen1 TODO change this condition
            image_path = self.screen1.global_image

            # Read image into grey scale
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            # Set next button as enabled TODO move just incase errors are made
            self.nextButton.setEnabled(True)

            # Canny Edge detection
            edges = cv2.Canny(image, 50, 150)
            # Pull contours using Tree, seems to work best
            contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Create a blank canvas the same size as original image and draw found edges, Black and White
            contour_image = np.zeros_like(image)
            cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 1)

            # Save image as edges output
            edge_image_path = "edges_output.png"
            cv2.imwrite(edge_image_path, contour_image)

            # Embroidery fileName for outlines #TODO change eventually
            outfile = "emb1"

            pattern = self.writeStitches(contours, smallthresh, scale_factor, image, max_stitch_length)
            # Write .pes and .png files for p1
            pe.write_pes(pattern, f"{outfile}.pes")
            pe.write_png(pattern, f"{outfile}.png")

            # Load emb1.png into image frame on UI
            pixmap = QPixmap(f"{outfile}.png")
            pixmap = pixmap.scaled(801, 521, Qt.AspectRatioMode.KeepAspectRatio)
            self.imageLabel.setPixmap(pixmap)
            self.imageLabel.setScaledContents(True)

        except Exception as e:
            # Error handling and crash prevention
            print(f"An error occurred: {e}")

    def cannyTrackbars(self):
        # Set global image to local path
        image_path = self.screen1.global_image

        # Read the image
        image = cv2.imread(image_path)

        # Gray and blur
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 1.4)

        # Enable track bars
        self.lowThresh.setEnabled(True)
        self.highThresh.setEnabled(True)

        # Detect changes and update edges
        self.lowThresh.valueChanged.connect(lambda: self.update_edges(blurred))
        self.highThresh.valueChanged.connect(lambda: self.update_edges(blurred))
        self.update_edges(blurred)

        # Submit for finalising edge detection
        self.submit = self.findChild(QPushButton, "Submit")
        if self.submit:
            # Write the design
            self.submit.clicked.connect(lambda: self.writeManual())
        self.submit.setEnabled(True)

    def update_edges(self, blurred):
        # Obtain values
        low = self.lowThresh.value()
        high = self.highThresh.value()

        # Obtain edges
        self.edges = cv2.Canny(blurred, low, high)
        edge_image_path = "edges_output.png"

        # Write image
        cv2.imwrite(edge_image_path, self.edges)

        # Load image into frame TODO change this
        pixmap = QPixmap(edge_image_path)
        if pixmap.isNull():
            print("Error: Failed to load the saved image.")
            return
        pixmap = pixmap.scaled(801, 521, Qt.AspectRatioMode.KeepAspectRatio)
        self.imageLabel.setPixmap(pixmap)
        self.imageLabel.setScaledContents(True)

    def writeManual(self):
        # Define dependencies
        max_stitch_length = 10
        outfile = "emb1"
        scale_factor = 2.0
        smallthresh = self.get_small_thresh()
        image = cv2.imread("edges_output.png", cv2.IMREAD_GRAYSCALE)
        contours = self.readImg(image)
        # Get pattern
        pattern = self.writeStitches(contours, smallthresh, scale_factor, image, max_stitch_length)
        # Write files
        pe.write_pes(pattern, f"{outfile}.pes")
        pe.write_png(pattern, f"{outfile}.png")

        # Load to label TODO change this
        pixmap = QPixmap(f"{outfile}.png")
        if pixmap.isNull():
            print("Error: Failed to load the saved image.")
            return
        pixmap = pixmap.scaled(801, 521, Qt.AspectRatioMode.KeepAspectRatio)
        self.imageLabel.setPixmap(pixmap)
        self.imageLabel.setScaledContents(True)

    def readImg(self, image):
        # Read image and collect contours TODO add this for vectorise in same funct
        dilated = cv2.dilate(self.edges, (3, 3), iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contour_image = np.zeros_like(image)
        cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 1)

        edge_image_path = "edges_output.png"
        cv2.imwrite(edge_image_path, contour_image)

        return contours

    def writeStitches(self, contours, smallthresh, scale_factor, image, max_stitch_length):
        # Create the Pattern as p1
        p1 = pe.EmbPattern()

        # The path of each contour found in the image
        contour_paths = []
        for c in contours:
            # Each contour must exceed the threshold
            if len(c) < smallthresh:
                continue

            # Define all stitches needed to make that contour with subdivisions
            stitches = []
            for i in range(len(c) - 1):
                start_pt = (c[i][0][0] * scale_factor, c[i][0][1] * scale_factor)
                end_pt = (c[i + 1][0][0] * scale_factor, c[i + 1][0][1] * scale_factor)

                # Subdivide long segments
                subdivided = self.subdivide_segment(start_pt, end_pt, max_stitch_length)
                stitches.extend(subdivided[:-1])  # Exclude last point to prevent duplicates

            stitches.append((c[-1][0][0] * scale_factor, c[-1][0][1] * scale_factor))  # Add last point
            contour_paths.append(stitches)

        # Define the height and width of the image
        height, width = image.shape

        # Define all corners of the image
        corner_stitches = [
            (0, 0),
            (width * scale_factor, 0),
            (width * scale_factor, height * scale_factor),
            (0, height * scale_factor)
        ]

        # Stitches need to be organized to prevent unnecessary jumps
        ordered_stitches = []
        current_pos = (0, 0)

        # Jump between all corners and make a stitch, defining the bounding box
        for corner in corner_stitches:
            ordered_stitches.append((corner[0], corner[1], "JUMP"))
            ordered_stitches.append((corner[0], corner[1]))

        while contour_paths:
            # Find the closest contour to the current position
            next_contour = min(contour_paths,
                               key=lambda path: np.linalg.norm(np.array(path[0]) - np.array(current_pos)))
            contour_paths.remove(next_contour)

            if ordered_stitches:
                ordered_stitches.append((next_contour[0][0], next_contour[0][1], "JUMP"))

            ordered_stitches.extend(next_contour)
            current_pos = next_contour[-1]

        stitches = 0
        # Add stitches to the pattern
        for stitch in ordered_stitches:
            if isinstance(stitch, tuple) and len(stitch) == 3 and stitch[2] == "JUMP":
                p1.add_stitch_absolute(pe.JUMP, stitch[0], stitch[1])
                stitches += 1
            else:
                p1.add_stitch_absolute(pe.STITCH, stitch[0], stitch[1])
                stitches += 1

        # End the pattern
        p1.end()
        print("TOTAL Stitches:", stitches)
        return p1

    def subdivide_segment(self, start_pt, end_pt, max_stitch_length):
        (x1, y1) = start_pt
        (x2, y2) = end_pt
        distance = math.hypot(x2 - x1, y2 - y1)

        if distance <= max_stitch_length:
            return [start_pt, end_pt]

        num_segments = int(math.ceil(distance / max_stitch_length))
        return [(x1 + t * (x2 - x1), y1 + t * (y2 - y1)) for t in np.linspace(0, 1, num_segments + 1)]


# Class for obtaining the shading of images
class Screen4(QMainWindow):
    def __init__(self, stacked_widget, screen1):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.screen1 = screen1

        # Load UI from .ui file
        uic.loadUi("Shading2.ui", self)
        self.setFixedSize(1200, 700)
        self.imageLabel = self.findChild(QLabel, "imageLabel")

        # Button for segmenting the colours of the image
        self.auto = self.findChild(QPushButton, "Auto")
        if self.auto:
            self.auto.clicked.connect(self.segmentColours)

        self.combine = self.findChild(QPushButton, "Combine")
        if self.combine:
            self.combine.clicked.connect(self.combineSelectedColours)

        # All checkboxes for each colour
        self.red1 = self.findChild(QCheckBox, "red1")
        self.red2 = self.findChild(QCheckBox, "red2")
        self.orange = self.findChild(QCheckBox, "orange")
        self.yellow = self.findChild(QCheckBox, "yellow")
        self.light_yellow = self.findChild(QCheckBox, "light_yellow")
        self.lGreen = self.findChild(QCheckBox, "lGreen")
        self.green = self.findChild(QCheckBox, "green")
        self.dGreen = self.findChild(QCheckBox, "dGreen")
        self.cyan = self.findChild(QCheckBox, "cyan")
        self.light_cyan = self.findChild(QCheckBox, "light_cyan")
        self.teal = self.findChild(QCheckBox, "teal")
        self.lBlue = self.findChild(QCheckBox, "lBlue")
        self.blue = self.findChild(QCheckBox, "mBlue")
        self.blue2 = self.findChild(QCheckBox, "blue2")
        self.dBlue = self.findChild(QCheckBox, "dBlue")
        self.purple = self.findChild(QCheckBox, "purple")
        self.magenta = self.findChild(QCheckBox, "magenta")
        self.pink = self.findChild(QCheckBox, "pink")
        self.black = self.findChild(QCheckBox, "black")
        self.gray = self.findChild(QCheckBox, "gray")
        self.white = self.findChild(QCheckBox, "white")
        self.brown = self.findChild(QCheckBox, "brown")
        self.beige = self.findChild(QCheckBox, "beige")
        self.maroon = self.findChild(QCheckBox, "maroon")
        self.olive = self.findChild(QCheckBox, "olive")
        self.turquoise = self.findChild(QCheckBox, "turquoise")
        self.indigo = self.findChild(QCheckBox, "indigo")
        self.lavender = self.findChild(QCheckBox, "lavender")
        self.peach = self.findChild(QCheckBox, "peach")
        self.tan = self.findChild(QCheckBox, "tan")

        # All checkboxes for each colour combination option
        self.red1C = self.findChild(QCheckBox, "red1_2")
        self.red2C = self.findChild(QCheckBox, "red2_2")
        self.orangeC = self.findChild(QCheckBox, "orange_2")
        self.yellowC = self.findChild(QCheckBox, "yellow_2")
        self.light_yellowC = self.findChild(QCheckBox, "light_yellow_2")
        self.lGreenC = self.findChild(QCheckBox, "lGreen_2")
        self.greenC = self.findChild(QCheckBox, "green_2")
        self.dGreenC = self.findChild(QCheckBox, "dGreen_2")
        self.cyanC = self.findChild(QCheckBox, "cyan_2")
        self.light_cyanC = self.findChild(QCheckBox, "light_cyan_2")
        self.tealC = self.findChild(QCheckBox, "teal_2")
        self.lBlueC = self.findChild(QCheckBox, "lBlue_2")
        self.blueC = self.findChild(QCheckBox, "mBlue_2")
        self.blue2C = self.findChild(QCheckBox, "blue2_2")
        self.dBlueC = self.findChild(QCheckBox, "dBlue_2")
        self.purpleC = self.findChild(QCheckBox, "purple_2")
        self.magentaC = self.findChild(QCheckBox, "magenta_2")
        self.pinkC = self.findChild(QCheckBox, "pink_2")
        self.blackC = self.findChild(QCheckBox, "black_2")
        self.grayC = self.findChild(QCheckBox, "gray_2")
        self.whiteC = self.findChild(QCheckBox, "white_2")
        self.brownC = self.findChild(QCheckBox, "brown_2")
        self.beigeC = self.findChild(QCheckBox, "beige_2")
        self.maroonC = self.findChild(QCheckBox, "maroon_2")
        self.oliveC = self.findChild(QCheckBox, "olive_2")
        self.turquoiseC = self.findChild(QCheckBox, "turquoise_2")
        self.indigoC = self.findChild(QCheckBox, "indigo_2")
        self.lavenderC = self.findChild(QCheckBox, "lavender_2")
        self.peachC = self.findChild(QCheckBox, "peach_2")
        self.tanC = self.findChild(QCheckBox, "tan_2")

        self.combined1 = self.findChild(QCheckBox, "combined_1")
        self.combined2 = self.findChild(QCheckBox, "combined_2")
        self.combined3 = self.findChild(QCheckBox, "combined_3")
        self.combined4 = self.findChild(QCheckBox, "combined_4")



        # arrays and dicts for segmented images and each chosen bridge
        self.segmented_images = {}
        self.colourBridge = []


        # Button for user to reconstruct the image
        self.reconstruct = self.findChild(QPushButton, "reconstruct")
        if self.reconstruct:
            self.reconstruct.clicked.connect(self.reconstructImg)

        # Button for generating the final shading patterns
        self.generate = self.findChild(QPushButton, "generate")
        if self.generate:
            self.generate.clicked.connect(self.generateImg)

        # Reconstruct the final image
        self.finalise = self.findChild(QPushButton, "finalise")
        if self.finalise:
            self.finalise.clicked.connect(self.reconstructGenImg)

    def segmentColours(self):
        # Make sure this function can see the image from the start
        if not hasattr(self.screen1, "global_image") or not self.screen1.global_image:
            print("Error: No image loaded.")
            return
        # Load the image locally
        imgPath = self.screen1.global_image
        image = cv2.imread(imgPath)
        if image is None:
            print("Error: Unable to read the image.")
            return

        # Load the image into the BGR space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Define colours
        color_ranges = {
            "red": [(0, 70, 50), (10, 255, 255)],
            "red2": [(160, 70, 50), (180, 255, 255)],
            "orange": [(11, 100, 100), (25, 255, 255)],
            "yellow": [(26, 100, 100), (35, 255, 255)],
            "light_yellow": [(36, 30, 100), (50, 255, 255)],
            "light_green": [(40, 30, 100), (70, 255, 255)],
            "green": [(70, 30, 100), (85, 255, 255)],
            "dark_green": [(86, 30, 50), (100, 255, 200)],
            "cyan": [(85, 30, 100), (100, 255, 255)],
            "light_cyan": [(101, 30, 100), (110, 255, 255)],
            "teal": [(111, 30, 100), (130, 255, 255)],
            "light_blue": [(131, 30, 100), (150, 255, 255)],
            "blue": [(151, 30, 100), (170, 255, 255)],
            "blue2": [(0, 30, 50), (10, 255, 200)],
            "dark_blue": [(171, 30, 50), (180, 255, 200)],
            "purple": [(120, 30, 50), (140, 255, 255)],
            "magenta": [(141, 30, 50), (160, 255, 255)],
            "pink": [(161, 30, 100), (170, 255, 255)],
            "black": [(0, 0, 0), (180, 255, 30)],
            "gray": [(0, 0, 31), (180, 30, 200)],
            "white": [(0, 0, 201), (180, 30, 255)],
            "brown": [(10, 50, 20), (20, 255, 200)],
            "beige": [(21, 30, 100), (30, 150, 255)],
            "maroon": [(0, 50, 20), (10, 255, 100)],
            "olive": [(25, 30, 20), (35, 255, 100)],
            "turquoise": [(80, 30, 100), (90, 255, 255)],
            "indigo": [(110, 30, 20), (130, 255, 150)],
            "lavender": [(130, 30, 100), (150, 255, 255)],
            "peach": [(10, 30, 100), (20, 255, 255)],
            "tan": [(20, 30, 100), (30, 255, 200)],
        }

        # Min pixels needed to segment colours
        # TODO let user choose this value
        PIXEL_THRESHOLD = 500

        # Main segmented colours
        mainColours = []

        # Cycle through each main colour
        for name, (lower, upper) in color_ranges.items():
            # Upper and lower colour bands
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")

            # Create image frame
            mask = cv2.inRange(hsv, lower, upper)
            result = cv2.bitwise_and(image, image, mask=mask)

            # Create a count for the non black pixels in the segment
            # TODO account for black pixels in dark images
            non_black_pixels = cv2.countNonZero(mask)

            # If the count is creater than the threshhold then same the colour
            if non_black_pixels >= PIXEL_THRESHOLD:
                output_path = f"{name}.png"
                cv2.imwrite(output_path, result)
                print(f"Saved: {output_path} ({non_black_pixels} pixels)")
                mainColours.append(name)
                self.segmented_images[name] = result
            else:
                print(f"Skipped: {name} (Only {non_black_pixels} pixels)")

        self.combine.setEnabled(True)

        for color in mainColours:
            if color == "red":
                self.red1.setEnabled(True)
                self.red1C.setEnabled(True)
            elif color == "red2":
                self.red2.setEnabled(True)
                self.red2C.setEnabled(True)
            elif color == "orange":
                self.orange.setEnabled(True)
                self.orangeC.setEnabled(True)
            elif color == "yellow":
                self.yellow.setEnabled(True)
                self.yellowC.setEnabled(True)
            elif color == "light_yellow":
                self.light_yellow.setEnabled(True)
                self.light_yellowC.setEnabled(True)
            elif color == "light_green":
                self.lGreen.setEnabled(True)
                self.lGreenC.setEnabled(True)
            elif color == "green":
                self.green.setEnabled(True)
                self.greenC.setEnabled(True)
            elif color == "dark_green":
                self.dGreen.setEnabled(True)
                self.dGreenC.setEnabled(True)
            elif color == "cyan":
                self.cyan.setEnabled(True)
                self.cyanC.setEnabled(True)
            elif color == "light_cyan":
                self.light_cyan.setEnabled(True)
                self.light_cyanC.setEnabled(True)
            elif color == "teal":
                self.teal.setEnabled(True)
                self.tealC.setEnabled(True)
            elif color == "light_blue":
                self.lBlue.setEnabled(True)
                self.lBlueC.setEnabled(True)
            elif color == "blue":
                self.blue.setEnabled(True)
                self.blueC.setEnabled(True)
            elif color == "blue2":
                self.blue2.setEnabled(True)
                self.blue2C.setEnabled(True)
            elif color == "dark_blue":
                self.dBlue.setEnabled(True)
                self.dBlueC.setEnabled(True)
            elif color == "purple":
                self.purple.setEnabled(True)
                self.purpleC.setEnabled(True)
            elif color == "magenta":
                self.magenta.setEnabled(True)
                self.magentaC.setEnabled(True)
            elif color == "pink":
                self.pink.setEnabled(True)
                self.pinkC.setEnabled(True)
            elif color == "black":
                self.black.setEnabled(True)
                self.blackC.setEnabled(True)
            elif color == "gray":
                self.gray.setEnabled(True)
                self.grayC.setEnabled(True)
            elif color == "white":
                self.white.setEnabled(True)
                self.whiteC.setEnabled(True)
            elif color == "brown":
                self.brown.setEnabled(True)
                self.brownC.setEnabled(True)
            elif color == "beige":
                self.beige.setEnabled(True)
                self.beigeC.setEnabled(True)
            elif color == "maroon":
                self.maroon.setEnabled(True)
                self.maroonC.setEnabled(True)
            elif color == "olive":
                self.olive.setEnabled(True)
                self.oliveC.setEnabled(True)
            elif color == "turquoise":
                self.turquoise.setEnabled(True)
                self.turquoiseC.setEnabled(True)
            elif color == "indigo":
                self.indigo.setEnabled(True)
                self.indigoC.setEnabled(True)
            elif color == "lavender":
                self.lavender.setEnabled(True)
                self.lavenderC.setEnabled(True)
            elif color == "peach":
                self.peach.setEnabled(True)
                self.peachC.setEnabled(True)
            elif color == "tan":
                self.tan.setEnabled(True)
                self.tanC.setEnabled(True)
        print("Checkboxes enabled for:", mainColours)

    def combineSelectedColours(self):
        imgPath = self.screen1.global_image
        original = cv2.imread(imgPath)
        if original is None:
            print("Error: Unable to read the original image.")
            return

        reconstructed = np.zeros_like(original)

        selected_colors2 = []

        checkboxes2 = {
            "red": self.red1C, "red2": self.red2C, "orange": self.orangeC, "yellow": self.yellowC,
            "light_yellow": self.light_yellowC, "light_green": self.lGreenC, "green": self.greenC,
            "dark_green": self.dGreenC, "cyan": self.cyanC, "light_cyan": self.light_cyanC,
            "teal": self.tealC, "light_blue": self.lBlueC, "blue": self.blueC, "blue2": self.blue2C,
            "dark_blue": self.dBlueC, "purple": self.purpleC, "magenta": self.magentaC,
            "pink": self.pinkC, "black": self.blackC, "gray": self.grayC, "white": self.whiteC,
            "brown": self.brownC, "beige": self.beigeC, "maroon": self.maroonC, "olive": self.oliveC,
            "turquoise": self.turquoiseC, "indigo": self.indigoC, "lavender": self.lavenderC,
            "peach": self.peachC, "tan": self.tanC,
        }

        for name, checkbox in checkboxes2.items():
            if checkbox.isChecked():
                selected_colors2.append(name)
        print("Combining", selected_colors2)

        for name in selected_colors2:
            if name in self.segmented_images:
                seg_img = self.segmented_images[name]

                if seg_img.shape[:2] != reconstructed.shape[:2]:
                    seg_img = cv2.resize(seg_img, (reconstructed.shape[1], reconstructed.shape[0]))

                reconstructed = cv2.bitwise_or(reconstructed, seg_img)

        if not hasattr(self, "current_combination"):
            self.current_combination = 0

        if self.current_combination >= 4:
            print("Maximum of 4 combinations reached. Overwriting the first slot.")
            self.current_combination = 0

        combination_name = f"combined{self.current_combination + 1}"
        output_filename = f"{combination_name}.png"

        cv2.imwrite(output_filename, reconstructed)
        print(f"Reconstructed image saved as {output_filename}")

        self.segmented_images[combination_name] = reconstructed

        combined_checkboxes = [self.combined1, self.combined2, self.combined3, self.combined4]

        if self.current_combination < len(combined_checkboxes):
            combined_checkboxes[self.current_combination].setEnabled(True)

        if self.current_combination == 0:
            pixmap = QPixmap("combined1.png")
        if self.current_combination == 1:
            pixmap = QPixmap("combined2.png")
        if self.current_combination == 2:
            pixmap = QPixmap("combined3.png")
        if self.current_combination == 3:
            pixmap = QPixmap("combined4.png")
        if pixmap.isNull():
            print("Error: Failed to load the saved image.")
            return
        pixmap = pixmap.scaled(801, 521, Qt.AspectRatioMode.KeepAspectRatio)
        self.imageLabel.setPixmap(pixmap)
        self.imageLabel.setScaledContents(True)
        self.current_combination += 1




    def reconstructImg(self):
        imgPath = self.screen1.global_image
        original = cv2.imread(imgPath)
        if original is None:
            print("Error: Unable to read the original image.")
            return

        reconstructed = np.zeros_like(original)

        selected_colors = []

        checkboxes = {
            "red": self.red1, "red2": self.red2, "orange": self.orange, "yellow": self.yellow,
            "light_yellow": self.light_yellow, "light_green": self.lGreen, "green": self.green,
            "dark_green": self.dGreen, "cyan": self.cyan, "light_cyan": self.light_cyan,
            "teal": self.teal, "light_blue": self.lBlue, "blue": self.blue, "blue2": self.blue2,
            "dark_blue": self.dBlue, "purple": self.purple, "magenta": self.magenta,
            "pink": self.pink, "black": self.black, "gray": self.gray, "white": self.white,
            "brown": self.brown, "beige": self.beige, "maroon": self.maroon, "olive": self.olive,
            "turquoise": self.turquoise, "indigo": self.indigo, "lavender": self.lavender,
            "peach": self.peach, "tan": self.tan, "combined1": self.combined1, "combined2": self.combined2,
            "combined3": self.combined3, "combined4": self.combined4,
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
        pixmap = QPixmap("reconstructed.png")
        if pixmap.isNull():
            print("Error: Failed to load the saved image.")
            return
        pixmap = pixmap.scaled(801, 521, Qt.AspectRatioMode.KeepAspectRatio)
        self.imageLabel.setPixmap(pixmap)
        self.imageLabel.setScaledContents(True)

    def generateImg(self):
        imgPath = self.screen1.global_image
        original = cv2.imread(imgPath)
        if original is None:
            print("Error: Unable to read the original image.")
            return

        selected_colors = []

        checkboxes = {
            "red": self.red1, "red2": self.red2, "orange": self.orange, "yellow": self.yellow,
            "light_yellow": self.light_yellow, "light_green": self.lGreen, "green": self.green,
            "dark_green": self.dGreen, "cyan": self.cyan, "light_cyan": self.light_cyan,
            "teal": self.teal, "light_blue": self.lBlue, "blue": self.mBlue, "blue2": self.blue2,
            "dark_blue": self.dBlue, "purple": self.purple, "magenta": self.magenta,
            "pink": self.pink, "black": self.black, "gray": self.gray, "white": self.white,
            "brown": self.brown, "beige": self.beige, "maroon": self.maroon, "olive": self.olive,
            "turquoise": self.turquoise, "indigo": self.indigo, "lavender": self.lavender,
            "peach": self.peach, "tan": self.tan, "combined1": self.combined1, "combined2": self.combined2,
            "combined3": self.combined3, "combined4": self.combined4,
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
                pattern = self.image_to_stitch_pattern(img_file, 5, 1.0, 5, 3)
                if pattern is None:
                    print(f"Warning: No valid pattern for {img_file}. Skipping.")
                    continue

                pe.write_pes(pattern, f"{name}Bridge.pes")
                pe.write_png(pattern, f"{name}Bridge.png")
            except Exception as e:
                print(f"Critical Error processing {img_file}: {e}")
                continue


    def build_mst(self, stitch_points):
        G = nx.Graph()
        for i, p1 in enumerate(stitch_points):
            for j, p2 in enumerate(stitch_points):
                if i != j:
                    distance = np.linalg.norm(np.array(p1) - np.array(p2))
                    G.add_edge(i, j, weight=distance)

        mst = nx.minimum_spanning_tree(G)
        return mst

    def traverse_mst(self, mst, start_node=0):
        path = list(nx.dfs_preorder_nodes(mst, start_node))
        return path

    def join_stitch_points(self, action_log):
        stitch_points = [(x, y) for x, y, cmd, _ in action_log if cmd == "STITCH"]

        if not stitch_points:
            return []

        # Start with the first point
        unvisited = set(range(len(stitch_points)))
        path = [0]  # Start with the first point
        unvisited.remove(0)

        while unvisited:
            current = path[-1]
            # Find the nearest unvisited point
            nearest = min(unvisited,
                          key=lambda i: np.linalg.norm(np.array(stitch_points[current]) - np.array(stitch_points[i])))

            path.append(nearest)
            unvisited.remove(nearest)

        ordered_points = [stitch_points[i] for i in path]
        return ordered_points


    def image_to_stitch_pattern(self, image_path, bridge_spacing, scale_factor, max_stitch_length, kernel_size):
        # Function to subdivide long stitch segments
        def subdivide_segment(start_pt, end_pt, max_stitch_length):
            (x1, y1) = start_pt
            (x2, y2) = end_pt
            distance = math.hypot(x2 - x1, y2 - y1)
            if distance <= max_stitch_length:
                return [start_pt, end_pt]
            num_segments = int(math.ceil(distance / max_stitch_length))
            return [(x1 + t * (x2 - x1), y1 + t * (y2 - y1)) for t in np.linspace(0, 1, num_segments + 1)]

        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Unable to load image.")
            return None
        max_jump_length_factor = 1.5
        # Convert image to RGB and create a binary mask (non-black pixels = 1, black = 0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.any(image != [0, 0, 0], axis=-1).astype(np.uint8)
        height, width = mask.shape

        # Store action log for later MST processing
        action_log = []
        direction = True  # Left to Right = True, Right to Left = False

        for y in range(0, height, bridge_spacing):
            inside_shape = False
            start_x = None
            previous_end = None
            scan_x_range = range(0, width) if direction else range(width - 1, -1, -1)

            for x in scan_x_range:
                kernel = mask[max(0, y - kernel_size // 2):min(height, y + kernel_size // 2 + 1),
                         max(0, x - kernel_size // 2):min(width, x + kernel_size // 2 + 1)]

                non_black_count = np.sum(kernel)
                threshold = kernel.size // 2

                if non_black_count > threshold:
                    if not inside_shape:
                        start_x = x
                        inside_shape = True
                else:
                    if inside_shape and start_x is not None:
                        first_scaled = (start_x * scale_factor, y * scale_factor)
                        last_scaled = (x * scale_factor, y * scale_factor)

                        stitch_points = subdivide_segment(first_scaled, last_scaled, max_stitch_length)
                        for pt in stitch_points:
                            action_log.append((pt[0], pt[1], "STITCH", direction))

                        previous_end = stitch_points[-1]
                        inside_shape = False

            if inside_shape and start_x is not None:
                first_scaled = (start_x * scale_factor, y * scale_factor)
                last_scaled = (width * scale_factor if direction else 0, y * scale_factor)

                stitch_points = subdivide_segment(first_scaled, last_scaled, max_stitch_length)
                for pt in stitch_points:
                    action_log.append((pt[0], pt[1], "STITCH", direction))

                previous_end = stitch_points[-1]

            direction = not direction

        ordered_points = self.join_stitch_points(action_log)

        pattern = pe.EmbPattern()
        max_jump_length = max_jump_length_factor * max_stitch_length
        # This section is preventing the "bounding box problem" allowing the full image to be displayed
        # Define the height and width of the image
        h1, w1 = mask.shape  # Use mask instead of image
        # Define all corners of the image
        corner_stitches = [
            (0, 0),
            (w1 * scale_factor, 0),
            (w1 * scale_factor, h1 * scale_factor),
            (0, h1 * scale_factor)
        ]

        # Jump between all corners and make a stitch, not visible on final design just defining extremes
        for corner in corner_stitches:
            # Jump to the corner
            pattern.add_stitch_absolute(pe.JUMP, corner[0], corner[1])
            # Make a stitch at the corner
            pattern.add_stitch_absolute(pe.STITCH, corner[0], corner[1])

        for i, (x, y) in enumerate(ordered_points):
            if i == 0:
                pattern.add_stitch_absolute(pe.JUMP, x, y)
            else:
                prev_x, prev_y = ordered_points[i - 1]
                jump_distance = math.hypot(prev_x - x, prev_y - y)

                if jump_distance > max_jump_length:
                    pattern.add_command(pe.TRIM)
                    pattern.add_stitch_absolute(pe.JUMP, x, y)
                pattern.add_stitch_absolute(pe.STITCH, x, y)

        pattern.end()  # Finalize the pattern


        return pattern  # Return the properly ordered embroidery pattern


    def reconstructGenImg(self):
        print("Selected Colors:", self.colourBridge)

        # Load the original image to determine canvas size
        imgPath = self.screen1.global_image
        original = cv2.imread(imgPath)

        if original is None:
            print("Error: Unable to read the original image.")
            return

        # Create a blank canvas (white) with the same size as the original image
        height, width = original.shape[:2]
        reconstructed = np.ones((height, width, 3), dtype=np.uint8) * 255

        try:
            for name in self.colourBridge:
                pes_path = os.path.join(f"{name}Bridge.pes")

                # Load PES file
                pattern = pe.read(pes_path)
                if pattern is None:
                    print(f"Error: Unable to read {pes_path}")
                    continue

                # Choose a color for this pattern (customize if needed)
                color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))

                # Draw stitches onto the image exactly as they are
                for i in range(len(pattern.stitches) - 1):
                    x1, y1, cmd1 = pattern.stitches[i]
                    x2, y2, cmd2 = pattern.stitches[i + 1]

                    # Only draw STITCH commands (ignore JUMP/TRIM commands)
                    if cmd1 == pe.STITCH and cmd2 == pe.STITCH:
                        cv2.line(reconstructed, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)

            # Save the final reconstructed image
            cv2.imwrite("reconstructedFinal.png", reconstructed)
            print("Reconstructed image saved as reconstructedFinal.png")

        except Exception as e:
            print(f"Critical Error processing PES files: {e}")


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