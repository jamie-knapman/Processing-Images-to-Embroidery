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
        self.setFixedSize(800, 450)  # TODO, Change this at some point so images are displayed correctly

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
        self.setFixedSize(800, 450)  # TODO see MainWindow

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
        self.setFixedSize(800, 450)  # TODO see MainWindow

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
            pixmap = pixmap.scaled(300, 200, Qt.AspectRatioMode.KeepAspectRatio)
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
        pixmap = pixmap.scaled(300, 200, Qt.AspectRatioMode.KeepAspectRatio)
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
        pixmap = pixmap.scaled(300, 200, Qt.AspectRatioMode.KeepAspectRatio)
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

            # Define all stitches needed to make that contour
            stitches = [(pt[0][0] * scale_factor, pt[0][1] * scale_factor) for pt in c]
            # Add stitches to the paths
            contour_paths.append(stitches)

        # This section is preventing the "bounding box problem" allowing the full image to be displayed
        # Define the height and width of the image
        height, width = image.shape
        # Define all corners of the image
        corner_stitches = [
            (0, 0),
            (width * scale_factor, 0),
            (width * scale_factor, height * scale_factor),
            (0, height * scale_factor)
        ]

        # Stitches need to be organised to prevent unnesscary jumping
        ordered_stitches = []
        current_pos = (0, 0)

        # Jump between all corners and make a stitch, not visible on final design just defining extremes
        for corner in corner_stitches:
            ordered_stitches.append((corner[0], corner[1], "JUMP"))
            ordered_stitches.append((corner[0], corner[1]))
        '''
        for contour in contour_paths:
            # If ordered stitches exists, add the contour to it
            if ordered_stitches:
                ordered_stitches.append((contour[0][0], contour[0][1], "JUMP"))

            # Add each subdivided segment to ordered stitches
            for i in range(len(contour) - 1):
                start_stitch = contour[i]
                end_stitch = contour[i + 1]

                subdivided_stitches = self.subdivide_segment(start_stitch, end_stitch, max_stitch_length)
                for stitch in subdivided_stitches:
                    ordered_stitches.append((stitch[0], stitch[1]))


        '''

        while contour_paths:
            # Find th closest contour to the current position
            # converts each contour position path[0] and currentpos into numoy arrays and calculates the distnace between the returning th min
            next_contour = min(contour_paths,
                               key=lambda path: np.linalg.norm(np.array(path[0]) - np.array(current_pos)))
            # remove since not using anymore
            contour_paths.remove(next_contour)

            if ordered_stitches:
                ordered_stitches.append((next_contour[0][0], next_contour[0][1], "JUMP"))

            ordered_stitches.extend(next_contour)
            current_pos = next_contour[-1]

        stitches = 0
        # Add stitches in array to the pattern where if the stitch is ended it jumps otherwise it moves to other stitch
        for stitch in ordered_stitches:
            if isinstance(stitch, tuple) and len(stitch) == 3 and stitch[2] == "JUMP":
                p1.add_stitch_absolute(pe.JUMP, stitch[0], stitch[1])
                stitches += 1
            else:
                p1.add_stitch_absolute(pe.STITCH, stitch[0], stitch[1])
                stitches += 1

        # End the pattern
        p1.end()
        print("TOTAL Stitches")
        print(stitches)
        return p1

    # Divide each "long stitch" into smaller segments based of max_stitch_length
    def subdivide_segment(self, start_pt, end_pt, max_stitch_length):
        (x1, y1) = start_pt  # Stasrt point
        (x2, y2) = end_pt  # End point
        distance = math.hypot(x2 - x1, y2 - y1)  # Define the distance between start and end of long stitch
        # Incorrect call to this function as dist < max len
        if distance <= max_stitch_length:
            return [start_pt, end_pt]
        # How many segments does this line need
        num_segments = int(math.ceil(distance / max_stitch_length))
        points = []
        # Loop though and append all valid "points" to the array
        for i in range(num_segments + 1):
            t = i / num_segments
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            points.append((x, y))
        # Return list of points
        return points


# Class for obtaining the shading of images
class Screen4(QMainWindow):
    def __init__(self, stacked_widget, screen1):
        super().__init__()
        self.stacked_widget = stacked_widget
        self.screen1 = screen1

        # Load UI from .ui file
        uic.loadUi("Shading2.ui", self)
        self.setFixedSize(800, 450)

        # Button for segmenting the colours of the image
        self.auto = self.findChild(QPushButton, "Auto")
        if self.auto:
            self.auto.clicked.connect(self.segmentColours)

        # All checkboxes for each colour
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
        # TODO broaden scope
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

        # Enable checkboxes for foumd colours
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
                pattern = self.image_to_stitch_pattern(img_file, 5, 1.0, 5, 3)
                if pattern is None:
                    print(f"Warning: No valid pattern for {img_file}. Skipping.")
                    continue

                pe.write_pes(pattern, f"{name}Bridge.pes")
                pe.write_png(pattern, f"{name}Bridge.png")
            except Exception as e:
                print(f"Critical Error processing {img_file}: {e}")
                continue

    '''
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
            '''

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

        # Group points by rows
        stitch_points.sort(key=lambda p: p[1])  # Sort by Y coordinate

        # Create rows
        rows = []
        current_row = []
        current_y = stitch_points[0][1]

        for point in stitch_points:
            if abs(point[1] - current_y) > 10:  # Threshold for new row
                rows.append(current_row)
                current_row = []
                current_y = point[1]
            current_row.append(point)

        if current_row:
            rows.append(current_row)

        # Alternate row traversal direction
        ordered_points = []
        for i, row in enumerate(rows):
            # Sort row points by x coordinate
            row.sort(key=lambda p: p[0])

            # Alternate traversal direction
            if i % 2 == 0:
                ordered_points.extend(row)
            else:
                ordered_points.extend(reversed(row))

        return ordered_points
    '''
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
    '''
    def plot_connected_stitches(self, ordered_points):
        plt.figure(figsize=(10, 10))

        x_vals, y_vals = zip(*ordered_points)
        plt.plot(x_vals, y_vals, marker="o", linestyle="-", color="blue")

        plt.gca().invert_yaxis()
        plt.xlabel("X Coordinate")
        plt.ylabel("Y Coordinate")
        plt.title("Connected Stitch Path (MST)")
        plt.grid(True)
        plt.show()

    def image_to_stitch_pattern(self, image_path, bridge_spacing, scale_factor, max_stitch_length, kernel_size):
        #Function to subdivide long stitch segments
        def subdivide_segment(start_pt, end_pt, max_stitch_length):
            (x1, y1) = start_pt
            (x2, y2) = end_pt
            distance = math.hypot(x2 - x1, y2 - y1)
            if distance <= max_stitch_length:
                return [start_pt, end_pt]
            num_segments = int(math.ceil(distance / max_stitch_length))
            return [(x1 + t * (x2 - x1), y1 + t * (y2 - y1)) for t in np.linspace(0, 1, num_segments + 1)]

        #Load the image
        image = cv2.imread(image_path)
        if image is None:
            print("Error: Unable to load image.")
            return None
        max_jump_length_factor = 1.5
        #Convert image to RGB and create a binary mask (non-black pixels = 1, black = 0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = np.any(image != [0, 0, 0], axis=-1).astype(np.uint8)
        height, width = mask.shape

        #Store action log for later MST processing
        action_log = []
        direction = True  #Left to Right = True, Right to Left = False

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

        pattern.end() #Finalize the pattern

        #Plot the connected stitches
        self.plot_connected_stitches(ordered_points)

        return pattern #Return the properly ordered embroidery pattern
    '''
    ordered_points = self.join_stitch_points(action_log)
    
    pattern = pe.EmbPattern()
    max_jump_length = max_jump_length_factor * max_stitch_length
    
    for i, (x, y) in enumerate(ordered_points):
        if i == 0:
            pattern.add_stitch_absolute(pe.JUMP, x, y)
        else:
            prev_x, prev_y = ordered_points[i - 1]
            jump_distance = math.hypot(prev_x - x, prev_y - y)
            
            # Implement a more intelligent jumping strategy
            if jump_distance > max_jump_length:
                # Find the closest point in the previous or next few points
                look_ahead = min(5, len(ordered_points) - i)
                look_behind = min(5, i)
                
                # Check nearby points for shorter jumps
                shortest_jump = jump_distance
                best_point_index = i
                
                for j in range(1, look_ahead):
                    if i + j < len(ordered_points):
                        alt_jump = math.hypot(prev_x - ordered_points[i+j][0], 
                                              prev_y - ordered_points[i+j][1])
                        if alt_jump < shortest_jump:
                            shortest_jump = alt_jump
                            best_point_index = i + j
                
                for j in range(1, look_behind):
                    if i - j >= 0:
                        alt_jump = math.hypot(prev_x - ordered_points[i-j][0], 
                                              prev_y - ordered_points[i-j][1])
                        if alt_jump < shortest_jump:
                            shortest_jump = alt_jump
                            best_point_index = i - j
                
                # Jump to the point with the shortest jump
                pattern.add_command(pe.TRIM)
                pattern.add_stitch_absolute(pe.JUMP, ordered_points[best_point_index][0], 
                                            ordered_points[best_point_index][1])
            
            pattern.add_stitch_absolute(pe.STITCH, x, y)
    '''



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