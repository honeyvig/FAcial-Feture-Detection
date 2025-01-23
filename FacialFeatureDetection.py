import sys
import cv2
import numpy as np
import dlib
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLabel, QFileDialog, QComboBox
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from PIL import Image

class FacialAnalysisApp(QWidget):
    def __init__(self):
        super().__init__()

        # Initialize dlib face detector and shape predictor
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # You need to download this model

        self.initUI()

    def initUI(self):
        layout = QVBoxLayout()

        # Create button for capturing photo from the camera
        self.capture_btn = QPushButton('Capture Photo', self)
        self.capture_btn.clicked.connect(self.capture_image)
        layout.addWidget(self.capture_btn)

        # Create combo box for feature analysis options (e.g., X-ray, ECG, etc.)
        self.analysis_combo = QComboBox(self)
        self.analysis_combo.addItem("Normal View")
        self.analysis_combo.addItem("X-Ray View")
        self.analysis_combo.addItem("MRI View")
        self.analysis_combo.currentIndexChanged.connect(self.apply_transformation)
        layout.addWidget(self.analysis_combo)

        # Label to display the image
        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)

        # Set the layout
        self.setLayout(layout)
        self.setWindowTitle('Facial Feature Analysis')
        self.setGeometry(100, 100, 800, 600)

        self.captured_image = None
        self.show()

    def capture_image(self):
        # Open the webcam and capture the image
        cap = cv2.VideoCapture(0)  # 0 for the default webcam
        if not cap.isOpened():
            print("Error: Cannot access camera.")
            return

        ret, frame = cap.read()
        cap.release()

        if ret:
            self.captured_image = frame
            self.detect_features(frame)
            self.update_image(frame)

    def detect_features(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        for face in faces:
            landmarks = self.predictor(gray, face)

            # Draw landmarks for facial features
            for n in range(36, 48):  # Eyes
                x, y = landmarks.part(n).x, landmarks.part(n).y
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

            for n in range(48, 60):  # Mouth
                x, y = landmarks.part(n).x, landmarks.part(n).y
                cv2.circle(frame, (x, y), 2, (255, 0, 0), -1)

            # Nose (nose tip)
            nose = landmarks.part(30)
            cv2.circle(frame, (nose.x, nose.y), 3, (255, 255, 0), -1)

            # Draw bounding box around face
            cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), (0, 0, 255), 2)

    def update_image(self, frame):
        # Convert image for display in PyQt
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        img = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        pixmap = QPixmap.fromImage(img)
        self.image_label.setPixmap(pixmap)

    def apply_transformation(self):
        if self.captured_image is None:
            return

        # Apply the selected transformation (e.g., X-ray, MRI, etc.)
        transformation_type = self.analysis_combo.currentText()
        transformed_image = self.captured_image.copy()

        if transformation_type == "X-Ray View":
            transformed_image = self.apply_xray(transformed_image)
        elif transformation_type == "MRI View":
            transformed_image = self.apply_mri(transformed_image)

        self.update_image(transformed_image)

    def apply_xray(self, image):
        # Apply an X-ray-like filter (just an example using image manipulation)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, xray_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(xray_image, cv2.COLOR_GRAY2BGR)

    def apply_mri(self, image):
        # Apply an MRI-like effect (using simple blurring to simulate the concept)
        return cv2.GaussianBlur(image, (15, 15), 0)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = FacialAnalysisApp()
    sys.exit(app.exec_())
