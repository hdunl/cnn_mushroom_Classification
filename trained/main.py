import sys
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog, QProgressBar, QFrame, QHBoxLayout
from PyQt5.QtGui import QPixmap, QImage, QFont
from PyQt5.QtCore import Qt, QThread, pyqtSignal

model = load_model('mushroom_species_classifier.h5')
IMG_SIZE = (260, 260)
train_data_dir = r'C:\Users\hdun\Downloads\mushroom_images'
class_names = sorted(os.listdir(train_data_dir))[:114]

def preprocess_image(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def resize_image(image, max_size=(800, 600)):
    h, w = image.shape[:2]
    if h > max_size[1] or w > max_size[0]:
        scale = min(max_size[0] / w, max_size[1] / h)
        new_size = (int(w * scale), int(h * scale))
        return cv2.resize(image, new_size)
    return image

def get_model_summary(model):
    total_params = model.count_params()
    total_layers = len(model.layers)
    total_neurons = "~71,124"
    return f"Total parameters: {total_params:,}\nTotal layers: {total_layers}\nTotal neurons: {total_neurons}"

class ImageProcessingThread(QThread):
    update_ui_signal = pyqtSignal(str, float, QPixmap, str)

    def __init__(self, image_path):
        super().__init__()
        self.image_path = image_path

    def run(self):
        processed_image = preprocess_image(self.image_path)
        prediction = model.predict(processed_image)

        predicted_class_index = np.argmax(prediction)
        confidence = np.max(prediction)
        predicted_class = class_names[predicted_class_index]

        image = cv2.imread(self.image_path)
        image = resize_image(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        qimage = QImage(image.data, image.shape[1], image.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)

        top_3_indices = prediction[0].argsort()[-3:][::-1]
        top_3_text = "Top 3 predictions:\n"
        for idx in top_3_indices:
            top_3_text += f"{class_names[idx]}: {prediction[0][idx]:.4f}\n"

        self.update_ui_signal.emit(predicted_class, confidence, pixmap, top_3_text)

class RoundedButton(QPushButton):
    def __init__(self, text, parent=None):
        super().__init__(text, parent)
        self.setFixedSize(200, 50)
        self.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border-radius: 25px;
                font-size: 16px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Mushroom Species Classifier')
        self.setGeometry(100, 100, 900, 800)
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                color: #333333;
                font-family: 'Segoe UI', Arial, sans-serif;
            }
            QLabel {
                color: #333333;
            }
            QProgressBar {
                border: 2px solid #4CAF50;
                border-radius: 5px;
                text-align: center;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
            }
        """)

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(40, 40, 40, 40)

        self.image_frame = QFrame(self)
        self.image_frame.setStyleSheet("background-color: white; border-radius: 10px;")
        self.image_frame.setFixedSize(820, 620)
        image_layout = QVBoxLayout(self.image_frame)

        self.image_label = QLabel(self.image_frame)
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("border-radius: 10px;")
        image_layout.addWidget(self.image_label)

        main_layout.addWidget(self.image_frame, alignment=Qt.AlignCenter)

        self.result_label = QLabel(self)
        self.result_label.setFont(QFont("Segoe UI", 14, QFont.Bold))
        self.result_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.result_label)

        self.top_3_label = QLabel(self)
        self.top_3_label.setFont(QFont("Segoe UI", 12))
        self.top_3_label.setAlignment(Qt.AlignCenter)
        self.top_3_label.setStyleSheet("color: #666666;")
        main_layout.addWidget(self.top_3_label)

        self.model_info_label = QLabel(self)
        self.model_info_label.setFont(QFont("Segoe UI", 10))
        self.model_info_label.setAlignment(Qt.AlignCenter)
        self.model_info_label.setText(get_model_summary(model))
        main_layout.addWidget(self.model_info_label)

        self.progress_bar = QProgressBar(self)
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedHeight(10)
        main_layout.addWidget(self.progress_bar)

        button_layout = QHBoxLayout()
        self.select_button = RoundedButton("Select Image", self)
        self.select_button.clicked.connect(self.open_file_dialog)
        button_layout.addWidget(self.select_button)

        self.back_button = RoundedButton("Back", self)
        self.back_button.clicked.connect(self.reset_ui)
        self.back_button.setVisible(False)
        button_layout.addWidget(self.back_button)

        main_layout.addLayout(button_layout)

    def open_file_dialog(self):
        image_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp)")
        if image_path:
            self.start_image_processing(image_path)

    def start_image_processing(self, image_path):
        self.select_button.setVisible(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)

        self.thread = ImageProcessingThread(image_path)
        self.thread.update_ui_signal.connect(self.update_ui)
        self.thread.start()

    def update_ui(self, predicted_class, confidence, pixmap, top_3_text):
        self.image_label.setPixmap(pixmap.scaled(800, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.result_label.setText(f"Predicted class: {predicted_class}\nConfidence: {confidence:.4f}")
        self.top_3_label.setText(top_3_text)

        self.progress_bar.setVisible(False)
        self.back_button.setVisible(True)

    def reset_ui(self):
        self.image_label.clear()
        self.result_label.clear()
        self.top_3_label.clear()
        self.select_button.setVisible(True)
        self.back_button.setVisible(False)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
