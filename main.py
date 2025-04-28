import sys
import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
from winotify import Notification
from scipy.spatial import distance as dist
from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QCheckBox, QSystemTrayIcon, QMenu, QAction, QHBoxLayout, QSpacerItem, QSizePolicy
)
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from PyQt5.QtGui import QIcon, QPixmap, QImage

# ---------------------- Alert Function ----------------------
def trigger_alert(message, voice_enabled, toast_enabled):
    if voice_enabled:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.say(message)
        engine.runAndWait()

    if toast_enabled:
        toast = Notification(
            app_id="💤 Sleep Monitor",
            title="⚠️ Alert!",
            msg=message,
            duration="short"
        )
        toast.show()

# ---------------------- Detection Thread ----------------------
class DetectionThread(QThread):
    status_update = pyqtSignal(str)
    frame_update = pyqtSignal(QImage)
    finished_signal = pyqtSignal()

    def __init__(self, voice_enabled, toast_enabled, yawn_enabled, head_enabled):
        super().__init__()
        self.voice_enabled = voice_enabled
        self.toast_enabled = toast_enabled
        self.yawn_enabled = yawn_enabled
        self.head_enabled = head_enabled
        self._running = True

    def run(self):
        LEFT_EYE = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE = [362, 385, 387, 263, 373, 380]
        LIPS = [13, 14]
        NOSE_TIP = 1

        EAR_THRESHOLD = 0.20
        CONSEC_FRAMES = 20
        HEAD_DROP_THRESHOLD = 20

        counter = 0
        alerted_drowsy = False
        alerted_yawn = False
        alerted_head = False
        prev_nose_y = None

        face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1)
        cap = cv2.VideoCapture(0)

        while cap.isOpened() and self._running:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                mesh = results.multi_face_landmarks[0].landmark
                mesh_points = np.array([[int(p.x * w), int(p.y * h)] for p in mesh])

                def ear(eye):
                    A = dist.euclidean(eye[1], eye[5])
                    B = dist.euclidean(eye[2], eye[4])
                    C = dist.euclidean(eye[0], eye[3])
                    return (A + B) / (2.0 * C)

                left_eye = mesh_points[LEFT_EYE]
                right_eye = mesh_points[RIGHT_EYE]
                avg_ear = (ear(left_eye) + ear(right_eye)) / 2.0

                if avg_ear < EAR_THRESHOLD:
                    counter += 1
                    if counter >= CONSEC_FRAMES and not alerted_drowsy:
                        self.status_update.emit("😴 Sleepiness Detected!")
                        trigger_alert("Alert mode, bitch. I didn’t come here to babysit you.", self.voice_enabled, self.toast_enabled)
                        alerted_drowsy = True
                else:
                    counter = 0
                    self.status_update.emit("🙂 Monitoring...")
                    alerted_drowsy = False

                if self.yawn_enabled:
                    upper_lip = mesh_points[LIPS[0]]
                    lower_lip = mesh_points[LIPS[1]]
                    mouth_dist = np.linalg.norm(upper_lip - lower_lip)
                    if mouth_dist > 35 and not alerted_yawn:
                        self.status_update.emit("😮 Yawning Detected!")
                        trigger_alert("Quit yawning like a lazy ass. We have got shit to do.", self.voice_enabled, self.toast_enabled)
                        alerted_yawn = True
                    elif mouth_dist <= 25:
                        alerted_yawn = False

                if self.head_enabled:
                    current_nose_y = mesh_points[NOSE_TIP][1]
                    if prev_nose_y is None:
                        prev_nose_y = current_nose_y

                    delta_y = current_nose_y - prev_nose_y
                    prev_nose_y = (0.8 * prev_nose_y) + (0.2 * current_nose_y)

                    if delta_y > HEAD_DROP_THRESHOLD and not alerted_head:
                        self.status_update.emit("📉 Head Drop Detected!")
                        trigger_alert("If your head hits the desk, I am not responsible for the dent.", self.voice_enabled, self.toast_enabled)
                        alerted_head = True
                    elif delta_y < 0:
                        alerted_head = False

                cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            else:
                self.status_update.emit("🙈 No face detected")

            qt_img = QImage(frame.data, w, h, 3 * w, QImage.Format_BGR888)
            self.frame_update.emit(qt_img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        self.status_update.emit("🛑 Monitoring stopped.")
        self.finished_signal.emit()

    def stop(self):
        self._running = False

# ---------------------- GUI ----------------------
class SleepMonitorApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("💼 Sleep Alert Monitor")
        self.setFixedSize(450, 620)

        self.label = QLabel("🙂 Monitoring...")
        self.label.setStyleSheet("font-size: 14px;")

        self.image_label = QLabel("Webcam feed")
        self.image_label.setFixedSize(400, 300)
        self.image_label.setAlignment(Qt.AlignCenter)

        self.start_button = QPushButton("Start Monitoring")
        self.stop_button = QPushButton("Stop Monitoring")

        self.start_button.clicked.connect(self.start_monitoring)
        self.stop_button.clicked.connect(self.stop_monitoring)

        self.voice_checkbox = QCheckBox("Voice Alerts")
        self.toast_checkbox = QCheckBox("Toast Notifications")
        self.yawn_checkbox = QCheckBox("Yawn Detection")
        self.head_checkbox = QCheckBox("Head Drop Detection")
        for cb in [self.voice_checkbox, self.toast_checkbox, self.yawn_checkbox, self.head_checkbox]:
            cb.setChecked(True)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.image_label)
        layout.addWidget(self.start_button)
        layout.addWidget(self.stop_button)
        layout.addWidget(self.voice_checkbox)
        layout.addWidget(self.toast_checkbox)
        layout.addWidget(self.yawn_checkbox)
        layout.addWidget(self.head_checkbox)

        self.setLayout(layout)

    def start_monitoring(self):
        self.label.setText("Monitoring started...")
        self.thread = DetectionThread(
            voice_enabled=self.voice_checkbox.isChecked(),
            toast_enabled=self.toast_checkbox.isChecked(),
            yawn_enabled=self.yawn_checkbox.isChecked(),
            head_enabled=self.head_checkbox.isChecked()
        )
        self.thread.status_update.connect(self.label.setText)
        self.thread.frame_update.connect(self.update_frame)
        self.thread.finished_signal.connect(self.thread.deleteLater)
        self.thread.start()

    def stop_monitoring(self):
        self.thread.stop()

    def update_frame(self, image):
        self.image_label.setPixmap(QPixmap.fromImage(image))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SleepMonitorApp()
    window.show()
    sys.exit(app.exec_())