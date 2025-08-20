import sys
import threading
import time
import os
import cv2
import pyzed.sl as sl
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel, QHBoxLayout
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QImage, QPixmap

# --- ArUco/ChArUco setup ---\
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
board = cv2.aruco.CharucoBoard((5, 7), 0.04, 0.03, aruco_dict)
aruco_params = cv2.aruco.DetectorParameters()
# --- Adaptive Thresholding ---
# aruco_params.adaptiveThreshWinSizeMin = 3       # smaller window to catch small markers
# aruco_params.adaptiveThreshWinSizeMax = 23      # max window size
# aruco_params.adaptiveThreshWinSizeStep = 2
# aruco_params.adaptiveThreshConstant = 7         # adjust depending on lighting

# # --- Marker size / filtering tweaks ---
# aruco_params.minMarkerPerimeterRate = 0.03      # lower so smaller markers are detected
# aruco_params.maxMarkerPerimeterRate = 4.0       # keep default
# aruco_params.polygonalApproxAccuracyRate = 0.03 # tighter contour approximation

# # --- Corner refinement ---
# aruco_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
# aruco_params.cornerRefinementWinSize = 5        # small enough for small markers
# aruco_params.cornerRefinementMaxIterations = 30
# aruco_params.cornerRefinementMinAccuracy = 0.01

# # --- Detection filtering ---
# aruco_params.minDistanceToBorder = 3             # reduce if markers are near edge
# aruco_params.minCornerDistanceRate = 0.05        # default ~0.05, ok

# # --- Optional: improve detection under blur / noise ---
# # aruco_params.doCornerRefinement = True
# aruco_params.errorCorrectionRate = 0.7           # relax strictness a bit


# Global state
exit_app = False
capture_image = []
image_counters = []
detection_ok = []   # one boolean per camera
capture_lock = threading.Lock()
CAPTURE_PER_STATION = 10

# ---------------- UTILS -----------------
def create_camera_folders(nb_cameras):
    for i in range(nb_cameras):
        folder_name = f"Cam_{(i+1):03d}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

def save_image(image, camera_index):
    global image_counters
    folder_name = f"Cam_{(camera_index+1):03d}"
    counter = image_counters[camera_index]
    filename = f"image_{counter:04d}.png"
    filepath = os.path.join(folder_name, filename)
    image_cv = image.get_data()
    cv2.imwrite(filepath, image_cv)
    print(f"Saved: {filepath}")
    image_counters[camera_index] += 1

def acquisition(zed, camera_index):
    global capture_image, exit_app
    infos = zed.get_camera_information()
    image = sl.Mat()
    while not exit_app:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            if capture_image[camera_index] > 0:
                zed.retrieve_image(image, sl.VIEW.LEFT)
                save_image(image, camera_index)
                capture_image[camera_index] -= 1
        time.sleep(0.01)
    print(f"{infos.camera_model}[{infos.serial_number}] QUIT")
    zed.close()

def open_camera(zed, sn, camera_fps=5):
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    init_params.camera_resolution = sl.RESOLUTION.HD2K
    init_params.set_from_serial_number(sn)
    init_params.camera_fps = camera_fps
    open_err = zed.open(init_params)
    if open_err == sl.ERROR_CODE.SUCCESS:
        print(f"{zed.get_camera_information().camera_model}_SN{sn} Opened")
        zed.set_camera_settings(sl.VIDEO_SETTINGS.AEC_AGC, 0)
        zed.set_camera_settings(sl.VIDEO_SETTINGS.EXPOSURE, 15)
        return True
    else:
        print(f"ZED SN{sn} Error: {open_err}")
        return False

# ---------------- GUI -----------------
class AcquireGui(QWidget):
    def __init__(self, zeds):
        super().__init__()
        self.zeds = zeds
        self.images = [sl.Mat() for _ in zeds]
        self.labels = []  # QLabel for each camera
        self.init_ui()

        # timer for realtime display
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_views)
        self.timer.start(200)   # 5 fps

    def init_ui(self):
        self.setWindowTitle("ZED Multi-Cam Capture")
        layout = QVBoxLayout()
        self.label = QLabel("Press 'Acquire' to capture images from all cameras")
        layout.addWidget(self.label)

        # camera views
        cam_layout = QHBoxLayout()
        for i in range(len(self.zeds)):
            l = QLabel(f"Camera {i+1}")
            l.setFixedSize(1280, 720)
            self.labels.append(l)
            cam_layout.addWidget(l)
        layout.addLayout(cam_layout)

        self.button = QPushButton("Acquire")
        self.button.clicked.connect(self.acquire_images)
        layout.addWidget(self.button)
        self.setLayout(layout)

    def acquire_images(self):
        global capture_image
        print(f"--- Capturing {CAPTURE_PER_STATION} images per camera ---")
        for i in range(len(self.zeds)):
            capture_image[i] = CAPTURE_PER_STATION

    def update_views(self):
        global detection_ok
        for i, zed in enumerate(self.zeds):
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                zed.retrieve_image(self.images[i], sl.VIEW.LEFT)
                frame = self.images[i].get_data()
                # convert to BGR for OpenCV
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
                # --- detect ChArUco corners ---
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=aruco_params)
                ok = False
                num_corners = 0
                if ids is not None and len(ids) > 0:
                    cv2.aruco.drawDetectedMarkers(frame, corners, ids)
                    num_corners = len(corners)
                    if num_corners >= 4:  # or any threshold you consider valid
                        ok = True

                detection_ok[i] = ok

                # --- overlay text ---
                text = f"Corners: {num_corners}"
                cv2.putText(frame, text, (50, 100), 
                            cv2.FONT_HERSHEY_SIMPLEX,   # font
                            2.0,                        # font scale (big & visible)
                            (0, 255, 0),                # text color (green)
                            4,                          # thickness
                            cv2.LINE_AA)                # anti-aliased

                # resize to preview
                disp = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_LINEAR)
                # convert BGR -> RGB for Qt display
                disp_rgb = cv2.cvtColor(disp, cv2.COLOR_BGR2RGB)

                # make sure buffer is contiguous
                disp_rgb = disp_rgb.copy()

                h, w, ch = disp_rgb.shape
                qimg = QImage(disp_rgb.data, w, h, ch*w, QImage.Format_RGB888)
                self.labels[i].setPixmap(QPixmap.fromImage(qimg))


        # enable button only if at least 2 cameras detect corners
        self.button.setEnabled(sum(detection_ok) >= 2)

# ---------------- MAIN -----------------
def main():
    global exit_app, capture_image, image_counters, detection_ok
    dev_list = sl.Camera.get_device_list()
    nb_cameras = len(dev_list)
    if nb_cameras == 0:
        print("No ZED Detected, exit program")
        return 1
    create_camera_folders(nb_cameras)

    zeds = []
    for dev in dev_list:
        zed = sl.Camera()
        if open_camera(zed, dev.serial_number):
            zeds.append(zed)
    if not zeds:
        print("No ZED opened, exit program")
        return 1

    capture_image = [0] * len(zeds)
    image_counters = [1] * len(zeds)
    detection_ok = [False] * len(zeds)

    # acquisition threads (for saving)
    threads = []
    for i, zed in enumerate(zeds):
        t = threading.Thread(target=acquisition, args=(zed, i), daemon=True)
        t.start()
        threads.append(t)

    app = QApplication(sys.argv)
    gui = AcquireGui(zeds)
    gui.show()
    try:
        app.exec_()
    finally:
        exit_app = True
        for t in threads:
            t.join()
        for zed in zeds:
            zed.close()
    return 0

if __name__ == "__main__":
    sys.exit(main())
