import sys
import threading
import time
import os
import cv2
import pyzed.sl as sl

from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QVBoxLayout, QLabel

# Global variables
exit_app = False
capture_image = []  # one flag per camera
image_counter = 0
capture_lock = threading.Lock()


# ---------------- UTILS -----------------
def create_camera_folders(nb_cameras):
    for i in range(nb_cameras):
        # save as Cam_000, Cam_001, ..., Cam_099
        folder_name = f"Cam_{(i+1):03d}"
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)


def save_image(image, camera_index, counter):
    folder_name = f"Cam_{(camera_index+1):03d}"
    filename = f"image_{counter:04d}.png"
    filepath = os.path.join(folder_name, filename)
    image_cv = image.get_data()
    cv2.imwrite(filepath, image_cv)
    print(f"Saved: {filepath}")


def acquisition(zed, camera_index):
    global capture_image, image_counter, capture_lock, exit_app

    infos = zed.get_camera_information()
    image = sl.Mat()

    while not exit_app:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            with capture_lock:
                if capture_image[camera_index]:
                    zed.retrieve_image(image, sl.VIEW.LEFT)
                    save_image(image, camera_index, image_counter)
                    capture_image[camera_index] = False

        time.sleep(0.01)

    print(f"{infos.camera_model}[{infos.serial_number}] QUIT")
    zed.close()


def open_camera(zed, sn, camera_fps=30):
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.NONE
    init_params.camera_resolution = sl.RESOLUTION.HD2K
    init_params.set_from_serial_number(sn)
    init_params.camera_fps = camera_fps

    open_err = zed.open(init_params)
    if open_err == sl.ERROR_CODE.SUCCESS:
        print(f"{zed.get_camera_information().camera_model}_SN{sn} Opened")
        return True
    else:
        print(f"ZED SN{sn} Error: {open_err}")
        return False


# ---------------- GUI -----------------
class AcquireGui(QWidget):
    def __init__(self, zeds):
        super().__init__()
        self.zeds = zeds
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("ZED Multi-Cam Capture")
        layout = QVBoxLayout()

        self.label = QLabel("Press 'Acquire' to capture images from all cameras")
        layout.addWidget(self.label)

        self.button = QPushButton("Acquire")
        self.button.clicked.connect(self.acquire_images)
        layout.addWidget(self.button)

        self.setLayout(layout)

    def acquire_images(self):
        global capture_image, image_counter, capture_lock
        with capture_lock:
            image_counter += 1
            print(f"--- Capturing image set #{image_counter} ---")
            for i in range(len(self.zeds)):
                capture_image[i] = True


# ---------------- MAIN -----------------
def main():
    global exit_app, capture_image

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

    # Init capture flags
    capture_image = [False] * len(zeds)

    # Start acquisition threads
    threads = []
    for i, zed in enumerate(zeds):
        t = threading.Thread(target=acquisition, args=(zed, i), daemon=True)
        t.start()
        threads.append(t)

    # Start Qt app
    app = QApplication(sys.argv)
    gui = AcquireGui(zeds)
    gui.show()
    app.exec_()

    # Exit
    exit_app = True
    for t in threads:
        t.join()

    print("Program exited")
    return 0


if __name__ == "__main__":
    sys.exit(main())