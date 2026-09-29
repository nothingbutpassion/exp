import sys
import time
import random
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QWidget, QMainWindow, QVBoxLayout, QLabel, QPushButton, QHBoxLayout
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer

class FocusUtils:

    @staticmethod
    def generate_box_blur_images(original, num_images, ksize=(7, 7)):
        image = original
        blured = []
        for _ in range(num_images):
            image = cv2.boxFilter(image, -1, ksize)
            blured.append(image)
        return blured

    @staticmethod
    def generate_pyramid_blur_images(original, num_images):
        down_scales = np.linspace(0.9, 0.1, num_images)
        image = original
        w, h = image.shape[1::-1]
        down_imgs = []
        for s in down_scales:
            image = cv2.resize(image, dsize=(round(s*w), round(s*h)))
            down_imgs = [image] + down_imgs

        blured = []
        for image in down_imgs:
            image = cv2.GaussianBlur(image, ksize=(5, 5), sigmaX=1, sigmaY=1)
            image = cv2.resize(image, dsize=(w, h))
            blured.append(image)
        return list(reversed(blured)) 

    # generate generate_defocus_blur_images
    @staticmethod
    def generate_defocus_blur_images(original, num_images, max_radius=20):
        blurred_images = []
        for i in range(1, num_images+1):
            # radius increases with blur level
            radius = int(i*max_radius/num_images)
            radius = max(1, radius)
            # create a circular (disk) kernel
            kernel_size = 2 * radius + 1
            Y, X = np.ogrid[:kernel_size, :kernel_size]
            dist_from_center = np.sqrt((X - radius)**2 + (Y - radius)**2)
            kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
            kernel[dist_from_center <= radius] = 1.0
            kernel /= kernel.sum()
            # apply the disk blur
            blurred = cv2.filter2D(original, -1, kernel)
            blurred_images.append(blurred)
        return blurred_images
    
    @staticmethod
    def generate_gaussian_blur_images(original, num_levels=11, max_kernel_size=51):
        blurred_images = []
        for i in range(1, num_levels+1):
            sigma = (i / num_levels) * (max_kernel_size // 4)
            kernel_size = 2 * int(2 * sigma) + 1
            kernel_size = max(3, min(kernel_size, max_kernel_size))
            blurred = cv2.GaussianBlur(original, (kernel_size, kernel_size), sigma)
            blurred_images.append(blurred)
        return blurred_images

    # NOTES:
    # In my testing, line/grid patter is shaper than chessboard
    @staticmethod
    def grid_pattern(w, h, dx, dy, d=8, color1=(0, 0, 0), color2=(255, 255, 255)):
        img = np.full((h, w, 3), color1, dtype=np.uint8)
        nx, ny = w//dx, h//dy
        ox, oy = (w - nx*dx)//2, (h - ny*dy)//2
        for i in range(nx+1):
            s = max(ox+i*dx, 0)
            e = min(ox+i*dx+d, w)
            img[:, s:e] = color2
        for i in range(ny+1):
            s = max(oy+i*dx, 0)
            e = min(oy+i*dx+d, h)
            img[s:e:,] = color2
        return img

    @staticmethod
    def chessboard_pattern(w, h, cell_w, cell_h, color1=(255, 255, 255), color2=(0, 0, 0)):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        nx, ny = w // cell_w, h // cell_h
        for i in range(ny):
            for j in range(nx):
                color = color1 if (i + j) % 2 == 0 else color2
                img[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w] = color
        return img

    @staticmethod
    def image_pattern(image_path, w, h):
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            print(f"can't load {image_path}")
            return img
        return cv2.resize(img, dsize=(w, h), interpolation=cv2.INTER_CUBIC)

    @staticmethod
    def laplacian_sharpness(image):
        # Apply the Laplacian filter to find edges
        # cv2.CV_64F prevents data overflow when calculating differences
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        # Calculate the variance (how much the edges pop out)
        var = laplacian.var()
        sharpness_score = var
        return sharpness_score


class MainWindow(QMainWindow):
    def __init__(self, app):
        super().__init__()
        self.app = app
        self.crate_main_window()
        self.prj_windows, self.prj_labels = self.create_projection_windows()
        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        # for emulating autofocus
        self.max_moto_pos = 63
        self.min_moto_pos = 0
        self.left_moto_pos = random.randint(self.min_moto_pos,  self.max_moto_pos)
        self.left_best_pos = random.randint(self.min_moto_pos+2,  self.max_moto_pos-2)
        self.right_moto_pos = random.randint(self.min_moto_pos,  self.max_moto_pos)
        self.right_best_pos = random.randint(self.min_moto_pos+2,  self.max_moto_pos-2)
        self.black_img = np.zeros((1080, 1920, 3), dtype=np.uint8)

        # pattern = FocusUtils.chessboard_pattern(1920, 1080, 120, 120)
        pattern = FocusUtils.image_pattern("D:/ws/images/brick_wall_1920x1280.jpg", 1920, 1080)
        generate_blur_images = FocusUtils.generate_pyramid_blur_images
        self.left_imgs = generate_blur_images(pattern, self.left_best_pos)[::-1]
        self.left_imgs += [pattern]
        self.left_imgs += generate_blur_images(pattern, self.max_moto_pos - self.left_best_pos)
        # for i, img in enumerate(self.left_imgs):
        #     sharpness = FocusUtils.laplacian_sharpness(img)
        #     print(f"current pos: {i}, sharpness: {sharpness}, best pos: {self.left_best_pos}")
        #     cv2.imshow("image", img)
        #     cv2.waitKey(0)

        self.right_imgs = generate_blur_images(pattern, self.right_best_pos)[::-1]
        self.right_imgs += [pattern]
        self.right_imgs += generate_blur_images(pattern, self.max_moto_pos - self.right_moto_pos)
        # for i, img in enumerate(self.right_imgs):
        #     print(f"current pos: {i}, best pos: {self.right_best_pos}")
        #     cv2.imshow("image", img)
        #     cv2.waitKey(0)

    def closeEvent(self, event):
        self.stop_camera()
        for win in self.prj_windows:
            win.close()
        event.accept()

    def crate_main_window(self):
        self.setWindowTitle("Dual-Projector autofocus experiment")
        self.resize(800, 600)
        self.video_label = QLabel("Camera not started")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.start_camera_btn = QPushButton("Start Camera")
        self.stop_camera_btn = QPushButton("Stop Camera")
        self.stop_camera_btn.setEnabled(False)
        self.auto_focus_btn = QPushButton("Autofocus")
        controls = QHBoxLayout()
        controls.addWidget(self.start_camera_btn)
        controls.addWidget(self.stop_camera_btn)
        controls.addWidget(self.auto_focus_btn)
        controls.addStretch()
        layout = QVBoxLayout()
        layout.addWidget(self.video_label)
        layout.addLayout(controls)
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)
        self.start_camera_btn.clicked.connect(self.start_camera)
        self.stop_camera_btn.clicked.connect(self.stop_camera)
        self.auto_focus_btn.clicked.connect(self.emulate_autofocus)

    def start_camera(self):
        if self.cap is None:
            self.cap = cv2.VideoCapture(1)
        if not self.cap.isOpened():
            self.video_label.setText("Failed to open camera")
            return
        self.timer.start(30)
        self.start_camera_btn.setEnabled(False)
        self.stop_camera_btn.setEnabled(True)

    def stop_camera(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.video_label.clear()
        self.video_label.setText("Camera stopped")
        self.video_label.setStyleSheet("background-color: black; color: white;")
        self.start_camera_btn.setEnabled(True)
        self.stop_camera_btn.setEnabled(False)

    def update_frame(self):
        if self.cap is None:
            return None
        ret, image = self.cap.read()
        if not ret:
            self.stop_camera()
            return None
        h, w, c = image.shape
        bytes_per_line = c * w
        qimg = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
        # scale to fit label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(qimg).scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)
        return image

    def emulate_autofocus(self):
        preview = self.cap is not None
        if preview:
            self.timer.stop()
        else:
            self.cap = cv2.VideoCapture(1)

        # autofocus for left & right projectors
        for projector_id in [1, 2]:
            self.adjust_projector_focus(projector_id)
            break
    
        # restore camera status
        if preview:
            self.timer.start(30)
        else:
            self.stop_camera()

    def adjust_projector_focus(self, projector_id):
        optimal_pose = self.left_best_pos if projector_id == 1 else self.right_best_pos
        best_score = 0
        best_pos = 0
        
        # step 1: coarse sweeping
        coarse_step = 4
        print(f"projector {projector_id}: starting coarse sweeping...")
        for pos in range(self.min_moto_pos, self.max_moto_pos, coarse_step):
            self.move_motor_to(projector_id, pos)
            images = self.get_projection_images(projector_id)
            self.project_images(images)
            self.wait_ui(1)
            frame = self.update_frame()
            self.wait_ui(1)
            score = FocusUtils.laplacian_sharpness(frame)
            print(f"projector {projector_id}: position={pos}, sharpness={score}, best-position={optimal_pose}")
            if score > best_score:
                best_score = score
                best_pos = pos
            elif score < best_score * 0.6 and best_score > 0:
                # if the score drops significantly, we passed the peak!
                break
        print(f"projector {projector_id}: coarse-position={best_pos}, sharpness={best_score}, best-position={optimal_pose}")

        # step 2: fine sweeping
        print(f"projector {projector_id}: starting fine sweep...")
        # define a small window around our best coarse position
        fine_start = max(self.min_moto_pos, best_pos - coarse_step)
        fine_end = min(self.max_moto_pos, best_pos + coarse_step)
        fine_step = 1
        # reset best score for fine calibration
        best_score = 0
        for pos in range(fine_start, fine_end, fine_step):
            self.move_motor_to(projector_id, pos)
            images = self.get_projection_images(projector_id)
            self.project_images(images)
            self.wait_ui(1)
            frame = self.update_frame()
            self.wait_ui(1)
            score = FocusUtils.laplacian_sharpness(frame)
            print(f"projector {projector_id}: position={pos}, sharpness={score}, best-position={optimal_pose}")
            if score > best_score:
                best_score = score
                best_pos = pos
        print(f"projector {projector_id}: final-position={best_pos}, sharpness={best_score}, best-position={optimal_pose}")
        

    def move_motor_to(self, projecotr_id, moto_pos):
        # emulator stepper/servo motor moving
        target_pos = max(self.min_moto_pos, min(moto_pos, self.max_moto_pos))
        if projecotr_id == 1:
            steps = abs(self.left_moto_pos - target_pos)
        else:
            steps = abs(self.right_moto_pos - target_pos)
        seconds_per_step = 0.01
        time.sleep(steps*seconds_per_step)
        if projecotr_id == 1:
            self.left_moto_pos = target_pos
        else:
            self.right_moto_pos = target_pos

    def get_moto_position(self, projector_id):
        return self.left_moto_pos if projector_id == 1 else self.right_moto_pos

    def get_projection_images(self, projector_id):
        if projector_id == 1:
            if self.left_moto_pos < self.min_moto_pos or self.left_moto_pos > self.max_moto_pos:
                pass
            return self.left_imgs[self.left_moto_pos], self.black_img
        else:
            if self.right_moto_pos < self.min_moto_pos or self.right_moto_pos > self.max_moto_pos:
                pass
            return self.black_img, self.right_imgs[self.right_moto_pos]

    def create_projection_windows(self):
        screens = self.app.screens()
        for i, s in enumerate(screens):
            geo = s.geometry()
            dpr = s.devicePixelRatio()
            print(f"screen {i}: geometry={geo}, dpr={dpr}, physical w/h={geo.width()*dpr:.0f} x {geo.height()*dpr:.0f}")
        screens = screens[1:]
        screens.sort(key=lambda s: s.geometry().x())
        windows = [QWidget() for _ in screens]
        labels = [QLabel(win) for win in windows]
        for idx, screen in enumerate(screens):
            windows[idx].setScreen(screen)
            windows[idx].setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
            windows[idx].setGeometry(screen.geometry())
            labels[idx].setAlignment(Qt.AlignmentFlag.AlignCenter)
        return windows, labels 

    def project_images(self, images):
        for window, label, image in zip(self.prj_windows, self.prj_labels, images):
            h, w, c = image.shape
            bytes_per_line = c * w
            q_img = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
            pixmap = QPixmap.fromImage(q_img)
            label.setPixmap(pixmap)
            window.show()

    def wait_ui(self, seconds):
        QTimer.singleShot(300, lambda: time.sleep(seconds))
        QApplication.processEvents()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow(app)
    main_window.show()
    sys.exit(app.exec())