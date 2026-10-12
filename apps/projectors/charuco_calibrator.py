import sys
import cv2
import time
import numpy as np
from PyQt6.QtWidgets import QApplication, QLabel, QWidget
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtCore import Qt, QTimer

from dual_projection import (
    load_image,
    warp_image,
    tranform_points,
    get_screen_regions,
    rectify_screen_regions,
    adjust_screen_regions,
    screen_buffer_homographies,
    get_buffer_width,
    split_buffers
)

def print_pixel_points(points, prefix=""):
    s = f"{prefix}"
    for x, y in points:
        s = s + f" ({round(x):>5}, {round(y):<5})"
    print(s)

def print_fpga_info(buf_w, prj_w, prj_h, Hl1, Hr2):
    print(f"C = {buf_w} O = {2*prj_w - buf_w} R = {buf_w - prj_w}")
    corners = [(0, 0), (prj_w, 0), (prj_w, prj_h), (0, prj_h)]
    pts1 = tranform_points(corners, np.linalg.inv(Hl1))
    print_pixel_points(pts1, "P1-P4:")
    pts2 = tranform_points(corners, np.linalg.inv(Hr2))
    print_pixel_points(pts2, "P5-P8:")

def demo_image(w, h):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # bounding box
    d = 4
    img[:d]     = (0, 0, 255)    # top
    img[-d:]    = (0, 0, 255)    # bottom
    img[:,:d]   = (0, 255, 0)    # left
    img[:,-d:]  = (0, 255, 0)    # right
    # lines
    hd = 2
    for i in range(1, 4):
        img[:,i*w//4-hd:i*w//4+hd]    = (0, 255, 255)
        img[i*h//4-hd:i*h//4+hd]      = (0, 255, 255)
    # circles
    cv2.circle(img, (w//2, h//2), min(w//2, h//2) - d, (255, 255, 0), d, cv2.LINE_AA)
    cv2.circle(img, (w//2, h//2), min(w//4, h//4) - d, (255, 255, 0), d, cv2.LINE_AA)
    return img

class ScreenHelper:
    def __init__(self):
        # create windows for projector screens
        self.app = QApplication(sys.argv)
        screens = self.app.screens()
        for i, s in enumerate(screens):
            geo = s.geometry()
            dpr = s.devicePixelRatio()
            print(f"screen {i}: geometry={geo}, dpr={dpr}, physical w/h={geo.width()*dpr:.0f} x {geo.height()*dpr:.0f}")
        screens = screens[1:]
        # screens = list(reversed(screens))
        screens.sort(key=lambda s: s.geometry().x())
        wins = [ QWidget() for _ in screens]
        labels = [QLabel(win) for win in wins]
        for screen, win, label in zip(screens, wins, labels):
            win.setScreen(screen)
            win.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
            win.setGeometry(screen.geometry())
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.wins = wins
        self.labels = labels

    def project_images(self, images, delay=3):
        for win, label, image in zip(self.wins, self.labels, images):
            h, w, c = image.shape
            bytes_per_line = c * w
            q_img = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
            pixmap = QPixmap.fromImage(q_img)
            label.setPixmap(pixmap)
            label.show()
            win.show()
        QApplication.processEvents()
        if delay > 0:
            QTimer.singleShot(0, lambda: time.sleep(delay))
            QApplication.processEvents()

    def exit(self, ret_code=0):
        self.app.exit(ret_code)

def generate_charuco_image(x_squares, y_squares, square_length, marker_length, out_w, out_h):
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    board = cv2.aruco.CharucoBoard((x_squares, y_squares), square_length, marker_length, dictionary)
    board_img = board.generateImage(outSize=(out_w, out_h), marginSize=0, borderBits=1)
    board_img = cv2.cvtColor(board_img, cv2.COLOR_GRAY2BGR)
    return board_img
    
def screen_projector_homography(image, x_squares, y_squares, square_length, marker_length, prj_w, prj_h):
    square_length = 12
    marker_length = 8
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_50)
    board = cv2.aruco.CharucoBoard((x_squares, y_squares), square_length, marker_length, dictionary)
    detector = cv2.aruco.CharucoDetector(board, None, None, None)
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(image)
    if charuco_ids is None or len(charuco_ids) < 4:
        cv2.imshow("capture", image)
        cv2.waitKey(0)
        return None
    s = min(prj_w//(x_squares*square_length), prj_h//(y_squares*square_length))
    square_size = s*square_length
    offset_x = (prj_w - x_squares*square_size)//2
    offset_y = (prj_h - y_squares*square_size)//2
    src_points = np.array([(offset_x + j*square_size - 0.5, offset_y + i*square_size - 0.5) 
                           for i in range(1, y_squares) for j in range(1, x_squares)])
    src = np.array([src_points[i] for i in charuco_ids])
    dst = charuco_corners.reshape(-1, 1, 2)
    # cv2.aruco.drawDetectedCornersCharuco(image, dst, charuco_ids, cornerColor=(255, 255, 255))
    H, _ = cv2.findHomography(src, dst)
    return H

def calibrate_projectors(cap, image_path):
    PRJ_W, PRJ_H = 1920, 1080
    MIN_OVERLAP = 32

    # generate chessboard image
    black_img = np.zeros((PRJ_H, PRJ_W, 3), dtype=np.uint8)
    x_squares, y_squares = 5, 3
    square_length = 12
    marker_length = 8
    pattern_img = generate_charuco_image(x_squares, y_squares, square_length, marker_length, PRJ_W, PRJ_H)

    # find homography for left projector
    screen_helper = ScreenHelper()
    screen_helper.project_images([pattern_img, black_img])
    for _ in range(11):
        _, captured_img = cap.read()
    H31 = screen_projector_homography(captured_img, x_squares, y_squares, square_length, marker_length, PRJ_W, PRJ_H)
    if H31 is None:
        screen_helper.exit(-1)
        return
    # find homography for right projector
    screen_helper.project_images([black_img, pattern_img])
    for _ in range(11):
        _, captured_img = cap.read()
    H32 = screen_projector_homography(captured_img, x_squares, y_squares, square_length, marker_length, PRJ_W, PRJ_H)
    if H32 is None:
        screen_helper.exit(-1)
        return
    screen_r1, screen_r2 = get_screen_regions(H31, H32, PRJ_W, PRJ_H)
    screen_r1, screen_r2 = rectify_screen_regions(screen_r1, screen_r2)
    screen_r1, screen_r2, overlap_r = adjust_screen_regions(screen_r1, screen_r2, PRJ_W, PRJ_H, MIN_OVERLAP)
    buf_w = get_buffer_width(overlap_r, PRJ_W, PRJ_H)
    if image_path is not None:
        image = load_image(image_path, buf_w, PRJ_H)
    else:
        image = demo_image(buf_w, PRJ_H)
    buf1, buf2 = split_buffers(image, PRJ_W, PRJ_H)
    Hl3, Hr3 = screen_buffer_homographies(overlap_r, PRJ_W, PRJ_H)
    Hl1 = Hl3 @ H31
    Hr2 = Hr3 @ H32
    print_fpga_info(buf_w, PRJ_W, PRJ_H, Hl1, Hr2)
    image1 = warp_image(buf1, Hl1)
    image2 = warp_image(buf2, Hr2)
    screen_helper.project_images([buf1, buf2], 3)
    screen_helper.project_images([image1, image2], 60)
    screen_helper.exit()

def show_camera_view(cap):
    if not cap.isOpened():
        print("can't open camera")
        return False
    while True:
        ok, img = cap.read()
        if not ok:
            print("read frame failed")
            return False
        cv2.putText(img, "Press Esc to exit application", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
        cv2.putText(img, "Press Enter to start calibration", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
        cv2.imshow("image", img)
        key = cv2.waitKey(30)
        if key == 27:
            return False
        elif key == 13:
            cv2.destroyAllWindows()
            return True

if __name__ == "__main__":
    cap = cv2.VideoCapture(1)
    do_calib = show_camera_view(cap)
    if not do_calib:
        sys.exit(1)
    image_path = sys.argv[1] if len(sys.argv) > 1 else None
    calibrate_projectors(cap, image_path)