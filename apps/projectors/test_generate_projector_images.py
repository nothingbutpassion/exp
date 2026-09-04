import sys
import math
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QLabel, QWidget
from PyQt6.QtGui import QImage, QPixmap, QKeySequence, QShortcut
from PyQt6.QtCore import Qt

from generate_projector_images import (
    load_image,
    warp_image,
    projection_matrix,
    screen_projector_transforms,
    screen_projector_homographies,
    tranform_points,
    get_screen_regions,
    rectify_screen_regions,
    screen_buffer_homographies,
    get_buffer_width,
    split_buffers
)

# NOTES: 
# There are 4 coordinate frames:
# frame0: dual-projectors-center frame (origin is ground projection point of the center)
# frame1: left projector frame
# frame2: right projector frame
# frame3: screen coordinate frame
PRJ_W = 1920                                            # projector image width
PRJ_H = 1080                                            # projector image height

FOV1_H = math.radians(44.34)                            # left projector horizontal fov        
FOV1_V = math.atan(2*math.tan(0.5*FOV1_H)*PRJ_H/PRJ_W)  # left projector vertical fov

FOV2_H = math.radians(44.34)                            # right projector horizontal fov        
FOV2_V = math.atan(2*math.tan(0.5*FOV2_H)*PRJ_H/PRJ_W)  # right projector vertical fov

D1 = 730                                                # distance between 2 projectors
D2 = 1450                                               # distance from projectors-center to screen
ALPHA = math.radians(22.5)                              # screen angle relative to projectors

def print_values(values, prefix=""):
    s = f"{prefix}"
    for v in values:
        if type(v) is [list, np.ndarray]:
            for e in v:
                s += f" {float(v):.6}"
            s += "\n"
        else:
            s += f" {float(v):.6}"
    print(s)

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
    pts1 = tranform_points(corners, np.linalg.inv(Hr2))
    print_pixel_points(pts1, "P5-P8:")

def imshow(win_name, image, fx=0.5, fy=0.5):
    image = cv2.resize(image, dsize=None, fx=fx, fy=fx)
    cv2.imshow(win_name, image)

# Only for debugging
def demo_image(w, h):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # bounding box
    d = 5
    img[:d]     = (0, 0, 255)    # top
    img[-d:]    = (0, 255, 0)    # bottom
    img[:,:d]   = (255, 0, 0)    # left
    img[:,-d:]  = (0, 255, 255)  # right
    # cross line
    img[:,(w-d)//2:(w+d)//2]    = (255, 255, 0)
    img[(h-d)//2:(h+d)//2]      = (255, 255, 0)
    # center circle
    r = min(w//2, h//2) - d
    cv2.circle(img, (w//2, h//2), r, (255, 0, 255), 3, cv2.LINE_AA)
    return img

def numpy_to_qpixmap(img):
    h, w, c = img.shape
    bytes_per_line = c * w
    q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
    return QPixmap.fromImage(q_img)

def test_multi_screen_show(image_path):
    K1 = projection_matrix(PRJ_W, PRJ_H, FOV1_H, FOV1_V)
    K2 = projection_matrix(PRJ_W, PRJ_H, FOV2_H, FOV2_V)
    T13, T23 = screen_projector_transforms(D1, D2, ALPHA)
    H13, H23 = screen_projector_homographies(K1, K2, T13, T23)
    H31, H32 = np.linalg.inv(H13), np.linalg.inv(H23)

    screen_r1, screen_r2 = get_screen_regions(H31, H32, PRJ_W, PRJ_H)
    screen_r1, screen_r2, overlap_r = rectify_screen_regions(screen_r1, screen_r2)
    buf_w = get_buffer_width(overlap_r, PRJ_W, PRJ_H)
    image = load_image(image_path, buf_w, PRJ_H)
    
    buf1, buf2 = split_buffers(image, PRJ_W, PRJ_H)
    Hl3, Hr3 = screen_buffer_homographies(overlap_r, PRJ_W, PRJ_H)
    Hl1 = Hl3 @ H31
    Hr2 = Hr3 @ H32

    print_fpga_info(buf_w, PRJ_W, PRJ_H, Hl1, Hr2)

    image1 = warp_image(buf1, Hl1)
    image2 = warp_image(buf2, Hr2)

    # only for debugging
    # imshow("input image", image, fx=0.5, fy=0.5)
    # imshow("left  image", image1, fx=0.5, fy=0.5)
    # imshow("right image", image2, fx=0.5, fy=0.5)
    # cv2.waitKey()

    app = QApplication(sys.argv)
    pixmaps = [numpy_to_qpixmap(image1), numpy_to_qpixmap(image2)]
    screens = app.screens()
    # print screen information
    for i, s in enumerate(screens):
        geo = s.geometry()
        dpr = s.devicePixelRatio()
        print(f"screen {i}: geometry={geo}, dpr={dpr}, physical w/h={geo.width()*dpr:.0f} x {geo.height()*dpr:.0f}")
    screens = screens[1:]   # screen 0 (main screen) is not used
    screens.sort(key=lambda s: s.geometry().x())
    wins = [ QWidget() for _ in screens]
    for idx, screen in enumerate(screens):
        win = wins[idx]
        QShortcut(QKeySequence(Qt.Key.Key_Escape), win, activated=app.quit) # ESC to quit
        win.setScreen(screen)
        win.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        win.setGeometry(screen.geometry())
        label = QLabel(win)
        label.setPixmap(pixmaps[idx % len(pixmaps)])
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        win.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    assert len(sys.argv), f"usage: {sys.argv[0]} <image-file>"
    test_multi_screen_show(sys.argv[1])