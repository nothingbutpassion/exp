import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QLabel, QWidget
from PyQt6.QtGui import QImage, QPixmap, QKeySequence, QShortcut
from PyQt6.QtCore import Qt, QTimer, QEventLoop

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

def numpy_to_qpixmap(img):
    h, w, c = img.shape
    bytes_per_line = c * w
    q_img = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
    return QPixmap.fromImage(q_img)

def project_images(image1, image2, grab_image=None, delay_ms=3000):
    app = QApplication(sys.argv)
    pixmaps = [numpy_to_qpixmap(image1), numpy_to_qpixmap(image2)]
    screens = app.screens()
    # debug print screen information
    for i, s in enumerate(screens):
        geo = s.geometry()
        dpr = s.devicePixelRatio()
        print(f"screen {i}: geometry={geo}, dpr={dpr}, physical w/h={geo.width()*dpr:.0f} x {geo.height()*dpr:.0f}")
    screens = screens[1:]
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
    loop = QEventLoop()
    if grab_image is not None:
        QTimer.singleShot(1000, grab_image)
    QTimer.singleShot(delay_ms, loop.quit)
    loop.exec() # enters main event loop and waits until exit/quit
    app.exit(0)

def generate_chessboard_image(w, h, cell_w, cell_h, color1=(255, 255, 255), color2=(0, 0, 0)):
    chessboard = np.zeros((h, w, 3), dtype=np.uint8)
    nx, ny = w // cell_w, h // cell_h
    for i in range(ny):
        for j in range(nx):
            color = color1 if (i + j) % 2 == 0 else color2
            chessboard[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w] = color
    return chessboard

def find_chessboard_corners(image, pattern_size):
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    # NOTES: Opencv 5.x corners would not be ordered from left to right, from top to bottom
    flags = cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_ADAPTIVE_THRESH
    ret, corners = cv2.findChessboardCornersSB(gray, pattern_size)
    if ret:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    else:
        print(f"find chessboard corners failed: patter_size={pattern_size}")
        cv2.imshow("chessabord", gray)
        cv2.waitKey(0)
    return ret, corners

def screen_projector_homography(captured_img, cell_w, cell_h, x_cells, y_cells):
    # find corners
    found, corners = find_chessboard_corners(captured_img, (x_cells-1, y_cells-1))
    if not found:
        return None
    # ensure corners are ordered from top-left to bottom-right
    grid = corners.reshape(y_cells-1, x_cells-1, 2)
    if grid[0, 0, 0] > grid[0, -1, 0]:      # first row starts from right
        grid = np.flip(grid, axis=1)        # flip horizontally
    if grid[0, 0, 1] > grid[-1, 0, 1]:      # first column starts from bottom
        grid = np.flip(grid, axis=0)        # flip vertically
    corners = grid.reshape(-1, 2)
    src_points = np.array([(j*cell_w-0.5, i*cell_h-0.5) for i in range(1, y_cells) for j in range(1, x_cells)])
    dst_points = corners
    # only use 4 pairs of points to compute H matrix
    indices = [0, x_cells-2, (y_cells-2)*(x_cells-1), (y_cells-1)*(x_cells-1)-1]
    src = np.array([src_points[i] for i in indices])
    dst = np.array([dst_points[i] for i in indices])
    H, _ = cv2.findHomography(src, dst)
    return H

def grab_projector_image(capture, image1, image2):
    # grab image
    captured = []
    def grab_image():
        ret, img = capture.read()
        if ret:
            captured.append(img)
        else:
            print("can't grab image")
    # project calibration images
    project_images(image1, image2, grab_image)
    if len(captured) == 0:
        return None
    return captured[0]

def calibrate_projectors(capture, image_path):
    PRJ_W, PRJ_H = 1920, 1080
    MIN_OVERLAP = 32
    CELL_W, CELL_H = 240, 180
    X_CELLS, Y_CELLS = PRJ_W//CELL_W, PRJ_H//CELL_H

    # generate chessboard image
    black_img = np.zeros((PRJ_H, PRJ_W, 3), dtype=np.uint8)
    chessbord_img = generate_chessboard_image(PRJ_W, PRJ_H, CELL_W, CELL_H)

    # project & grap the left projector image
    captured_img = grab_projector_image(capture, chessbord_img, black_img)
    if captured_img is None:
        return False
    H31 = screen_projector_homography(captured_img, CELL_W, CELL_H, X_CELLS, Y_CELLS)
    if H31 is None:
        return False
    
    # project & grap the right projector image
    captured_img = grab_projector_image(capture, black_img, chessbord_img)
    if captured_img is None:
        return False
    H32 = screen_projector_homography(captured_img, CELL_W, CELL_H, X_CELLS, Y_CELLS)
    if H32 is None:
        return False
    
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
    project_images(buf1, buf2, None, 3*1000)
    project_images(image1, image2, None, 180*1000)
    return True

def show_camera_view(capture):
    if not capture.isOpened():
        print("can't open camera")
        return False
    while True:
        ok, img = capture.read()
        if not ok:
            print("read frame failed")
            return False
        cv2.putText(img, "Press Esc to exit application", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
        cv2.putText(img, "Press Enter to start calibration", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255))
        cv2.imshow("image", img)
        key = cv2.waitKey(30)
        if key == 27:
            return False
        elif key == 13:
            cv2.destroyAllWindows()
            return True

if __name__ == "__main__":
    capture = cv2.VideoCapture(1)
    do_calib = show_camera_view(capture)
    if not do_calib:
        sys.exit(1)
    image_path = sys.argv[1] if len(sys.argv) > 1 else None 
    calibrate_projectors(capture, image_path)
