import sys
import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSlider, QLabel, QStatusBar
from PyQt6.QtCore import Qt, QPointF, pyqtSignal, QRectF, QTimer, QEventLoop
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QPolygonF, QFont, QImage, QPixmap, QShortcut, QKeySequence

class QuadWidget(QWidget):
    # Signal emitted when any corner coordinates change
    coordsChanged = pyqtSignal(int, list)  # widget_id, list of corner coordinates
    
    def __init__(self, widget_id=0):
        super().__init__()
        
        self.widget_id = widget_id

        # Selection and drag state
        self.selected_corner = -1  # -1 means no corner selected
        self.drag_start = None
        self.is_dragging = False
        
        # Colors
        self.selected_color = QColor(255, 150, 50)  # Red for selected
        self.normal_color = QColor(50, 150, 255)    # Blue for normal
        self.quad_color = QBrush(QColor(200, 220, 255, 100))
        self.quad_pen = QPen(QColor(100, 130, 200), 2)
        
        self.corner_size = 6
        self.selection_distance = 11
        self.move_step = 1

        # Settings
        self.prj_w = 1920
        self.prj_h = 1080
        self.scale = 0.25
        self.overlap = 960

        # Initialize quadrilateral corners (relative to center)
        self.corners = [
            QPointF(0, 0),                      # Top-left
            QPointF(self.prj_w, 0),             # Top-right
            QPointF(self.prj_w, self.prj_h),    # Bottom-right
            QPointF(0, self.prj_h)              # Bottom-left
        ]
        
        # Mouse tracking
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        # Initialize signal emission
        self._emit_coords_changed()
    
    def get_widget_corners(self, corners):
        center_x = self.width() / 2
        center_y = self.height() / 2
        return [
            QPointF(
                center_x + (corner.x() - self.prj_w/2)*self.scale, 
                center_y + (corner.y() - self.prj_h/2)*self.scale
            ) 
            for corner in corners
        ]
    
    def pixel_to_relative(self, pixel_point):
        center_x = self.width() / 2
        center_y = self.height() / 2
        return QPointF(
            self.prj_w/2 + (pixel_point.x() - center_x)/self.scale,
            self.prj_h/2 + (pixel_point.y() - center_y)/self.scale
        )
    
    def find_nearest_corner(self, point):
        widget_corners = self.get_widget_corners(self.corners)
        nearest = -1
        min_distance = float('inf')
        for i, corner in enumerate(widget_corners):
            dx = point.x() - corner.x()
            dy = point.y() - corner.y()
            distance = (dx * dx + dy * dy) ** 0.5
            if distance < min_distance:
                min_distance = distance
                nearest = i
        
        if min_distance < self.selection_distance:
            return nearest
        return -1

    def set_overlap(self, overlap):
        overlap = max(0, min(overlap, self.prj_w))
        if overlap != self.overlap:
            self.overlap = overlap
            self.update()
    
    def _emit_coords_changed(self):
        coords = [(int(c.x()), int(c.y())) for c in self.corners]
        self.coordsChanged.emit(self.widget_id, coords)
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw background grid (optional)
        painter.setPen(QPen(QColor(230, 230, 230), 1))
        for x in range(0, self.width(), 20):
            painter.drawLine(x, 0, x, self.height())
        for y in range(0, self.height(), 20):
            painter.drawLine(0, y, self.width(), y)
        # Draw center cross
        center_x = int(self.width() / 2)
        center_y = int(self.height() / 2)
        painter.setPen(QPen(QColor(200, 200, 200, 100), 1))
        painter.drawLine(center_x - 20, center_y, center_x + 20, center_y)
        painter.drawLine(center_x, center_y - 20, center_x, center_y + 20)
        
        widget_corners = self.get_widget_corners(self.corners)
        
        # Draw the quadrilateral
        painter.setPen(self.quad_pen)
        painter.setBrush(self.quad_color)
        quad = QPolygonF(widget_corners)
        painter.drawPolygon(quad)
        p1, p2, p3, p4 = widget_corners
        if self.widget_id == 1:             # left quad
            ratio = 1 - self.overlap/self.prj_w
        else:                               # right quad
            ratio = self.overlap/self.prj_w
        start = p1 + (p2-p1)*ratio
        end   = p4 + (p3-p4)*ratio
        painter.drawLine(start, end)

        # Draw corner points
        labels = ['TL', 'TR', 'BR', 'BL']
        font = QFont()
        font.setBold(True)
        painter.setFont(font)
        
        for i, (point, label) in enumerate(zip(widget_corners, labels)):
            # Determine corner color
            if i == self.selected_corner:
                color = self.selected_color 
            else:
                color = self.normal_color
            
            # Draw corner circle
            painter.setPen(QPen(color, 3))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(point, self.corner_size, self.corner_size)
            
            # Draw corner label
            painter.setPen(QPen(QColor(0, 0, 0), 1))
            painter.drawText(
                QPointF(point.x() + self.corner_size + 5, point.y() + 5),
                label
            )
        
        # Draw coordinates info
        painter.setPen(QPen(QColor(0, 0, 0, 150), 1))
        font.setPointSize(8)
        painter.setFont(font)
        
        info_start_y = 15
        for i, (corner, label) in enumerate(zip(self.corners, labels)):
            painter.drawText(
                10,
                info_start_y + i * 18,
                f"{label}: ({corner.x():.0f}, {corner.y():.0f})"
            )
    
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            corner_idx = self.find_nearest_corner(event.position())
            if corner_idx >= 0:
                self.selected_corner = corner_idx
                self.drag_start = event.position()
                self.is_dragging = True
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                self.update()

    def mouseMoveEvent(self, event):
        if self.is_dragging and self.selected_corner >= 0 and self.drag_start is not None:
            current_pos = event.position()
            delta = current_pos - self.drag_start
            widget_corners = self.get_widget_corners(self.corners)
            new_widget_pos = widget_corners[self.selected_corner] + delta
            self.corners[self.selected_corner] = self.pixel_to_relative(new_widget_pos)
            self.drag_start = current_pos
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.is_dragging:
            self.is_dragging = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.update()
            self._emit_coords_changed()

    def keyPressEvent(self, event):
        if self.selected_corner < 0:
            return
        corner = self.corners[self.selected_corner]
        moved = False
        if event.key() == Qt.Key.Key_Up:
            corner.setY(corner.y() - self.move_step)
            moved = True
        elif event.key() == Qt.Key.Key_Down:
            corner.setY(corner.y() + self.move_step)
            moved = True
        elif event.key() == Qt.Key.Key_Left:
            corner.setX(corner.x() - self.move_step)
            moved = True
        elif event.key() == Qt.Key.Key_Right:
            corner.setX(corner.x() + self.move_step)
            moved = True
        if moved:
            self.update()
            self._emit_coords_changed()
    
    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.selected_corner = -1
            self.is_dragging = False
            self.drag_start = None
            self.setCursor(Qt.CursorShape.ArrowCursor)
            self.update()

class MainWindow(QMainWindow):
    
    def __init__(self, app):
        super().__init__()
        self.setWindowTitle("Dual-projector calibrator")
        self.setGeometry(100, 100, 1100, 600)
        
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setMinimum(0)
        slider.setMaximum(1920)
        slider.setValue(960)
        slider.valueChanged.connect(self.on_overlap_changed)
        self.overlap_label = QLabel(f"Overlap: {slider.value()}")
        main_layout.addWidget(self.overlap_label)
        main_layout.addWidget(slider)

        quad_layout = QHBoxLayout()
        self.left_quad = QuadWidget(widget_id=1)
        self.right_quad = QuadWidget(widget_id=2)
        quad_layout.addWidget(self.left_quad, 1)
        quad_layout.addWidget(self.right_quad, 1)
        main_layout.addLayout(quad_layout, 1)
        self.left_quad.coordsChanged.connect(self.on_coords_changed)
        self.right_quad.coordsChanged.connect(self.on_coords_changed)

        # create windows for projector screens
        screens = app.screens()
        for i, s in enumerate(screens):
            geo = s.geometry()
            dpr = s.devicePixelRatio()
            print(f"screen {i}: geometry={geo}, dpr={dpr}, physical w/h={geo.width()*dpr:.0f} x {geo.height()*dpr:.0f}")
        screens = screens[1:]
        screens = list(reversed(screens))
        # screens.sort(key=lambda s: s.geometry().x())
        wins = [ QWidget() for _ in screens]
        labels = [QLabel(win) for win in wins]
        for idx, screen in enumerate(screens):
            wins[idx].setScreen(screen)
            wins[idx].setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
            wins[idx].setGeometry(screen.geometry())
            labels[idx].setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.wins = wins
        self.labels = labels
        self.prj_w = 1920
        self.prj_h = 1080
        self.overlap = 960
        self.H1 = np.identity(3) 
        self.H2 = np.identity(3)
        self.buffer = self.image_buffer(2*self.prj_w, self.prj_h)

    def closeEvent(self, event):
        for win in self.wins:
            win.close()
        buf_w = 2*self.prj_w - self.overlap
        print(f"C = {buf_w} O = {self.overlap} R = {buf_w - self.prj_w}")
        pts1 = [(int(c.x()), int(c.y())) for c in self.left_quad.corners]
        pts2 = [(int(c.x()), int(c.y())) for c in self.right_quad.corners]
        print("P1-P4:" + " ".join([f"({p[0]:>5}, {p[1]:<5})" for p in pts1]))
        print("P5-P8:" + " ".join([f"({p[0]:>5}, {p[1]:<5})" for p in pts2]))

    def on_overlap_changed(self, value):
        self.overlap_label.setText(f"Overlap: {value}")
        self.overlap = value
        self.left_quad.set_overlap(value)
        self.right_quad.set_overlap(value)
        print(f"overlap changed: {value}")
        src1 = np.array([(int(c.x()), int(c.y())) for c in self.left_quad.corners])
        src2 = np.array([(int(c.x()), int(c.y())) for c in self.right_quad.corners])
        dst = np.array([(0, 0), (self.prj_w, 0), (self.prj_w, self.prj_h), (0, self.prj_h)])
        self.H1, _ = cv2.findHomography(src1, dst)
        self.H2, _ = cv2.findHomography(src2, dst)
        buf1, buf2 = self.split_buffer()
        img1 = cv2.warpPerspective(buf1, self.H1, (self.prj_w, self.prj_h), flags=cv2.WARP_INVERSE_MAP)
        img2 = cv2.warpPerspective(buf2, self.H2, (self.prj_w, self.prj_h), flags=cv2.WARP_INVERSE_MAP)
        self.project_images([img1, img2])

    def on_coords_changed(self, widget_id, coords):
        print(f"widget {widget_id} coordinates changed:")
        for i, (x, y) in enumerate(coords):
            labels = ['TL', 'TR', 'BR', 'BL']
            print(f"  {labels[i]}: ({x}, {y})")
        src = np.array(coords)
        dst = np.array([(0, 0), (self.prj_w, 0), (self.prj_w, self.prj_h), (0, self.prj_h)])
        if widget_id == 1:
            self.H1, _ = cv2.findHomography(src, dst)
        else:
            self.H2, _ = cv2.findHomography(src, dst)
        buf1, buf2 = self.split_buffer()
        img1 = cv2.warpPerspective(buf1, self.H1, (self.prj_w, self.prj_h), flags=cv2.WARP_INVERSE_MAP)
        img2 = cv2.warpPerspective(buf2, self.H2, (self.prj_w, self.prj_h), flags=cv2.WARP_INVERSE_MAP)
        self.project_images([img1, img2])

    def project_images(self, images):
        for win, label, image in zip(self.wins, self.labels, images):
            h, w, c = image.shape
            bytes_per_line = c * w
            q_img = QImage(image.data, w, h, bytes_per_line, QImage.Format.Format_BGR888)
            pixmap = QPixmap.fromImage(q_img)
            label.setPixmap(pixmap)
            win.show()

    def image_buffer(self, w, h):
        img = np.zeros((h, w, 3), dtype=np.uint8)
        # bounding box
        d = 4
        img[:d]     = (0, 0, 255)    # top
        img[-d:]    = (0, 0, 255)    # bottom
        img[:,:d]   = (0, 255, 0)    # left
        img[:,-d:]  = (0, 255, 0)    # right
        # lines
        hd = 2
        nx, ny = 32, 9
        for i in range(1, nx):
            img[:,i*w//nx-hd:i*w//nx+hd]    = (0, 255, 255)
        for i in range(1, ny):
            img[i*h//ny-hd:i*h//ny+hd]      = (0, 255, 255)
        # circles
        for k in range(1, min(nx, ny)//2+1):
            cv2.circle(img, (w//2, h//2), min(k*w//nx, k*h//ny) - d, (255, 255, 0), d, cv2.LINE_AA)

        # for i in range(1, 4):
        #     img[:,i*w//4-hd:i*w//4+hd]    = (0, 255, 255)
        #     img[i*h//4-hd:i*h//4+hd]      = (0, 255, 255)
        # cv2.circle(img, (w//2, h//2), min(w//2, h//2) - d, (255, 255, 0), d, cv2.LINE_AA)
        # cv2.circle(img, (w//2, h//2), min(w//4, h//4) - d, (255, 255, 0), d, cv2.LINE_AA)
        return img.astype(np.float64)

    def split_buffer(self):
        buf_w = 2*self.prj_w - self.overlap
        x1 = (2*self.prj_w - buf_w)//2
        x2 = x1 + self.prj_w - self.overlap
        buf1 = np.copy(self.buffer[:,x1:x1+self.prj_w])
        buf2 = np.copy(self.buffer[:,x2:x2+self.prj_w])
        buf1[:,self.prj_w-self.overlap:] *= self.smoothstep_mask(self.overlap, 1, 0)
        buf2[:,:self.overlap] *= self.smoothstep_mask(self.overlap, 0, 1)
        return buf1.astype(np.uint8), buf2.astype(np.uint8)

    def smoothstep_mask(self, w, s, e):
        t = np.linspace(s, e, w)
        t = 3*t**2 - 2*t**3
        return t.reshape(w, 1)

def main():
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow(app)
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()