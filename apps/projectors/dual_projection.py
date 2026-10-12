import math
import cv2
import numpy as np

def load_image(image_file, target_width, target_height):
    img = cv2.imread(image_file, cv2.IMREAD_COLOR)
    if img is None:
        print(f"can't load {image_file}")
        return img
    return cv2.resize(img, dsize=(target_width, target_height), interpolation=cv2.INTER_CUBIC)

def warp_image(img, H):
    img_h, img_w = img.shape[:2]
    # NOTES:
    # In default, OpenCV internally will compute inverse of H for sampling.
    # If cv2.WARP_INVERSE_MAP specified, OpenCV does not invert your matrix
    return cv2.warpPerspective(img, H, dsize=(img_w, img_h), flags=cv2.WARP_INVERSE_MAP)

def save_calib_data(yaml_path, img_size, img_roi):
    data = {
        "image_size": img_size, 
        "image_roi": img_roi,
    }
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_WRITE)
    for k, v in data.items():
        fs.write(k, v)
    fs.release()
    print(f"saved {yaml_path}")

def load_calib_data(yaml_path):
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    img_size = fs.getNode("image_size").mat 
    img_roi = fs.getNode("image_roi").mat()
    return img_size, img_roi

# sepcial camera matrix for debugging
def projection_matrix(w, h, fov_h, fov_v):
    cx = 0.5*w
    cy = h
    fx = 0.5*w/math.tan(0.5*fov_h)
    fy = h/math.tan(fov_v)
    return np.array([
            [fx, 0,  cx],
            [0,  fy, cy],
            [0,  0,  1 ]
        ])

def tranform_points(src_points, H):
    dst_points = []
    for u, v in src_points:
        x, z, w = H @ [u, v, 1]
        dst_points.append([x/w, z/w])
    return dst_points

def get_screen_regions(H31, H32, prj_w, prj_h):
    # corner points of projector
    corners = [(0, 0), (prj_w-1, 0), (prj_w-1, prj_h-1), (0, prj_h-1)]
    screen_r1 = tranform_points(corners, H31)   # left  projection points in screen
    screen_r2 = tranform_points(corners, H32)   # right projection points in screen
    return screen_r1, screen_r2

def rectify_screen_regions(screen_r1, screen_r2):
    p1, p2, p3, p4 = screen_r1  # left quad in screen
    p5, p6, p7, p8 = screen_r2  # right quad in screen
    # adjust top & bottom points
    if p3[1] < p1[1]:
        p1[1] = p2[1] = p5[1] = p6[1] = min(p1[1], p2[1], p5[1], p6[1])
        p3[1] = p4[1] = p7[1] = p8[1] = max(p3[1], p4[1], p7[1], p8[1])
    else:
        p1[1] = p2[1] = p5[1] = p6[1] = max(p1[1], p2[1], p5[1], p6[1])
        p3[1] = p4[1] = p7[1] = p8[1] = min(p3[1], p4[1], p7[1], p8[1])
    # adjust left region
    p1[0] = p4[0] = max(p1[0], p4[0])
    p2[0] = p3[0] = min(p2[0], p3[0])
    # adjust right region
    p5[0] = p8[0] = max(p5[0], p8[0])
    p6[0] = p7[0] = min(p6[0], p7[0])
    return screen_r1, screen_r2

def get_buffer_width(overlap_r, prj_w, prj_h):
    p1, _, p3, _ = overlap_r
    screen_w = p3[0] - p1[0]
    screen_h = abs(p3[1] - p1[1])
    overlap_w = round(prj_h*screen_w/screen_h)
    overlap_w = max(0, min(overlap_w, prj_w))
    buf_w = 2*prj_w - overlap_w
    return buf_w

def adjust_screen_regions(r1, r2, prj_w, prj_h, min_overlap=32):
    p1, p2, p3, p4 = r1
    p5, p6, p7, p8 = r2
    h = abs(p4[1] - p1[1])          # screen height
    s = prj_h/h                     # scale to prj_h

    overlap_r = [p5, p2, p3, p8]
    overlap_w = s*(p2[0] - p5[0])

    # overlap is too small
    if overlap_w < min_overlap:
        return r1, r2, overlap_r

    w1 = s*(p2[0] - p1[0]) 
    w2 = s*(p6[0] - p5[0])

    # left quad is smaller
    if w1 < w2:
        left_w = s*(p5[0] - p1[0])
        if left_w + min_overlap < prj_w:
            # extends to left bounding   
            p2[0] = p3[0] = min(p1[0] + prj_w/s, p2[0]) 
        else:
            p2[0] = p3[0] = min(p5[0] + min_overlap/s, p2[0]) 
            p1[0] = p4[0] = max(p2[0] - prj_w/s, p1[0])
        p6[0] = p7[0] = min(p5[0] + prj_w/s, p6[0])
        return r1, r2, overlap_r
    
    # right quad is smaller
    right_w = s*(p6[0] - p2[0])
    if right_w + min_overlap < prj_w:
        # extends to right bounding
        p5[0] = p8[0] = max(p7[0] - prj_w/s, p5[0]) 
    else:
        p5[0] = p8[0] = max(p2[0] - min_overlap/s, p5[0]) 
        p6[0] = p7[0] = min(p5[0] + prj_w/s, p6[0])
    p1[0] = p4[0] = max(p2[0] - prj_w/s, p1[0])
    return r1, r2, overlap_r

def smoothstep_mask(w, s, e):
    t = np.linspace(s, e, w)
    t = 3*t**2 - 2*t**3
    return t.reshape(w, 1)

def split_buffers(image, prj_w, prj_h):
    buf_w = image.shape[1]; 
    x = buf_w - prj_w;      # (x, 0) is top left of right buffer
    ow = prj_w - x;         # overlap width

    # left buffer
    buf1 = np.zeros((prj_h, prj_w, 3))
    buf1[:,0:prj_w,:] = image[:,0:prj_w,:]
    buf1[:,x:x+ow,:] *= smoothstep_mask(ow, 1, 0)
    buf1 = np.array(buf1, dtype=np.uint8)

    # right buffer
    buf2 = np.zeros((prj_h, prj_w, 3))
    buf2[:,0:prj_w,:] = image[:,x:x+prj_w,:]
    buf2[:,0:ow,:] *= smoothstep_mask(ow, 0, 1)
    buf2 = np.array(buf2, dtype=np.uint8)
    return buf1, buf2

def screen_buffer_homographies(overlap_r, prj_w, prj_h):
    p1, _, p3, _ = overlap_r
    h = p3[1] - p1[1]       # p3[1] - p1[1] may be negative
    sx = abs(prj_h/h);      # sx is positive 
    sz = prj_h/h;           # sz may be negative
    p0 = [p3[0] - abs(h*prj_w/prj_h), p1[1]]
    # homography: screen (x, z, 1) -> image (u, v, 1)
    Hl3 = np.array([
        [sx,  0,  -sx*p0[0]],
        [0,  sz,  -sz*p0[1]],
        [0,   0,   1       ]
        ])
    Hr3 = np.array([
        [sx,  0,   -sx*p1[0]],
        [0,   sz,  -sz*p1[1]],
        [0,   0,   1        ]
        ])
    return Hl3, Hr3
