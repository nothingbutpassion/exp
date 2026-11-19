#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include "chessboard_calibrator.h"
#include "config.h"
#include "log.h"

using namespace std;
using namespace cv;

namespace {
constexpr int   kPatternWidth    = 6;
constexpr int   kPatternHeight   = 4;


void dumpMat(const char* prefix, const Mat& m) {
    string s;
    for (int y=0; y < m.rows; y++) {
        s += "\n";
        for (int x=0; x < m.cols; x++) {
            double v = m.at<double>(y, x);
            s += ("  " + to_string(v));
        }
    }
    LOGI("%s: %s", prefix, s.c_str());
}

} // namespace


ChessboardCalibrator::ChessboardCalibrator() {
    pattern_image_ = generate_chessboard_image(Size(Projector::width, Projector::height), Size(kPatternWidth, kPatternHeight));
    pattern_corners_ = get_pattern_corners(Size(Projector::width, Projector::height), Size(kPatternWidth, kPatternHeight));
}

ChessboardCalibrator::~ChessboardCalibrator() {
    pattern_image_.release();
    pattern_corners_.clear();
    for (int i=0; i < 2; i++) {
        H_p2s_[i].release();
        H_s2b_[i].release();
        H_p2b_[i].release();
        screen_quads_[i].clear();
        screen_rois_[i].clear();
    }
}

bool ChessboardCalibrator::calibrate(const Mat& left_image, const Mat& right_image, const Rect& roi, CalibParameters& params) {
    const char* names[] = {"left projector", "right projector"};
    Mat images[2];
    left_image(roi).copyTo(images[0]);
    right_image(roi).copyTo(images[1]);
    vector<Point2f> pattern_region = {
        {0, 0}, {Projector::width-1, 0}, {Projector::width-1, Projector::height-1}, {0, Projector::height-1}
    };
    for (int i = 0; i < 2; i++) {
         LOGI("%s: find corners ...", names[i]);
        vector<Point2f> corners = find_chessboard_corners(images[i], Size(kPatternWidth-1, kPatternHeight-1));
        if (corners.size() < 4) {
            LOGE("%s: find corners faild, count=%llu", names[i], corners.size());
            return false;
        }
        LOGI("%s: find corners ok, count=%llu", names[i], corners.size());
        // Add offset for each corner
        for (Point2f& c: corners) {
            c.x += roi.x;
            c.y += roi.y;
        }
        // NOTES: Only use 4 pairs of points to compute H matrix
        int indices[4] = {
            0,                                      kPatternWidth-2, 
            (kPatternHeight-2)*(kPatternWidth-1),   (kPatternHeight-1)*(kPatternWidth-1)-1
        };
        vector<Point2f> src(4);
        vector<Point2f> dst(4);
        for (int k=0; k < 4; k++) {
            src[k] = pattern_corners_[indices[k]];
            dst[k] = corners[indices[k]];
        }
        Mat H = findHomography(src, dst);
        if (H.empty()) {
            LOGE("%s: find homography matrix failed", names[i]);
            return false;
        }
        H.copyTo(H_p2s_[i]);
        perspectiveTransform(pattern_region, screen_quads_[i], H);
        LOGI("%s: find homography matrix ok", names[i]);
    }
    vector<Point2f> screen_rects[2];
    dp::rectify_screen_region(screen_quads_, screen_rects);
    dp::compute_display_region(screen_rects, screen_rois_);
    dp::screen_buffer_homographies(screen_rois_, H_s2b_);
    H_p2b_[0] = H_s2b_[0] * H_p2s_[0];
    H_p2b_[1] = H_s2b_[1] * H_p2s_[1];

    // Only for debugging!
    dumpMat("H_p2s[0]", H_p2s_[0]);
    dumpMat("H_p2s[1]", H_p2s_[1]);
    dumpMat("H_s2b[0]", H_s2b_[0]);
    dumpMat("H_s2b[1]", H_s2b_[1]);
    dumpMat("H_p2b[0]", H_p2b_[0]);
    dumpMat("H_p2b[1]", H_p2b_[1]);
    // for (int i=0; i < 2; i++) {
    //     for (int j=0; j < 4; j++) {
    //         auto p = screen_quads_[i][j];
    //         circle(images[i], Point(int(p.x), int(p.y)), 3, Scalar(255, 0, 0), 2);
    //         p = screen_rects[i][j];
    //         circle(images[i], Point(int(p.x), int(p.y)), 3, Scalar(0, 255, 0), 2);
    //         p = screen_rois_[i][j];
    //         circle(images[i], Point(int(p.x), int(p.y)), 3, Scalar(0, 0, 255), 2);
    //     }
    // }

    // Update calib params
    params = CalibParameters();
    params.type = 1;
    params.overlap = dp::compute_buffer_overlap(screen_rois_);
    for (int i = 0; i < 2; i++) {
        H_p2b_[i].copyTo(params.h_p2b[i]);

        // NOTES: screen roi may be clipped, so restor it
        auto p1 = screen_rois_[i][0], p3 = screen_rois_[i][2];
        auto h = p3.y - p1.y;
        if (h < 1.f) {
            LOGE("no vertical overlap");
            return false;
        }
        auto s = h/Projector::height;
        if (i == 0)
            p1.x = p3.x - s*Projector::width;
        else
            p3.x = p1.x + s*Projector::width;
        params.r_screen[i] = {int(p1.x), int(p1.y), int(p3.x - p1.x), int(p3.y- p1.y)};
    }
    return true;
}

const Mat& ChessboardCalibrator::get_pattern_image() {
    return pattern_image_;
}

Mat ChessboardCalibrator::generate_chessboard_image(const Size& image_size, const Size& pattern_size) {
    Mat chessboard = Mat::zeros(image_size, CV_8UC3);
    auto cell_w = image_size.width/pattern_size.width;
    auto cell_h = image_size.height/pattern_size.height;
    for (int i=0; i < pattern_size.height; i++) {
        for (int j=0; j < pattern_size.width; j++) {
            if ((i+j) % 2 == 0)
                chessboard(Rect(j*cell_w, i*cell_h, cell_w, cell_h)) = Scalar(255, 255, 255);
        }
    }
    return chessboard;
}

vector<Point2f> ChessboardCalibrator::get_pattern_corners(const Size& image_size, const Size& pattern_size) {
    auto cell_w = image_size.width/pattern_size.width;
    auto cell_h = image_size.height/pattern_size.height;
    vector<Point2f> corners;
    for (int i=1; i < pattern_size.height; i++)
        for (int j=1; j < pattern_size.width; j++)
            corners.push_back(Point2f(float(j*cell_w-0.5), float(i*cell_h-0.5)));
    return corners;
}


vector<Point2f> ChessboardCalibrator::find_chessboard_corners(const Mat& image, const Size& pattern_size) {
    Mat gray;
    if (image.channels() > 1)
        cvtColor(image, gray, COLOR_BGR2GRAY);
    else
        image.copyTo(gray);
    // medianBlur(gray, gray, 3);
    // GaussianBlur(gray, gray, Size(3, 3), 0.7);
    // NOTES:
    // CALIB_CB_NORMALIZE_IMAGE - Use adaptive thresholding to convert image to black/white
    // CALIB_CB_EXHAUSTIVE - Run an exhaustive search to improve detection rate.
    // CALIB_CB_ACCURACY Up sample input image to improve sub-pixel accuracy due to aliasing effects
    // int flags = CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_EXHAUSTIVE | CALIB_CB_ACCURACY;
    vector<Point2f> corners;
    bool found = findChessboardCornersSB(gray, pattern_size, corners);
    if (!found) {
        LOGE("findChessboardCornersSB failed");
        corners.clear();
        return corners;
    } 
    
    TermCriteria criteria(TermCriteria::MAX_ITER|TermCriteria::EPS, 30, 0.001);
    cornerSubPix(gray, corners, Size(11, 11), Size(-1, -1), criteria);
    // NOTES: 
    // Make sure the corners are ordered from top-left to bottom-righ
    auto w = pattern_size.width;
    auto h = pattern_size.height;
    auto s = corners[0];
    auto e = corners[w*h-1];
    if (s.y > e.y || s.x > e.x) {
        int row_start = s.y > e.y ? h-1 : 0;
        int row_end   = s.y > e.y ? -1  : h;
        int row_step  = s.y > e.y ? -1  : 1;
        int col_start = s.x > e.x ? w-1 : 0;
        int col_end   = s.x > e.x ? -1  : w;
        int col_step  = s.x > e.x ? -1  : 1;
        vector<Point2f> ordered;
        for (int r=row_start; r != row_end; r += row_step) {
            for (int c=col_start; c != col_end; c += col_step) {
                ordered.push_back(corners[r*w+c]);
            }
        }
        corners.swap(ordered);
    }
    return corners;
}


