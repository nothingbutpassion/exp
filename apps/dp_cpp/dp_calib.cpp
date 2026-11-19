#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "config.h"
#include "calib_parameters.h"
#include "display_helper.h"
#include "chessboard_calibrator.h"

using namespace cv;
using namespace std;

Mat demoImage(int w, int h) {
    Mat img = Mat::zeros(h, w, CV_8UC3);
    int d = 4;
    img(Rect(0,   0,   w,   d)).setTo(Scalar(0, 0, 255));
    img(Rect(0,   h-d, w,   d)).setTo(Scalar(0, 0, 255));
    img(Rect(0,   0,   d,   h)).setTo(Scalar(0, 255, 0));
    img(Rect(w-d, 0,   d,   h)).setTo(Scalar(0, 255, 0));
    for (int i=1; i < 4; i++) {
        img(Rect(0, i*h/4-d/2, w, d)).setTo(Scalar(0, 255, 255));
        img(Rect(i*w/4-d/2, 0, d, h)).setTo(Scalar(0, 255, 255));
    }
    int r1 = min(w/2, h/2) - d;
    int r2 = min(w/4, h/4) - d;
    circle(img, Point(w/2, h/2), r1, Scalar(255, 255, 0), d, LINE_AA);
    circle(img, Point(w/2, h/2), r2, Scalar(255, 255, 0), d, LINE_AA);
    return img;
}

Mat loadImage(const char* path, int w, int h) {
    Mat img = imread(path);
    if (img.data && (img.cols != w || img.rows != h)) {
        resize(img, img, Size(w, h));
    }
    return img;
}

Mat smoothStepMask(int width, int height, int edge0, int edge1) {
    Mat x(1, width, CV_32FC3);
    for (int c = 0; c < width; c++)
        x.at<Point3f>(0, c) = Point3f(float(c), float(c), float(c));

    // t = clamp((x - edge0) / (edge1 - edge0), 0, 1)
    Mat t = (x - edge0) * (1.0f / (edge1 - edge0));
    cv::max(t, 0.0f, t);
    cv::min(t, 1.0f, t);

    // smoothstep: t*t*(3 - 2t)
    Mat row = t.mul(t).mul(3.0f - 2.0f*t);

    // tile 1 x width to height x width
    Mat mask;
    repeat(row, height, 1, mask);
    return mask;
}

void splitBuffers(const Mat& src, int buf_w, int prj_w, int prj_h,  Mat& left, Mat& right) {
    Mat buffer = src;
    if (buffer.cols != buf_w || buffer.rows != Projector::height)
        resize(buffer, buffer, Size(buf_w, Projector::height));
    Mat img1, img2;
    buffer(Rect(0, 0, prj_w, prj_h)).copyTo(img1);
    buffer(Rect(buf_w-prj_w, 0, prj_w, prj_h)).copyTo(img2);
    int overlap_w = prj_w*2 - buf_w;
    Mat mask1 = smoothStepMask(prj_w, prj_h, prj_w, prj_w-overlap_w);
    Mat mask2 = smoothStepMask(prj_w, prj_h, 0, overlap_w);
    img1.convertTo(img1, CV_32F);
    img2.convertTo(img2, CV_32F);
    multiply(img1, mask1, left);
    multiply(img2, mask2, right);
    left.convertTo(left, CV_8U);
    right.convertTo(right, CV_8U);
}


bool showCameaView(VideoCapture& capture, const Rect& roi) {
    bool calib = false;
    if (!capture.isOpened()) {
        LOGE("can't open camera");
        return calib;
    }
    if (Camera::width >= 1920 || Camera::height >= 1080)
        capture.set(CAP_PROP_FOURCC, VideoWriter::fourcc('M', 'J', 'P', 'G'));
    capture.set(CAP_PROP_FRAME_WIDTH, Camera::width);
    capture.set(CAP_PROP_FRAME_HEIGHT, Camera::height);
    int w = (int) capture.get(CAP_PROP_FRAME_WIDTH);
    int h = (int) capture.get(CAP_PROP_FRAME_HEIGHT);
    LOGI("camera frame size: %d x %d", w, h);
    Mat img;
    while (capture.read(img)) {
        rectangle(img, roi, Scalar(0, 255, 0), 2, LINE_AA);
        putText(img, "Press Esc to exit application", Point(10, 30), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
        putText(img, "Press Enter to start calibration", Point(10, 60), FONT_HERSHEY_SIMPLEX, 1.0, Scalar(0, 255, 0), 2);
        imshow("Dual-projector calibrator", img);
        auto key = waitKey(30);
        if (key == 27)      // ESC 
            break;
        if (key == 13) {    // Enter
            calib = true;
            break;
        }
    }
    destroyAllWindows();
    return calib;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        LOGE("Usage: %s <param_path> [image_path]", argv[0]);
        return -1;
    }
    const char* param_path = argv[1];
    const char* image_path = argc > 2 ? argv[2]: nullptr;

    VideoCapture capture(1);
    const Rect roi = {160, 120, 320, 240};
    if (!showCameaView(capture, roi))
        return -1;
    
    ChessboardCalibrator calibrator;
    Mat pattern = calibrator.get_pattern_image();
    
    DisplayHelper displayer;
    displayer.init();
    displayer.setPattern(pattern);

    Mat cap1, cap2;
    displayer.showPattern(0, 5);
    for (int i=0; i < 5; i++)
        capture.read(cap1);
    displayer.showPattern(1, 3);
    for (int i=0; i < 5; i++)
        capture.read(cap2);

    CalibParameters params;
    if (!calibrator.calibrate(cap1, cap2, roi, params)) {
        LOGE("calibration failed");
        return -1;
    }
    LOGI("calibration ok");
    if (!params.save(param_path)) {
        LOGE("save calib params to %s failed", param_path);
        return -1;
    }
    if (!params.load(param_path)) {
        LOGE("load calib params from %s failed", param_path);
        return -1;
    }
    int buf_w = 2*Projector::width - params.overlap;
    Mat image;
    if (image_path)
        image = loadImage(image_path, buf_w, Projector::height);
    else
        image = demoImage(buf_w, Projector::height);
    Mat buf1, buf2;
    splitBuffers(image, buf_w, Projector::width, Projector::height, buf1, buf2);
    Mat left, right;
    warpPerspective(buf1, left,  params.h_p2b[0], buf1.size(), WARP_INVERSE_MAP);
    warpPerspective(buf2, right, params.h_p2b[1], buf2.size(), WARP_INVERSE_MAP);
    displayer.setImages(left, right);
    displayer.showImages(3);
    displayer.release();
    capture.release();
    return 0;
}


