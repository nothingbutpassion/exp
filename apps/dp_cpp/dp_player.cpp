// dp_player.cpp
//
// Dual-projector player with live face detection:
//   - plays a video through the calibration pipeline of dp_render: one
//     fullscreen window per projector, warped by H_p2b with overlap blending
//   - runs FaceDetector on camera 1 on a background thread and shows the
//     latest detections in a preview window
//
// Usage:
//   dp_player <param_path> <video_path> [model_path]
//     model_path defaults to models/face_detection_yunet_2023mar.onnx
//     -h, --help   show this help

#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "calib_parameters.h"
#include "face_detector.h"
#include "log.h"
#include "video_render.h"

namespace {

const char* kDefaultModel = "models/face_detection_yunet_2026may.onnx";
const char* kWindow = "Face detector camera preview";
const int kCameraIndex = 1;
const cv::Size kInputSize(640/2, 480/2);

struct Options {
    std::string paramPath;
    std::string videoPath;
    std::string modelPath;
};

enum class ParseResult { Ok, ExitSuccess, Error };

void printUsage(const char* program) {
    std::printf(
        "Usage: %s <param_path> <video_path> [model_path]\n"
        "  param_path  calibration parameters (YAML)\n"
        "  video_path  video to play through the projectors\n"
        "  model_path  ONNX face model (default: %s)\n"
        "  -h, --help  show this help\n",
        program, kDefaultModel);
}

ParseResult parseOptions(int argc, char* argv[], Options& options) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            printUsage(argv[0]);
            return ParseResult::ExitSuccess;
        }
        if (!arg.empty() && arg[0] == '-') {
            LOGE("unknown argument: %s", arg.c_str());
            printUsage(argv[0]);
            return ParseResult::Error;
        }
        if (options.paramPath.empty())
            options.paramPath = arg;
        else if (options.videoPath.empty())
            options.videoPath = arg;
        else if (options.modelPath.empty())
            options.modelPath = arg;
        else {
            LOGE("unexpected argument: %s", arg.c_str());
            printUsage(argv[0]);
            return ParseResult::Error;
        }
    }
    if (options.paramPath.empty() || options.videoPath.empty()) {
        LOGE("calibration file and video path are required");
        printUsage(argv[0]);
        return ParseResult::Error;
    }
    if (options.modelPath.empty())
        options.modelPath = kDefaultModel;
    return ParseResult::Ok;
}

void drawDetections(cv::Mat& frame, const std::vector<Detection>& faces) {
    for (const Detection& face : faces) {
        cv::rectangle(frame, face.box, cv::Scalar(0, 255, 0), 2);

        char score[32];
        std::snprintf(score, sizeof(score), "%.2f", face.score);
        cv::putText(frame, score,
                    cv::Point(cvRound(face.box.x),
                              std::max(15, cvRound(face.box.y) - 6)),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);

        for (const cv::Point2f& landmark : face.landmarks)
            cv::circle(frame, landmark, 2, cv::Scalar(0, 0, 255), -1);
    }
}

// Shows the latest camera frame with the latest detections; returns false
// when the user asked to quit (Esc or q).
bool showPreview(FaceDetector& detector) {
    // Both come from the same processed frame; the detector keeps the
    // camera fed on its own thread.
    cv::Mat frame = detector.getLatestImage();
    const std::vector<Detection> faces = detector.getLatestDetection();

    if (!frame.empty()) {
        drawDetections(frame, faces);

        char status[64];
        std::snprintf(status, sizeof(status), "faces: %d",
                      static_cast<int>(faces.size()));
        cv::putText(frame, status, cv::Point(10, 30),
                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 255), 2);

        cv::imshow(kWindow, frame);
    }

    const int key = cv::waitKey(1);
    return key != 27 && key != 'q';
}

void toVideoBoxes(const CalibParameters& params, const cv::Size& vsize, const std::vector<cv::Rect2f>& src, std::vector<cv::Rect2f>& dst) {
    cv::Rect bounds = {params.r_screen[0].tl(), params.r_screen[1].br()};
    float sw = float(bounds.width)/vsize.width;
    float sh = float(bounds.height)/vsize.height;
    float s = std::min(sw, sh);

    float vw = s*vsize.width;
    float vh = s*vsize.height;
    float vx = 0.5f*(bounds.width - vw);
    float vy = 0.5f*(bounds.height - vh);

    dst.clear();
    dst.reserve(src.size());
    for (const cv::Rect2f& box: src) {
        float x1 = vsize.width*(box.x - bounds.x - vx)/vw;
        float y1 = vsize.height*(box.y - bounds.y - vy)/vh;
        float x2 = vsize.width*(box.x + box.width - bounds.x - vx)/vw;
        float y2 = vsize.height*(box.y + box.height - bounds.y - vy)/vh;
        dst.push_back(cv::Rect2f(cv::Point2f(x1, y1), cv::Point2f(x2, y2)));
    }
}


} // namespace

int main(int argc, char* argv[]) {
    Options options;
    const ParseResult parsed = parseOptions(argc, argv, options);
    if (parsed == ParseResult::ExitSuccess)
        return 0;
    if (parsed == ParseResult::Error)
        return 1;

    CalibParameters params;
    if (!params.load(options.paramPath.c_str())) {
        LOGE("load calib params from %s failed", options.paramPath.c_str());
        return 1;
    }

    FaceDetector detector;
    if (!detector.init(options.modelPath, kCameraIndex, kInputSize))
        return 1;

    VideoRender render;
    if (!render.init()) {
        LOGE("video render init failed (are both projector displays connected?)");
        detector.release();
        return 1;
    }
    render.setParameters(params);
    render.show();

    cv::namedWindow(kWindow, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);

    cv::VideoCapture capture(options.videoPath);
    if (!capture.isOpened()) {
        LOGE("cannot open video %s", options.videoPath.c_str());
        render.release();
        detector.release();
        return 1;
    }

    const int fps = 30;
    const double invFps = 1.0 / fps;

    // Keep blacking a face after the detector stops reporting it, to smooth
    // over single-frame detection dropouts (a missed frame would otherwise
    // briefly unblank the region).
    const double kBoxHoldSeconds = 0.25;
    std::vector<cv::Rect2f> heldBoxes;
    int64 lastBoxTick = 0;

    cv::Mat frame;
    double duration = 0.0;
    int64_t nFrames = 0;
    bool running = true;
    while (running && capture.read(frame) && !frame.empty()) {
        const int64 start = cv::getTickCount();

        // Fill face boxes with black color
        const std::vector<Detection> detections = detector.getLatestDetection();
        std::vector<cv::Rect2f> boxes;
        for (const Detection& d: detections) {
            if (d.score > 0.9)
                boxes.push_back(d.box);
        }
        std::vector<cv::Rect2f> vboxes;
        toVideoBoxes(params, frame.size(), boxes, vboxes);

        const int64 now = cv::getTickCount();
        if (!vboxes.empty()) {
            heldBoxes = vboxes;
            lastBoxTick = now;
        } else if (!heldBoxes.empty() &&
                   double(now - lastBoxTick) / cv::getTickFrequency() > kBoxHoldSeconds) {
            heldBoxes.clear();
        }

        for (const cv::Rect2f& box: heldBoxes) {
            if (box.br().x < 0 || box.br().y < 0)
                continue;
            if (box.tl().x > frame.cols || box.tl().y > frame.rows)
                continue;
            cv::rectangle(frame, box, cv::Scalar(255,0,255), cv::FILLED);
        } 

        render.setFrame(frame);
        render.drawFrame();
        running = showPreview(detector);

        const double seconds = double(cv::getTickCount() - start) / cv::getTickFrequency();
        if (invFps > seconds)
            render.pollEvents(invFps - seconds);
        duration += seconds;
        nFrames += 1;
        if (duration > 1) {
            //LOGD("fps: %.2f", double(nFrames) / duration);
            duration = 0;
            nFrames = 0;
        }
    }

    LOGI("playback finished");
    detector.release();
    cv::destroyWindow(kWindow);
    render.release();
    return 0;
}