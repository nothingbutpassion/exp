// face_detector.cpp
//
// See face_detector.h for the threading model.

#include <chrono>
#include <exception>

#include <opencv2/imgproc.hpp>

#include "config.h"
#include "face_detector.h"
#include "log.h"

FaceDetector::~FaceDetector() {
    release();
}

bool FaceDetector::init(const std::string& modelPath, int cameraIndex,
                        const cv::Size& inputSize) {
    release();

    if (!yunet_.init(modelPath, inputSize))
        return false;

    if (cameraIndex >= 0) {
        camera_.open(cameraIndex);
        if (camera_.isOpened()) {
            // Keep the capture in the coordinate space the calibration was
            // made in (dp_calib uses Camera::width/height).
            camera_.set(cv::CAP_PROP_FRAME_WIDTH, Camera::width);
            camera_.set(cv::CAP_PROP_FRAME_HEIGHT, Camera::height);
            LOGI("face detector: camera %d opened (%dx%d requested)",
                 cameraIndex, Camera::width, Camera::height);
        } else {
            LOGW("face detector: cannot open camera %d, detector will idle",
                 cameraIndex);
        }
    }

    frameSizeChecked_ = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        running_ = true;
    }
    try {
        thread_ = std::thread(&FaceDetector::detectionLoop, this);
    } catch (const std::exception& e) {
        LOGE("face detector: cannot start detection thread: %s", e.what());
        std::lock_guard<std::mutex> lock(mutex_);
        running_ = false;
        camera_.release();
        yunet_.release();
        return false;
    }

    LOGI("face detector: started");
    return true;
}

void FaceDetector::release() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        running_ = false;
    }
    wakeup_.notify_all();

    if (thread_.joinable())
        thread_.join();

    camera_.release();
    yunet_.release();

    {
        std::lock_guard<std::mutex> lock(mutex_);
        latestDetections_.clear();
        latestImage_.release();
    }
}

std::vector<Detection> FaceDetector::getLatestDetection() {
    std::lock_guard<std::mutex> lock(mutex_);
    return latestDetections_;
}

cv::Mat FaceDetector::getLatestImage() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (latestImage_.empty())
        return cv::Mat();
    // The caller draws on the result, so hand out a private copy.
    return latestImage_.clone();
}

void FaceDetector::detectionLoop() {
    while (true) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (!running_)
                break;
        }

        if (!camera_.isOpened()) {
            // No camera: idle until release() asks us to stop.
            std::unique_lock<std::mutex> lock(mutex_);
            wakeup_.wait(lock, [this] { return !running_; });
            continue;
        }

        cv::Mat frame;
        camera_ >> frame;
        if (frame.empty()) {
            std::unique_lock<std::mutex> lock(mutex_);
            wakeup_.wait_for(lock, std::chrono::milliseconds(5));
            continue;
        }

        if (!frameSizeChecked_) {
            frameSizeChecked_ = true;
            if (frame.cols != Camera::width || frame.rows != Camera::height)
                LOGW("face detector: camera frame %dx%d != calibration size "
                     "%dx%d, detections may be misaligned", frame.cols,
                     frame.rows, Camera::width, Camera::height);
            else
                LOGI("face detector: camera frame %dx%d", frame.cols,
                     frame.rows);
        }

        const std::vector<Detection> detections = yunet_.detect(frame);
        // Camera backends may reuse their frame buffer, so keep a private copy.
        const cv::Mat snapshot = frame.clone();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            latestDetections_ = detections;
            latestImage_ = snapshot;
        }
    }
}