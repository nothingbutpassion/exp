// face_detector.h
//
// FaceDetector runs YuNet continuously on a background thread, reading frames
// from the camera opened in init(). The thread publishes its most recent
// result, which the owner reads with getLatestDetection()/getLatestImage().
//
// Usage:
// FaceDetector detector;
// if (!detector.init("models/face_detection_yunet_2023mar.onnx", 0, cv::Size(640, 480)))
//     return false;
// ...
// const std::vector<Detection> faces = detector.getLatestDetection();
// detector.release();

#ifndef __FACE_DETECTOR_H__
#define __FACE_DETECTOR_H__

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <condition_variable>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include "yunet.h"

class FaceDetector {
public:
    FaceDetector() = default;
    ~FaceDetector();

    FaceDetector(const FaceDetector&) = delete;
    FaceDetector& operator=(const FaceDetector&) = delete;

    // Loads the YuNet model, opens the camera (negative index: no camera, the
    // thread idles) and starts the detection thread. Returns false on error.
    bool init(const std::string& modelPath, int cameraIndex,
              const cv::Size& inputSize);

    // Stops the detection thread, then releases the camera and the model.
    // Safe to call more than once and without a successful init().
    void release();

    // Returns a copy of the detections produced by the last processed frame.
    std::vector<Detection> getLatestDetection();

    // Returns a copy of the frame the latest detections were computed on.
    // Empty until the first frame has been processed.
    cv::Mat getLatestImage();

private:
    void detectionLoop();

    YuNet yunet_;
    cv::VideoCapture camera_;

    std::thread thread_;
    std::mutex mutex_;
    std::condition_variable wakeup_;
    bool running_ = false;
    bool frameSizeChecked_ = false;   // detection thread only

    // Guarded by mutex_.
    std::vector<Detection> latestDetections_;
    cv::Mat latestImage_;
};

#endif // __FACE_DETECTOR_H__