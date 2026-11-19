// yunet.h
//
// YuNet face detector: thin wrapper around cv::FaceDetectorYN that turns its
// raw Nx15 result matrix into Detection records.
//
// Usage:
// YuNet yunet;
// if (!yunet.init("models/face_detection_yunet_2023mar.onnx", cv::Size(1920, 1080)))
//     return false;
// const std::vector<Detection> faces = yunet.detect(frame);
// ...
// yunet.release();

#ifndef __YUNET_H__
#define __YUNET_H__

#include <opencv2/core.hpp>
#include <opencv2/objdetect.hpp>

#include <array>
#include <string>
#include <vector>

struct Detection {
    cv::Rect2f box;                          // x, y, w, h in image pixels
    float score = 0.0f;                      // detection confidence
    std::array<cv::Point2f, 5> landmarks{};  // right eye, left eye, nose tip,
                                             // right mouth corner, left mouth corner
};

class YuNet {
public:
    // Loads the ONNX model and fixes the network input size. Returns false on
    // error (missing/unreadable model, invalid size).
    bool init(const std::string& modelPath, const cv::Size& inputSize,
              float scoreThreshold = 0.9f, float nmsThreshold = 0.3f,
              int topK = 5000);

    // Runs inference on image and returns the detected faces. Coordinates are
    // always in image pixels: the frame is resized to the configured input
    // size for the network and the boxes are mapped back. Returns an empty
    // vector when the detector is not initialised or the image is empty.
    std::vector<Detection> detect(const cv::Mat& image);

    // Drops the model and frees the detector.
    void release();

private:
    cv::Ptr<cv::FaceDetectorYN> detector_;
    cv::Size inputSize_;
};

#endif // __YUNET_H__