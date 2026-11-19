// yunet.cpp
//
// YuNet face detector implementation. cv::FaceDetectorYN returns one 15-float
// row per face: [x, y, w, h, lm0x, lm0y, ..., lm4x, lm4y, score].

#include <exception>

#include <opencv2/imgproc.hpp>

#include "log.h"
#include "yunet.h"

bool YuNet::init(const std::string& modelPath, const cv::Size& inputSize,
                 float scoreThreshold, float nmsThreshold, int topK) {
    release();

    if (modelPath.empty() || inputSize.width <= 0 || inputSize.height <= 0) {
        LOGE("invalid YuNet setup: model='%s', input size=%dx%d",
             modelPath.c_str(), inputSize.width, inputSize.height);
        return false;
    }

    try {
        // auto backend_id = cv::dnn::DNN_BACKEND_CUDA;
        // auto target_id = cv::dnn::DNN_TARGET_CUDA;
        detector_ = cv::FaceDetectorYN::create(modelPath, "", inputSize,
                                               scoreThreshold, nmsThreshold,
                                               topK);
    } catch (const cv::Exception& e) {
        LOGE("cannot load YuNet model %s: %s", modelPath.c_str(), e.what());
        return false;
    } catch (const std::exception& e) {
        LOGE("cannot load YuNet model %s: %s", modelPath.c_str(), e.what());
        return false;
    }

    if (detector_.empty()) {
        LOGE("cannot load YuNet model %s", modelPath.c_str());
        return false;
    }

    inputSize_ = inputSize;
    LOGI("YuNet loaded: %s (%dx%d, score>=%.2f, nms=%.2f)",
         modelPath.c_str(), inputSize.width, inputSize.height, scoreThreshold,
         nmsThreshold);
    return true;
}

std::vector<Detection> YuNet::detect(const cv::Mat& image) {
    std::vector<Detection> detections;
    if (detector_.empty() || image.empty())
        return detections;

    // Feed the network its configured size; remember how to map results back.
    const bool resized = image.size() != inputSize_;
    cv::Mat input = image;
    if (resized)
        cv::resize(image, input, inputSize_);

    cv::Mat faces;   // N x 15, CV_32F
    detector_->detect(input, faces);
    if (faces.empty())
        return detections;

    const float sx = resized
                         ? static_cast<float>(image.cols) / inputSize_.width
                         : 1.0f;
    const float sy = resized
                         ? static_cast<float>(image.rows) / inputSize_.height
                         : 1.0f;

    detections.reserve(static_cast<size_t>(faces.rows));
    for (int i = 0; i < faces.rows; ++i) {
        const float* row = faces.ptr<float>(i);

        Detection detection;
        detection.box = cv::Rect2f(row[0] * sx, row[1] * sy, row[2] * sx,
                                   row[3] * sy);
        for (int k = 0; k < 5; ++k)
            detection.landmarks[static_cast<size_t>(k)] =
                cv::Point2f(row[4 + 2 * k] * sx, row[5 + 2 * k] * sy);
        detection.score = row[14];
        detections.push_back(detection);
    }
    return detections;
}

void YuNet::release() {
    detector_.reset();
    inputSize_ = cv::Size();
}