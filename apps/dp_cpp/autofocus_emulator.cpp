// autofocus_emulator.cpp
//
// AfEmulator implementation. See autofocus_emulator.h for an overview of the
// emulation; the Python prototype in autofocus.py remains the reference.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <thread>
#include <vector>

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include "autofocus_emulator.h"
#include "autofocus_utils.h"
#include "log.h"

namespace {

const int kMinMotorPos = 0;
const int kMaxMotorPos = 63;
const int kBestMotorPos = 36;      // true in-focus position of both motors
const int kCoarseStep = 4;
const double kSecondsPerMotorStep = 0.01;
const double kSettleSeconds = 0.1;     // wait after projecting (camera settle)
const double kFinalHoldSeconds = 3.0;  // hold the best frame for inspection
const char* kPreviewWindow = "Autofocus camera preview";

} // namespace

bool AfEmulator::init(const cv::Mat& pattern, const cv::Rect& roi) {
    roi_ = roi;
    blackImage_ = cv::Mat::zeros(pattern.size(), CV_8UC3);

    leftImages_ = buildMotorImages(pattern, kBestMotorPos);
    rightImages_ = buildMotorImages(pattern, kBestMotorPos);
    if (leftImages_.empty() || rightImages_.empty()) {
        LOGE("no motor images generated");
        return false;
    }

    if (options_.cameraIndex >= 0) {
        camera_.open(options_.cameraIndex);
        if (!camera_.isOpened())
            LOGW("cannot open camera %d, falling back to emulated capture",
                 options_.cameraIndex);
        else
            LOGI("camera %d opened for capture", options_.cameraIndex);
    } else {
        LOGI("no camera requested, using emulated capture");
    }

    if (!display_.init()) {
        LOGE("display helper init failed (are the projector monitors connected?)");
        return false;
    }
    display_.show();

    if (options_.preview) {
        cv::namedWindow(kPreviewWindow, cv::WINDOW_NORMAL);
        cv::resizeWindow(kPreviewWindow, 960, 540);
    }
    return true;
}

void AfEmulator::run() {
    for (int projectorId = 1; projectorId <= 2; ++projectorId)
        adjustProjectorFocus(projectorId);

    // Leave both projectors black, like the Python version does once the
    // emulation ends.
    display_.setImages(blackImage_, blackImage_);
    display_.showImages(kSettleSeconds);
    if (options_.preview)
        cv::destroyWindow(kPreviewWindow);
    LOGI("autofocus emulation finished");
}

std::vector<cv::Mat> AfEmulator::buildMotorImages(
    const cv::Mat& pattern, int bestPos) const {
    const std::vector<cv::Mat> below =
        AfUtils::generatePyramidBlurImages(pattern, bestPos);
    std::vector<cv::Mat> images(below.rbegin(), below.rend());
    images.push_back(pattern);
    const std::vector<cv::Mat> above =
        AfUtils::generatePyramidBlurImages(pattern, kMaxMotorPos - bestPos);
    images.insert(images.end(), above.begin(), above.end());
    return images;
}

void AfEmulator::adjustProjectorFocus(int projectorId) {
    const int optimalPos = kBestMotorPos;

    LOGI("projector %d: starting coarse sweeping...", projectorId);
    const int coarseStart = kMinMotorPos;
    const int coarseEnd = kMaxMotorPos + 1;
    SearchResult result = searchOptimalPos(projectorId, coarseStart, coarseEnd,
                                           kCoarseStep);
    LOGI("projector %d: coarse-position=%d, sharpness=%.3f, best-position=%d",
         projectorId, result.pos, result.score, optimalPos);

    LOGI("projector %d: starting fine sweep...", projectorId);
    const int fineStart = std::max(kMinMotorPos, result.pos - kCoarseStep);
    const int fineEnd = std::min(kMaxMotorPos, result.pos + kCoarseStep + 1);
    result = searchOptimalPos(projectorId, fineStart, fineEnd, 1);
    LOGI("projector %d: final-position=%d, sharpness=%.3f, best-position=%d",
         projectorId, result.pos, result.score, optimalPos);
}

AfEmulator::SearchResult AfEmulator::searchOptimalPos(
    int projectorId, int begin, int end, int step) {
    double bestScore = 0.0;
    int bestPos = 0;
    std::vector<double> historyScores;

    for (int pos = begin; pos < end; pos += step) {
        moveMotorTo(projectorId, pos);
        projectImages(projectorId);
        cv::Mat frame = grabLatestFrame(projectorId);
        const cv::Mat image = preprocessFrame(projectorId, frame);
        const double score = AfUtils::laplacianSharpness(image);
        showPreview(frame, score);
        waitUi(0.0);
        LOGI("projector %d: position=%d, sharpness=%.3f", projectorId, pos,
             score);

        historyScores.push_back(score);
        if (score > bestScore) {
            bestScore = score;
            bestPos = pos;
        }

        // If the score dropped over the last two intervals, the peak is
        // behind us: stop the sweep (same rule as the Python version).
        const size_t count = historyScores.size();
        if (count > 2) {
            const bool rose = historyScores[count - 1] > historyScores[count - 2] ||
                              historyScores[count - 2] > historyScores[count - 3];
            if (!rose)
                break;
        }
    }

    if (step == 1) {
        // Park on the winner and hold the frame for inspection.
        moveMotorTo(projectorId, bestPos);
        projectImages(projectorId);
        cv::Mat frame = grabLatestFrame(projectorId);
        const cv::Mat image = preprocessFrame(projectorId, frame);
        const double score = AfUtils::laplacianSharpness(image);
        showPreview(frame, score);
        LOGI("projector %d: parked at position=%d, sharpness=%.3f",
             projectorId, bestPos, score);
        waitUi(kFinalHoldSeconds);
    }

    return SearchResult{bestPos, bestScore};
}

void AfEmulator::moveMotorTo(int projectorId, int motorPos) {
    // Emulated stepper/servo: travel time proportional to the step count.
    const int targetPos = std::max(kMinMotorPos, std::min(motorPos, kMaxMotorPos));
    int& currentPos = (projectorId == 1) ? leftMotorPos_ : rightMotorPos_;
    const int steps = std::abs(currentPos - targetPos);
    std::this_thread::sleep_for(
        std::chrono::duration<double>(steps * kSecondsPerMotorStep));
    currentPos = targetPos;
}

int AfEmulator::motorPosition(int projectorId) const {
    return projectorId == 1 ? leftMotorPos_ : rightMotorPos_;
}

void AfEmulator::projectImages(int projectorId) {
    cv::Mat left, right;
    getProjectionImages(projectorId, left, right);
    display_.setImages(left, right);
    display_.showImages(kSettleSeconds);
}

void AfEmulator::getProjectionImages(int projectorId, cv::Mat& left,
                                            cv::Mat& right) const {
    const int pos = motorPosition(projectorId);
    if (projectorId == 1) {
        left = leftImages_[static_cast<size_t>(pos)];
        right = blackImage_;
    } else {
        left = blackImage_;
        right = rightImages_[static_cast<size_t>(pos)];
    }
}

cv::Mat AfEmulator::grabLatestFrame(int projectorId) {
    if (camera_.isOpened()) {
        // Drop buffered frames so the capture shows the current projection.
        for (int i = 0; i < 11; ++i)
            camera_.grab();
        cv::Mat frame;
        if (camera_.read(frame) && !frame.empty())
            return frame;
        LOGW("camera read failed, falling back to emulated capture");
    }

    // Emulated capture: the camera sees the active projector's pattern. The
    // clone matters: the preview draws the score onto the frame.
    cv::Mat left, right;
    getProjectionImages(projectorId, left, right);
    return (projectorId == 1 ? left : right).clone();
}

cv::Mat AfEmulator::preprocessFrame(int /*projectorId*/, const cv::Mat& image) const {
    cv::Mat gray;
    if (image.channels() == 3)
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else
        gray = image;
    cv::Mat clip = gray(roi_);
    cv::Mat blurred;
    cv::GaussianBlur(clip, blurred, cv::Size(3, 3), 0.8);
    return blurred;
}

void AfEmulator::showPreview(cv::Mat& frame, double score) {
    if (roi_.area() > 0)
        cv::rectangle(frame, roi_, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);

    char text[64];
    std::snprintf(text, sizeof(text), "Sharpness: %.2f", score);
    cv::putText(frame, text, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
                cv::Scalar(0, 0, 255));
    if (!options_.preview)
        return;
    cv::imshow(kPreviewWindow, frame);
    cv::waitKey(1);
}

void AfEmulator::waitUi(double seconds) {
    if (seconds > 0.0) {
        // Redraw the current projection and keep the projector windows
        // responsive while we wait.
        display_.showImages(seconds);
    } else if (options_.preview) {
        cv::waitKey(1);   // pump the preview window
    }
}