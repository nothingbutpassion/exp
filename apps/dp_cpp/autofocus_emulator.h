// autofocus_emulator.h
//
// AfEmulator: C++ port of the dual-projector autofocus emulation from
// autofocus.py (MainWindow::emulate_autofocus). The left/right projector
// patterns are shown on the secondary monitors through DisplayHelper instead
// of Qt widgets; the camera preview is a regular OpenCV window on the primary
// monitor.
//
// Motor travel, the pre-generated per-position blur images and the sweep
// strategy all mirror the Python version:
//   1. For each projector, build one pattern image per motor position:
//      sharp at the true best position and progressively blurrier towards
//      both travel limits (generatePyramidBlurImages).
//   2. Coarse sweep (step 4) over [min, max], then a fine sweep (step 1)
//      around the coarse result, scoring every position with
//      AfUtils::laplacianSharpness.
//   3. A sweep stops early once two consecutive positions fail to improve.
//   4. Park the motor on the best position and hold that frame for 3 s.
//
// Capture reads camera 1 by default, like the Python version. If the camera
// cannot be opened, the emulator falls back to reading the active projector's
// image directly (the blur is already baked into the generated images). Pass
// a negative camera index to force that emulated capture.
//
// Usage:
// AfEmulator emulator(options);
// if (!emulator.init(pattern, cv::Rect()))
//     return 1;
// emulator.run();

#ifndef __AUTOFOCUS_EMULATOR_H__
#define __AUTOFOCUS_EMULATOR_H__

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <string>
#include <vector>

#include "display_helper.h"

struct Options {
    std::string patternPath;   // empty: use the generated grid pattern
    int cameraIndex = 1;       // < 0: emulate the camera
    bool preview = true;
};

class AfEmulator {
public:
    explicit AfEmulator(const Options& options) : options_(options) {}

    // roi selects the region of the captured frame used for scoring; an empty
// rect means the whole frame.
    bool init(const cv::Mat& pattern, const cv::Rect& roi);
    void run();

private:
    struct SearchResult {
        int pos;
        double score;
    };

    void adjustProjectorFocus(int projectorId);
    SearchResult searchOptimalPos(int projectorId, int begin, int end, int step);
    void moveMotorTo(int projectorId, int motorPos);
    int motorPosition(int projectorId) const;
    std::vector<cv::Mat> buildMotorImages(const cv::Mat& pattern, int bestPos) const;

    void projectImages(int projectorId);
    void getProjectionImages(int projectorId, cv::Mat& left, cv::Mat& right) const;
    cv::Mat grabLatestFrame(int projectorId);
    cv::Mat preprocessFrame(int projectorId, const cv::Mat& image) const;
    void showPreview(cv::Mat& frame, double score);
    void waitUi(double seconds);

    Options options_;
    DisplayHelper display_;

    // One pattern image per motor position, indexed by position.
    std::vector<cv::Mat> leftImages_;
    std::vector<cv::Mat> rightImages_;
    cv::Mat blackImage_;
    cv::Rect roi_;   // empty (0x0): the whole frame is used

    // Both motors start at the travel minimum (kMinMotorPos).
    int leftMotorPos_ = 0;
    int rightMotorPos_ = 0;
    cv::VideoCapture camera_;
};

#endif // __AUTOFOCUS_EMULATOR_H__