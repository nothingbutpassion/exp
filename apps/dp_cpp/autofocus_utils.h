// autofocus_utils.h
//
// C++ counterpart of the AfUtils helper class from autofocus.py. Every static
// method mirrors its Python original one-to-one: same defaults, same OpenCV
// calls, same output order. The Python file stays the reference prototype, so
// the two implementations can be compared side by side.
//
// Conventions
//  - Images are cv::Mat in the layout OpenCV uses by default (BGR, CV_8U) and
//    the blur generators return std::vector<cv::Mat> in the same order as the
//    Python lists.
//  - Blurred-image generators return fewer entries than requested only when
//    numImages / numLevels <= 0 (Python then returns an empty list).
//  - imagePattern() returns an empty Mat when the file cannot be read, where
//    Python prints a message and returns None.
//
// Usage:
// cv::Mat pattern = AfUtils::imagePattern("autofocus_target.png", 1920, 1080);
// auto blurry = AfUtils::generatePyramidBlurImages(pattern, 25);
// double score = AfUtils::laplacianSharpness(blurry.back());

#ifndef __AUTOFOCUS_UTILS_H__
#define __AUTOFOCUS_UTILS_H__

#include <opencv2/core.hpp>

#include <string>
#include <vector>

class AfUtils {
public:
    // Box filter applied repeatedly to the previous result: entry i is the
    // original blurred (i + 1) times.
    static std::vector<cv::Mat> generateBoxBlurImages(
        const cv::Mat& original, int numImages, cv::Size ksize = cv::Size(7, 7));

    // Down-scale pyramid, each level Gaussian-blurred (5x5, sigma 1) and
    // scaled back to the original size. Index 0 is the least blurred (0.5
    // scale), the last entry the most blurred (0.025 scale).
    static std::vector<cv::Mat> generatePyramidBlurImages(
        const cv::Mat& original, int numImages);

    // Disk-kernel defocus blur; radius grows linearly from 1 to maxRadius.
    static std::vector<cv::Mat> generateDefocusBlurImages(
        const cv::Mat& original, int numImages, int maxRadius = 25);

    // Gaussian blur with sigma and kernel size growing linearly per level.
    static std::vector<cv::Mat> generateGaussianBlurImages(
        const cv::Mat& original, int numLevels = 11, int maxKernelSize = 51);

    // Line/grid test pattern. In practice sharper for autofocus than a
    // chessboard.
    static cv::Mat gridPattern(int w, int h, int dx, int dy, int d = 8,
                               cv::Scalar color1 = cv::Scalar(0, 0, 0),
                               cv::Scalar color2 = cv::Scalar(255, 255, 255));

    // Checkerboard test pattern (cell size in pixels).
    static cv::Mat chessboardPattern(int w, int h, int cellW, int cellH,
                                     cv::Scalar color1 = cv::Scalar(255, 255, 255),
                                     cv::Scalar color2 = cv::Scalar(0, 0, 0));

    // Loads an image and resizes it to (w, h) with cubic interpolation.
    // Returns an empty Mat when the file cannot be read.
    static cv::Mat imagePattern(const std::string& imagePath, int w, int h);

    // sqrt of the variance of the Laplacian. Higher = sharper.
    static double laplacianSharpness(const cv::Mat& image);

    // Mean Tenengrad (Sobel gradient magnitude). Higher = sharper.
    static double tenengradSharpness(const cv::Mat& image);
};

#endif // __AUTOFOCUS_UTILS_H__