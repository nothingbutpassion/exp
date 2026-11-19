#include <algorithm>
#include <cmath>
#include <cstdio>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "autofocus_utils.h"

namespace {

// numpy.linspace(start, stop, num), used by generatePyramidBlurImages.
std::vector<double> linspace(double start, double stop, int num) {
    std::vector<double> values(static_cast<size_t>(std::max(num, 0)));
    if (num <= 0)
        return values;
    if (num == 1) {
        values[0] = start;
        return values;
    }
    const double step = (stop - start) / (num - 1);
    for (int i = 0; i < num; ++i)
        values[static_cast<size_t>(i)] = start + step * i;
    return values;
}

// numpy's Mat.var() over the whole array: all rows, columns and channels
// combined. cv::meanStdDev reports per-channel values, so this is done by
// hand. Expects a CV_64F Mat (the Laplacian output).
double globalVariance(const cv::Mat& m) {
    const cv::Mat source = m.isContinuous() ? m : m.clone();
    const cv::Mat flat = source.reshape(1, 1);   // rows*cols*channels x 1
    const size_t count = flat.total();
    if (count == 0)
        return 0.0;

    const double* p = flat.ptr<double>(0);
    double sum = 0.0;
    for (size_t i = 0; i < count; ++i)
        sum += p[i];
    const double mean = sum / static_cast<double>(count);

    double squareSum = 0.0;
    for (size_t i = 0; i < count; ++i) {
        const double diff = p[i] - mean;
        squareSum += diff * diff;
    }
    return squareSum / static_cast<double>(count);
}

} // namespace

std::vector<cv::Mat> AfUtils::generateBoxBlurImages(
    const cv::Mat& original, int numImages, cv::Size ksize) {
    std::vector<cv::Mat> blurred;
    if (numImages <= 0 || original.empty())
        return blurred;

    blurred.reserve(static_cast<size_t>(numImages));
    cv::Mat image = original;
    for (int i = 0; i < numImages; ++i) {
        cv::Mat next;
        // ddepth -1 (same as source), normalize = true: cv2.boxFilter defaults.
        cv::boxFilter(image, next, -1, ksize);
        blurred.push_back(next);
        image = next;
    }
    return blurred;
}

std::vector<cv::Mat> AfUtils::generatePyramidBlurImages(
    const cv::Mat& original, int numImages) {
    std::vector<cv::Mat> blurred;
    if (numImages <= 0 || original.empty())
        return blurred;

    const int w = original.cols;
    const int h = original.rows;
    const std::vector<double> downScales = linspace(0.5, 0.025, numImages);

    // Cumulative down-scaling, same as the Python loop: each step resizes the
    // previous result using a scale derived from the original size. The
    // Python version builds its list back-to-front and reverses the blurred
    // result at the end, which yields exactly this order.
    //
    // NOTE: Python rounds with round() (half-to-even); std::lround rounds half
    // away from zero. The two agree except on exact .5 pixel sizes.
    cv::Mat image = original;
    blurred.reserve(static_cast<size_t>(numImages));
    for (double s : downScales) {
        const int dw = std::max(1, static_cast<int>(std::lround(s * w)));
        const int dh = std::max(1, static_cast<int>(std::lround(s * h)));
        cv::Mat resized;
        cv::resize(image, resized, cv::Size(dw, dh));
        image = resized;

        cv::Mat gaussian;
        cv::GaussianBlur(image, gaussian, cv::Size(5, 5), 1.0, 1.0);
        cv::Mat restored;
        cv::resize(gaussian, restored, cv::Size(w, h));
        blurred.push_back(restored);
    }
    return blurred;
}

std::vector<cv::Mat> AfUtils::generateDefocusBlurImages(
    const cv::Mat& original, int numImages, int maxRadius) {
    std::vector<cv::Mat> blurredImages;
    if (numImages <= 0 || original.empty())
        return blurredImages;

    blurredImages.reserve(static_cast<size_t>(numImages));
    for (int i = 1; i <= numImages; ++i) {
        // Radius increases with blur level; integer truncation matches the
        // Python int(i * max_radius / num_images).
        int radius = i * maxRadius / numImages;
        radius = std::max(1, radius);

        // Circular (disk) kernel.
        const int kernelSize = 2 * radius + 1;
        cv::Mat kernel = cv::Mat::zeros(kernelSize, kernelSize, CV_32F);
        double kernelSum = 0.0;
        for (int y = 0; y < kernelSize; ++y) {
            float* row = kernel.ptr<float>(y);
            for (int x = 0; x < kernelSize; ++x) {
                const double dx = x - radius;
                const double dy = y - radius;
                if (dx * dx + dy * dy <= static_cast<double>(radius) * radius) {
                    row[x] = 1.0f;
                    kernelSum += 1.0;
                }
            }
        }
        kernel = kernel / kernelSum;   // same as kernel /= kernel.sum()
        cv::Mat blurred;
        cv::filter2D(original, blurred, -1, kernel);
        blurredImages.push_back(blurred);
    }
    return blurredImages;
}

std::vector<cv::Mat> AfUtils::generateGaussianBlurImages(
    const cv::Mat& original, int numLevels, int maxKernelSize) {
    std::vector<cv::Mat> blurredImages;
    if (numLevels <= 0 || original.empty())
        return blurredImages;

    blurredImages.reserve(static_cast<size_t>(numLevels));
    for (int i = 1; i <= numLevels; ++i) {
        // maxKernelSize // 4 is integer division in Python; keep it that way.
        const double sigma = (static_cast<double>(i) / numLevels) * (maxKernelSize / 4);
        int kernelSize = 2 * static_cast<int>(2 * sigma) + 1;   // int() truncates
        kernelSize = std::max(3, std::min(kernelSize, maxKernelSize));
        cv::Mat blurred;
        // cv::GaussianBlur uses sigmaX for sigmaY when sigmaY == 0, matching
        // the Python call that passes only sigmaX.
        cv::GaussianBlur(original, blurred, cv::Size(kernelSize, kernelSize), sigma);
        blurredImages.push_back(blurred);
    }
    return blurredImages;
}

cv::Mat AfUtils::gridPattern(int w, int h, int dx, int dy, int d,
                             cv::Scalar color1, cv::Scalar color2) {
    if (w <= 0 || h <= 0 || dx <= 0 || dy <= 0)
        return cv::Mat();

    cv::Mat img(h, w, CV_8UC3, color1);
    const int nx = w / dx;
    const int ny = h / dy;
    const int ox = (w - nx * dx) / 2;
    const int oy = (h - ny * dy) / 2;

    for (int i = 0; i <= nx; ++i) {
        const int s = std::max(ox + i * dx, 0);
        const int e = std::min(ox + i * dx + d, w);
        if (s < e)
            img.colRange(s, e).setTo(color2);
    }
    for (int i = 0; i <= ny; ++i) {
        // NOTE: the Python original uses dx on this axis too (not dy); kept
        // as-is so both implementations produce the same pattern.
        const int s = std::max(oy + i * dx, 0);
        const int e = std::min(oy + i * dx + d, h);
        if (s < e)
            img.rowRange(s, e).setTo(color2);
    }
    return img;
}

cv::Mat AfUtils::chessboardPattern(int w, int h, int cellW, int cellH,
                                   cv::Scalar color1, cv::Scalar color2) {
    if (w <= 0 || h <= 0 || cellW <= 0 || cellH <= 0)
        return cv::Mat();

    cv::Mat img = cv::Mat::zeros(h, w, CV_8UC3);   // Python uses np.zeros
    const int nx = w / cellW;
    const int ny = h / cellH;
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            const cv::Scalar& color = ((i + j) % 2 == 0) ? color1 : color2;
            img(cv::Rect(j * cellW, i * cellH, cellW, cellH)).setTo(color);
        }
    }
    return img;
}

cv::Mat AfUtils::imagePattern(const std::string& imagePath, int w, int h) {
    cv::Mat img = cv::imread(imagePath, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::fprintf(stderr, "can't load %s\n", imagePath.c_str());
        return img;
    }
    cv::Mat resized;
    cv::resize(img, resized, cv::Size(w, h), 0.0, 0.0, cv::INTER_CUBIC);
    return resized;
}

double AfUtils::laplacianSharpness(const cv::Mat& image) {
    if (image.empty())
        return 0.0;
    // CV_64F prevents data overflow when calculating differences.
    cv::Mat laplacian;
    cv::Laplacian(image, laplacian, CV_64F);
    return std::sqrt(globalVariance(laplacian));
}

double AfUtils::tenengradSharpness(const cv::Mat& image) {
    if (image.empty())
        return 0.0;

    cv::Mat gray;
    if (image.channels() > 1)
        cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);
    else
        gray = image;

    cv::Mat gX, gY;
    cv::Sobel(gray, gX, CV_64F, 1, 0);
    cv::Sobel(gray, gY, CV_64F, 0, 1);
    cv::Mat magnitude;   // sqrt(gX^2 + gY^2)
    cv::magnitude(gX, gY, magnitude);
    return cv::mean(magnitude)[0];
}