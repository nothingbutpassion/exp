#ifndef __CHESSBOARD_CALIBRATOR_H__
#define __CHESSBOARD_CALIBRATOR_H__

#include <vector>
#include <opencv2/core.hpp>
#include "calib_parameters.h"
#include "dual_projection.h"

class ChessboardCalibrator {
public:
    ChessboardCalibrator();
    ~ChessboardCalibrator();

    // Called before calibrate()
    const cv::Mat& get_pattern_image();

    // Calibrate with left & right projection image
    bool calibrate(const cv::Mat& left_image, const cv::Mat& right_image, const cv::Rect& roi, CalibParameters& out_params);

private:
    cv::Mat generate_chessboard_image(const cv::Size& image_size, const cv::Size& pattern_size);
    std::vector<cv::Point2f> get_pattern_corners(const cv::Size& image_size, const cv::Size& pattern_size);
    std::vector<cv::Point2f> find_chessboard_corners(const cv::Mat& image, const cv::Size& pattern_size);

private:
    cv::Mat pattern_image_;
    std::vector<cv::Point2f> pattern_corners_;

    cv::Mat H_p2s_[2];  // Homography from projector coords to screen (camera image) coords
    cv::Mat H_s2b_[2];  // Homography from screen coords to video buffer coords
    cv::Mat H_p2b_[2];  // Homography from projector coords to video buffer coords
                        // H_p2b_[i] =  H_p2s_[i] * H_s2b_[i]  

    std::vector<cv::Point2f> screen_quads_[2];  // screen regions
    std::vector<cv::Point2f> screen_rois_[2];   // screen rois for video image
};

#endif