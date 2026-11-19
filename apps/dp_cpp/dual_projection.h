#ifndef __DUAL_PROJECTION_H__
#define __DUAL_PROJECTION_H__

#include <vector>
#include <opencv2/core.hpp>

namespace dp {
    
void rectify_screen_region(std::vector<cv::Point2f> quads[2], std::vector<cv::Point2f> rects[2]);
void compute_display_region(std::vector<cv::Point2f> rects[2], std::vector<cv::Point2f> rois[2]);
int  compute_buffer_overlap(std::vector<cv::Point2f> rois[2]);
void screen_buffer_homographies(std::vector<cv::Point2f> rois[2], cv::Mat H_s2b[2]);

};



#endif // __DUAL_PROJECTION_H__