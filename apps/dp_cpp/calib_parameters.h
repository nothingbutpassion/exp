#ifndef __CALIBRATION_PARAMETERS_H__
#define __CALIBRATION_PARAMETERS_H__

#include <opencv2/core.hpp>
#include "config.h"

struct CalibParameters {
    int type = 0;
    int overlap = 0;
    cv::Mat h_p2b[2] = { 
        cv::Mat::eye(3, 3, CV_64F), 
        cv::Mat::eye(3, 3, CV_64F) 
    };
    cv::Rect r_screen[2] = { 
        cv::Rect(0, 0, Projector::width, Projector::height),
        cv::Rect(0, 0, Projector::width, Projector::height)
    };
    bool save(const char* path);
    bool load(const char* path);
};


#endif