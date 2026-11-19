#include "calib_parameters.h"


bool CalibParameters::save(const char* path) {
    try {
        cv::FileStorage fs(path, cv::FileStorage::WRITE);
        if (!fs.isOpened())
            return false;
        fs << "type" << type;
        fs << "overlap" << overlap;
        fs << "h_p2b_0" << h_p2b[0];
        fs << "h_p2b_1" << h_p2b[1];
        fs << "roi0" << "{" << "x" << r_screen[0].x << "y" << r_screen[0].y
           << "width" << r_screen[0].width << "height" << r_screen[0].height << "}";
        fs << "roi1" << "{" << "x" << r_screen[1].x << "y" << r_screen[1].y
           << "width" << r_screen[1].width << "height" << r_screen[1].height << "}";
        fs.release();
    } catch (...) {
        return false;
    }
    return true;
}

bool CalibParameters::load(const char* path) {
    try {
        cv::FileStorage fs(path, cv::FileStorage::READ);
        if (!fs.isOpened())
            return false;
        type = (int)fs["type"];
        overlap = (int)fs["overlap"];
        fs["h_p2b_0"] >> h_p2b[0];
        fs["h_p2b_1"] >> h_p2b[1];
        cv::FileNode roi0Node = fs["roi0"];
        if (!roi0Node.empty()) {
            r_screen[0].x = (int)roi0Node["x"];
            r_screen[0].y = (int)roi0Node["y"];
            r_screen[0].width = (int)roi0Node["width"];
            r_screen[0].height = (int)roi0Node["height"];
        }
        cv::FileNode roi1Node = fs["roi1"];
        if (!roi1Node.empty()) {
            r_screen[1].x = (int)roi1Node["x"];
            r_screen[1].y = (int)roi1Node["y"];
            r_screen[1].width = (int)roi1Node["width"];
            r_screen[1].height = (int)roi1Node["height"];
        }
        fs.release();
    } catch (...) {
        return false;
    }
    return true;
}