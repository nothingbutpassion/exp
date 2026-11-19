#include <vector>
#include <opencv2/core.hpp>
#include "dual_projection.h"
#include "config.h"
#include "log.h"

using namespace std;
using namespace cv;

namespace dp {

void rectify_screen_region(vector<Point2f> src[2], vector<Point2f> dst[2]) {
    auto p1 = src[0][0], p2 = src[0][1], p3 = src[0][2], p4 = src[0][3];
    auto p5 = src[1][0], p6 = src[1][1], p7 = src[1][2], p8 = src[1][3];

    // adjust top points
    auto t = max(max(p1.y, p2.y), max(p5.y, p6.y));
    p1.y = t; p2.y = t; p5.y = t; p6.y = t;

    // adjust bottom points
    auto b = min(min(p3.y, p4.y), min(p7.y, p8.y));
    p3.y = b; p4.y = b; p7.y = b; p8.y = b;

    // adjust left projector region
    p1.x = max(p1.x, p4.x); 
    p4.x = p1.x;
    p2.x = min(p2.x, p3.x); 
    p3.x = p2.x;

    // adjust right projector region
    p5.x = max(p5.x, p8.x); 
    p8.x = p5.x;
    p6.x = min(p6.x, p7.x);
    p7.x = p6.x;

    dst[0] = vector<Point2f> {p1, p2, p3, p4};
    dst[1] = vector<Point2f> {p5, p6, p7, p8};
}


void compute_display_region(vector<Point2f> src[2], vector<Point2f> dst[2]) {
    constexpr float overlap_threshold = 32;
    auto p1 = src[0][0], p2 = src[0][1], p3 = src[0][2], p4 = src[0][3];
    auto p5 = src[1][0], p6 = src[1][1], p7 = src[1][2], p8 = src[1][3];
    auto h = p4.y - p1.y;
    auto s = Projector::height/h;
    auto overlap_w = s*(p2.x - p5.x);
    if (overlap_w < 1.f) {
        LOGW("overlap is %.2f, no overlap", overlap_w);
        dst[0] = vector<Point2f> {p1, p2, p3, p4};
        dst[1] = vector<Point2f> {p5, p6, p7, p8};
        return;
    }
    if (overlap_w <  overlap_threshold) {
        LOGW("overlap is %.2f, too small", overlap_w);
    }
    auto min_overlap = min(overlap_threshold, overlap_w);
    auto w1 = s*(p2.x - p1.x);
    auto w2 = s*(p6.x - p5.x);
    if (w1 < w2) {
        // left region is smaller
        if (s*(p5.x - p1.x) + min_overlap < Projector::width) {
            // extend to left edge
            p2.x = min(p1.x + Projector::width/s, p2.x);
            p3.x = p2.x;
        } else {
            p2.x = min(p5.x + min_overlap/s, p2.x);
            p3.x = p2.x;
            p1.x = max(p2.x - Projector::width/s, p1.x);
            p4.x = p1.x;
        }
        p6.x = min(p5.x + Projector::width/s, p6.x);
        p7.x = p6.x;
    } else {
        // right region is smaller
        if (s*(p6.x - p2.x) + min_overlap < Projector::width) {
            // extends to right edge
            p5.x = max(p7.x - Projector::width/s, p5.x);
            p8.x = p5.x;
        } else {
            p5.x = max(p2.x - min_overlap/s, p5.x);
            p8.x = p5.x;
            p6.x = min(p5.x + Projector::width/s, p6.x);
            p7.x = p6.x;
        }
        p1.x = max(p2.x - Projector::width/s, p1.x);
        p4.x = p1.x;
    }
    dst[0] = vector<Point2f> {p1, p2, p3, p4};
    dst[1] = vector<Point2f> {p5, p6, p7, p8};
}

int compute_buffer_overlap(vector<Point2f> rois[2]) {
    auto p2 = rois[0][1], p3 = rois[0][2], p5 = rois[1][0];
    auto h = p3.y - p2.y;
    auto w = p2.x - p5.x;
    auto s = Projector::height/h;
    return min(max(0, int(s*w)), Projector::width);
}

void screen_buffer_homographies(vector<Point2f> rois[2], Mat H_s2b[2]) {
    auto p2 = rois[0][1], p3 = rois[0][2], p5 = rois[1][0];
    auto h = p3.y - p2.y;
    auto s = Projector::height/h;
    Point2f p1 = {p2.x - Projector::width/s, p2.y};
    double H1[] = {
        s, 0, -s*p1.x,
        0, s, -s*p1.y,
        0, 0,  1
    };
    double H2[] = {
        s, 0, -s*p5.x,
        0, s, -s*p5.y,
        0, 0,  1
    };
    Mat(3, 3, CV_64FC1, H1).copyTo(H_s2b[0]);
    Mat(3, 3, CV_64FC1, H2).copyTo(H_s2b[1]);
}


} // namespace dp
