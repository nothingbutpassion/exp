#include <opencv2/videoio.hpp>
#ifdef VIDEO_RENDER_EX
#include "video_render_ex.h"
#else
#include "video_render.h"
#endif
#include "calib_parameters.h"
#include "log.h"

using namespace std;
using namespace cv;

int main(int argc, char* argv[]) {
    if (argc < 3) {
        LOGE("Usage %s <param-path> <video_path> [fps]", argv[0]);
        return -1;
    }

    const char* param_path = argv[1];
    const char* video_path = argv[2];
    int fps = argc > 3 ? stoi(argv[3]) : 30;
    if (fps < 1)
        fps = 1;
    const double inv_fps = 1.0/fps;

    CalibParameters params;
    if (!params.load(param_path)) {
        LOGE("load calib params from %s failed", param_path);
        return -1;
    }

    VideoRender render;
    render.init();
    render.setParameters(params);
    VideoCapture cap(video_path); 
    Mat frame;
    int64 n_frames = 0;
    double duration = 0;
    while (cap.isOpened()) {
        auto start = getTickCount();
        if (!cap.read(frame)) {
            LOGE("read frame failed");
            break;
        }
        render.setFrame(frame);
        render.drawFrame();
        auto seconds = double(getTickCount() - start)/getTickFrequency();
        if (inv_fps > seconds)
            render.pollEvents(inv_fps - seconds);
        duration += seconds;
        n_frames += 1;
        if (duration > 1) {
            LOGD("fps: %.2f", double(n_frames)/duration);
            duration = 0;
            n_frames = 0;
        }
    }
}