#ifndef __VIDEO_RENDER__
#define __VIDEO_RENDER__

#include <vector>
#include <opencv2/core.hpp>
#include "gl_shader.h"

struct CalibParameters;
struct GLFWwindow;
class VideoRender {
public:
    ~VideoRender() { release(); }
    bool init();
    void release();
    void show();
    void hide();
    void setParameters(const CalibParameters& params);
    void setFrame(const cv::Mat& image);
    void drawFrame();

private:
    bool initGL();
    void releaseGL();
public:
    void pollEvents(double timeout);
private:
    std::vector<GLFWwindow*> windows_;
    std::vector<unsigned> vaos_;
    unsigned vbo_ = 0;
    unsigned texture_ = 0;
    GLShader shader_;
    
    int overlap_ = 0;
    cv::Mat H_p2b_[2];
    cv::Size video_size_;
};



#endif // __VIDEO_RENDER__