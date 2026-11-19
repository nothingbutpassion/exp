#ifndef __DISPLAY_HELPER__
#define __DISPLAY_HELPER__

#include <vector>
#include <opencv2/core.hpp>
#include "gl_shader.h"


struct GLFWwindow;
class DisplayHelper {
public:
    ~DisplayHelper() { release(); }
    bool init();
    void release();
    void show();
    void hide();
    void setPattern(const cv::Mat& image);
    void showPattern(int windowIndex, double timeout);
    void setImages(const cv::Mat& left, const cv::Mat& right);
    void showImages(double timeout);

private:
    bool initGL();
    void releaseGL();
    void drawPattern(int windowIndex);
    void drawImages();
    void pollEvents(double timeout);
    
private:
    std::vector<GLFWwindow*> windows_;
    std::vector<unsigned> vaos_;
    unsigned vbo_ = 0;
    unsigned texture_ = 0;
    unsigned image_textures_[2] = {0, 0};
    GLShader shader_;

};



#endif // __DISPLAY_HELPER__