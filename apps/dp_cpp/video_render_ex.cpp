#include <thread>
#include <chrono>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "video_render_ex.h"
#include "calib_parameters.h"
#include "config.h"
#include "log.h"

using namespace std;
using namespace cv;

namespace {
const char* kVertexShaderSrc =
R"(#version 330 core
    layout (location = 0) in vec3 aPos;
    void main() {
        gl_Position = vec4(aPos.xy, -0.1, 1.0);
    };
)";

const char* kFragmentShaderSrc =
R"(#version 330 core
    out vec4 FragColor;
    in vec2 TexCoord;
    uniform sampler2D uTexture;
    uniform bool uLeft;
    uniform float uOverlap;
    uniform vec2 uDisplaySize;
    uniform vec2 uVideoOffset;
    uniform vec2 uVideoSize;
    uniform mat3 uH;
    void main() {
        vec3 xyw1 = vec3(uLeft ? gl_FragCoord.x : gl_FragCoord.x - uDisplaySize.x , uDisplaySize.y - gl_FragCoord.y, 1.0);
        vec3 xyw2 = uH * xyw1;
        vec2 xy2 = vec2(xyw2.x/xyw2.z, xyw2.y/xyw2.z);
        if (0 < xy2.x && xy2.x < uDisplaySize.x && 0 < xy2.y && xy2.y < uDisplaySize.y) {
            float s = uLeft ? uDisplaySize.x : 0.0;
            float e = uLeft ? uDisplaySize.x - uOverlap : uOverlap;
            float t = clamp((xy2.x - s)/(e - s), 0.0, 1.0);
            float alpha = t*t*(3.0 - 2.0*t);
            vec2 xy3 = uLeft ? xy2 : vec2(xy2.x + uDisplaySize.x - uOverlap, xy2.y);
            vec2 uv = vec2((xy3.x - uVideoOffset.x)/uVideoSize.x, (xy3.y - uVideoOffset.y)/uVideoSize.y);
            FragColor = alpha*texture(uTexture, uv).bgra;
        } else {
            FragColor = vec4(0.0, 0.0, 0.0, 1.0);
        }
    };
)";

void checkGLError() {
    GLenum code;
    while ((code = glGetError()) != GL_NO_ERROR) {
        LOGE("OpenGL error: 0x%x", code);
    }
}

void dumpMonitors() {
    int count = 0;
    GLFWmonitor** monitors = glfwGetMonitors(&count);
    for (int i=0; i < count; i++) {
        const char* name = glfwGetMonitorName(monitors[i]);
        LOGI("monitor %d: name=%s", i, name);

        int xpos = -1, ypos = -1;
        glfwGetMonitorPos(monitors[i], &xpos, &ypos);
        LOGI("monitor %d: xpos=%d, ypos=%d", i, xpos, ypos);

        int width = 0, height = 0;
        glfwGetMonitorPhysicalSize(monitors[i], &width, &height);
        LOGI("monitor %d: physical_size=%dmm x %dmm", i, width, height);

        float xscale = 0, yscale = 0;
        glfwGetMonitorContentScale(monitors[i], &xscale, &yscale);
        LOGI("monitor %d: xscale=%.2f, yscale=%.2f", i, xscale, yscale);

        int x = -1, y = -1, w = 0, h = 0;
        glfwGetMonitorWorkarea(monitors[i], &x, &y, &w, &h);
        LOGI("monitor %d: work_area=(%d, %d, %d, %d)", i, x, y, w, h);

        const GLFWvidmode* mode = glfwGetVideoMode(monitors[i]);
        LOGI("monitor %d: resolution=%d x %d", i, mode->width, mode->height);
        LOGI("monitor %d: refresh=%dHz", i, mode->refreshRate);
    }
}


} // namespace


bool VideoRenderEx::init() {
    if (GLFW_TRUE != glfwInit()) {
        LOGE("glfwInit failed");
        return false;
    }
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
    glfwWindowHint(GLFW_AUTO_ICONIFY, GLFW_FALSE);
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);
    dumpMonitors();
    int count = 0;
    GLFWmonitor** monitors = glfwGetMonitors(&count);
    for (int i=0; i < count; i++) {
        const GLFWvidmode* mode = glfwGetVideoMode(monitors[i]);
        int xpos = -1, ypos = -1;
        glfwGetMonitorPos(monitors[i], &xpos, &ypos);
        if (xpos == 0 && ypos == 0)     // exclude main display
            continue;
        const GLFWvidmode* v = glfwGetVideoMode(monitors[i]);
        monitor_rects_.push_back({xpos, ypos, v->width, v->height});

    }
    if (monitor_rects_.size() != 2) {
        LOGE("%llu extral displayer found", monitor_rects_.size());
        return false;
    }
    int x_min = std::min(monitor_rects_[0].x, monitor_rects_[1].x);
    int y_min = std::min(monitor_rects_[0].y, monitor_rects_[1].y);
    int x_max = std::max(monitor_rects_[0].x + monitor_rects_[0].width,  monitor_rects_[1].x + monitor_rects_[1].width);
    int y_max = std::max(monitor_rects_[0].y + monitor_rects_[0].height, monitor_rects_[1].y + monitor_rects_[1].height);
    int win_w = x_max - x_min;
    int win_h = y_max - y_min;
    window_ = glfwCreateWindow(win_w, win_h, "borderless-window", nullptr, nullptr);
    if (!window_) {
        LOGE("glfwCreateWindow failed");
        return false;
    }
    glfwSetWindowPos(window_, x_min, y_min);
    glfwShowWindow(window_);
    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1);
    // glfwSetInputMode(window_, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // Make monitor_rects as GL viewpotrs
    if (monitor_rects_[0].x > monitor_rects_[1].x)
        swap(monitor_rects_[0], monitor_rects_[1]);
    for (auto& r: monitor_rects_) {
        LOGI("monitor rect: (%d, %d, %d, %d)",  r.x, r.y, r.width, r.height);
        r.x = r.x - x_min;
        r.y = win_h - (r.y + r.height - y_min);
        LOGI("viewport rect: (%d, %d, %d, %d)",  r.x, r.y, r.width, r.height);
    }


    // Make the first window's context current to initialize GLAD
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        LOGE("gladLoadGLLoader failed");
        return false;
    }

    if (!initGL())
        return false;
    LOGI("display helper is initialized");
    return true;
}

void VideoRenderEx::release() {
    releaseGL();
    if (window_) {
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }

    // No effect if GLFW is not initialized.
    glfwTerminate();

    overlap_ = 0;
    H_p2b_[0] = Mat();
    H_p2b_[1] = Mat();
    video_size_ = Size();
    monitor_rects_.clear();
    LOGI("display helper is released");
}


void VideoRenderEx::show() {
    glfwShowWindow(window_);
    glfwRestoreWindow(window_);
}


void VideoRenderEx::hide() {
    glfwHideWindow(window_);
}

bool VideoRenderEx::initGL() {
    // Shared shader program
    if (!shader_.build(kVertexShaderSrc, kFragmentShaderSrc))
        return false;

    // Shared texture
    glGenTextures(1, &texture_);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // Shared VBO
    // Perimeter order (TL, BL, BR, TR): consistent CCW winding for both
    // GL_TRIANGLE_STRIP and GL_TRIANGLE_FAN.
    float vertices[] = {
        -1.0f,  1.0f, 0.0f,
        -1.0f, -1.0f, 0.0f,
         1.0f, -1.0f, 0.0f,
         1.0f,  1.0f, 0.0f
    };
    glGenBuffers(1, &vbo_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // Shared VAO
    glGenVertexArrays(1, &vao_);
    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3*sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glBindVertexArray(0);
    return true;
}

void VideoRenderEx::releaseGL() {
    if (texture_) {
        glDeleteTextures(1, &texture_);
        texture_ = 0;
    }
    if (vbo_) {
        glDeleteBuffers(1, &vbo_);
        vbo_ = 0;
    }
    if (vao_) {
        glDeleteVertexArrays(1, &vao_);
        vao_ = 0;
    }
    shader_.release();
}

void VideoRenderEx::setParameters(const CalibParameters& params) {
    overlap_ = params.overlap;
    for (int i = 0; i < 2; i++) {
        if (params.h_p2b[i].empty())
            H_p2b_[i]= Mat::eye(3, 3, CV_32F);
        else
            params.h_p2b[i].convertTo(H_p2b_[i], CV_32F);
    }
}

void VideoRenderEx::setFrame(const cv::Mat& image) {
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture_);
    GLenum format = (image.channels() == 4) ? GL_RGBA : (image.channels() == 3) ? GL_RGB : GL_ALPHA;
    glTexImage2D(GL_TEXTURE_2D, 0, format, image.cols, image.rows, 0, format, GL_UNSIGNED_BYTE, image.data);
    video_size_ = image.size();
    checkGLError();
}

void VideoRenderEx::drawFrame() {
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture_);
    for (int i = 0; i < monitor_rects_.size(); ++i) {
        glViewport(monitor_rects_[i].x, monitor_rects_[i].y, monitor_rects_[i].width, monitor_rects_[i].height);
        shader_.use();
        // Set uniforms
        // uniform sampler2D uTexture;
        // uniform bool uLeft;
        // uniform float uOverlap;
        // uniform vec2 uDisplaySize;
        // uniform vec2 uVideoSize;
        // uniform vec2 uVideoOffset;
        // uniform mat3 uH;
        glUniform1i(shader_.location("uTexture"), 0);
        glUniform1i(shader_.location("uLeft"), i == 0 ? 1 : 0);
        glUniform1f(shader_.location("uOverlap"), float(overlap_));
        glUniform2f(shader_.location("uDisplaySize"), Projector::width, Projector::height);
        // compute video region
        float buf_w = float(2*Projector::width - overlap_);
        float sw = buf_w/video_size_.width;
        float sh = float(Projector::height)/video_size_.height;
        float s = min(sw, sh);
        float vw = s*video_size_.width;
        float vh = s*video_size_.height;
        float vx = 0.5f*(buf_w - vw);
        float vy = 0.5f*(Projector::height - vh);
        glUniform2f(shader_.location("uVideoSize"), vw, vh);
        glUniform2f(shader_.location("uVideoOffset"), vx, vy);
        // Set H matrix for GL shader, 
        Mat H = Mat::eye(3, 3, CV_32F);
        if (!H_p2b_[i].empty()) {
            if (H_p2b_[i].depth() != CV_32F)
                H_p2b_[i].convertTo(H, CV_32F);
            else
                H_p2b_[i].copyTo(H);
            // OpenGL expects column-major.
            transpose(H, H);
        }
        glUniformMatrix3fv(shader_.location("uH"), 1, GL_FALSE, (const GLfloat*)H.data);
        glBindVertexArray(vao_);
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
    }
    glfwSwapBuffers(window_);
}


void VideoRenderEx::pollEvents(double timeout) {
    auto start = getTickCount();
    auto duration = double(getTickCount() - start)/getTickFrequency();
    while (duration < timeout) {
        if (timeout - duration > 0.001)
            this_thread::sleep_for(chrono::microseconds(1));
        if (glfwWindowShouldClose(window_) || glfwGetKey(window_, GLFW_KEY_ESCAPE) == GLFW_PRESS)
            break;
        glfwPollEvents();
        duration = double(getTickCount() - start)/getTickFrequency();
    }
}