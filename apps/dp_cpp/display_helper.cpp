#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "display_helper.h"
#include "log.h"

using namespace std;
using namespace cv;

namespace {

// Shader sources
const char* kImageVextexShader = 
    "#version 330 core\n"
    "layout (location = 0) in vec3 aPos;\n"
    "layout (location = 1) in vec2 aTexCoord;\n"
    "out vec2 TexCoord;\n"
    "void main() {\n"
    "   gl_Position = vec4(aPos, 1.0);\n"
    "   TexCoord = aTexCoord;\n"
    "}\0";

const char* kImageFragmentShader = 
    "#version 330 core\n"
    "out vec4 FragColor;\n"
    "in vec2 TexCoord;\n"
    "uniform sampler2D uTexture;\n"
    "void main() {\n"
    "   FragColor = texture(uTexture, TexCoord).bgra;\n"
    "}\0";


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


bool DisplayHelper::init() {
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

    dumpMonitors();
    int count = 0;
    GLFWmonitor** monitors = glfwGetMonitors(&count);
    vector<int> slave_coords;
    vector<GLFWmonitor*> slave_monitors;
    for (int i=0; i < count; i++) {
        const GLFWvidmode* mode = glfwGetVideoMode(monitors[i]);
        int xpos = -1, ypos = -1;
        glfwGetMonitorPos(monitors[i], &xpos, &ypos);
        if (xpos == 0 && ypos == 0)     // exclude main display
            continue;
        slave_monitors.push_back(monitors[i]);
        slave_coords.push_back(xpos);
    }
    if (slave_monitors.size() == 0) {
        LOGE("no extral displayer found");
        return false;
    }
    auto index = (std::min_element(slave_coords.begin(), slave_coords.end()) - slave_coords.begin());
    if (index != 0)
        swap(slave_monitors[index], slave_monitors[0]);
    for (int i = 0; i < slave_monitors.size(); i++) {
        const GLFWvidmode* mode = glfwGetVideoMode(slave_monitors[i]);
        // first window's GL context can be shared
        GLFWwindow* share = windows_.empty() ? nullptr : windows_[0];
        GLFWwindow* win = glfwCreateWindow(
            mode->width, mode->height,      // full native resolution of this monitor
            "fullscreen",
            slave_monitors[i],              // monitor = fullscreen on that display
            share);
        if (!win) {
            LOGE("glfwCreateWindow failed: monitor=%d", i);
            continue;
        }
        windows_.push_back(win);  
    }
    if (windows_.empty()) {
        LOGE("no GLFW window created");
        return false;
    }
    // Make the first window's context current to initialize GLAD
    glfwMakeContextCurrent(windows_[0]);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        LOGE("gladLoadGLLoader failed");
        return false;
    }
    if (!initGL())
        return false;
    LOGI("display helper is initialized");
    return true;
}

void DisplayHelper::release() {
    releaseGL();
    for (auto window : windows_)
        glfwDestroyWindow(window);
    windows_.clear();
    
    // This function has no effect if GLFW is not initialized.
    glfwTerminate();
    LOGI("display helper is released");
}


void DisplayHelper::show() {
    for (auto win: windows_) {
        glfwShowWindow(win);
        glfwRestoreWindow(win);
    }
}


void DisplayHelper::hide() {
    for (auto win: windows_) {
        glfwHideWindow(win);
    }
}

bool DisplayHelper::initGL() {
    // Pattern texture
    glGenTextures(1, &texture_);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, texture_);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

    // left & rihgt frame textures
    for (int i=0; i < 2; i++) {
        glGenTextures(1, &image_textures_[i]);
        glActiveTexture(i == 0 ? GL_TEXTURE1: GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, image_textures_[i]);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); 
    }

    // Shared VBO
    // Perimeter order (TL, BL, BR, TR): consistent CCW winding for both
    // GL_TRIANGLE_STRIP and GL_TRIANGLE_FAN.
    float vertices[] = {
        // positions         // texture coords
        -1.0f,  1.0f, 0.0f,  0.0f, 0.0f,
        -1.0f, -1.0f, 0.0f,  0.0f, 1.0f,
         1.0f, -1.0f, 0.0f,  1.0f, 1.0f,
         1.0f,  1.0f, 0.0f,  1.0f, 0.0f
    };
    glGenBuffers(1, &vbo_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // VAOs is not shared for each window
    for (size_t i = 0; i < windows_.size(); ++i) {
        glfwMakeContextCurrent(windows_[i]);
        unsigned vao = 0;
        glGenVertexArrays(1, &vao);
        glBindVertexArray(vao);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_);
        glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)0);
        glEnableVertexAttribArray(0);
        glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5*sizeof(float), (void*)(3*sizeof(float)));
        glEnableVertexAttribArray(1);
        glBindVertexArray(0);
        vaos_.push_back(vao);
    }
    return shader_.build(kImageVextexShader, kImageFragmentShader);
}

void DisplayHelper::releaseGL() {
    if (texture_) {
        glDeleteTextures(1, &texture_);
        texture_ = 0;
    }

    for (int i=0; i < 2; i++) {
        if (image_textures_[i]) {
            glDeleteTextures(1, &image_textures_[i]);
            image_textures_[i] = 0;
        } 
    }

    if (vbo_) {
        glDeleteBuffers(1, &vbo_);
        vbo_ = 0;
    }
    for (auto vao: vaos_)
        glDeleteVertexArrays(1, &vao);
    vaos_.clear();
    shader_.release();

}

void DisplayHelper::setPattern(const cv::Mat& image) {
    if (windows_.size() > 0) {
        glfwMakeContextCurrent(windows_[0]);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, texture_);
        GLenum format = (image.channels() == 4) ? GL_RGBA : (image.channels() == 3) ? GL_RGB : GL_ALPHA;
        glTexImage2D(GL_TEXTURE_2D, 0, format, image.cols, image.rows, 0, format, GL_UNSIGNED_BYTE, image.data);
        checkGLError();
    }
}

void DisplayHelper::drawPattern(int windowIndex) {
    for (int i = 0; i < windows_.size(); ++i) {
        glfwMakeContextCurrent(windows_[i]);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        if (i == windowIndex) {
            glActiveTexture(GL_TEXTURE0);
            glBindTexture(GL_TEXTURE_2D, texture_);
            shader_.use();
            glUniform1i(shader_.location("uTexture"), 0);
            glBindVertexArray(vaos_[i]);
            glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
        }
        glfwSwapBuffers(windows_[i]);
    }
}

void DisplayHelper::showPattern(int windowIndex, double timeout) {
    drawPattern(windowIndex);
    pollEvents(timeout);
}

void DisplayHelper::setImages(const cv::Mat& left, const cv::Mat& right) {
    for (int i = 0; i < windows_.size(); ++i) {
        glfwMakeContextCurrent(windows_[i]);
        glActiveTexture(i == 0 ? GL_TEXTURE1 : GL_TEXTURE2);
        const Mat& image = (i == 0 ? left : right);
        glBindTexture(GL_TEXTURE_2D, image_textures_[i]);
        GLenum format = (image.channels() == 4) ? GL_RGBA : (image.channels() == 3) ? GL_RGB : GL_ALPHA;
        glTexImage2D(GL_TEXTURE_2D, 0, format, image.cols, image.rows, 0, format, GL_UNSIGNED_BYTE, image.data);
        checkGLError();
    }
}

void DisplayHelper::drawImages() {
    for (int i = 0; i < windows_.size(); ++i) {
        glfwMakeContextCurrent(windows_[i]);
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glActiveTexture(i == 0 ? GL_TEXTURE1 : GL_TEXTURE2);
        glBindTexture(GL_TEXTURE_2D, image_textures_[i]);
        shader_.use();
        glUniform1i(shader_.location("uTexture"), (i == 0 ? 1 : 2));
        glBindVertexArray(vaos_[i]);
        glDrawArrays(GL_TRIANGLE_FAN, 0, 4);
        glfwSwapBuffers(windows_[i]);
    }
}

void DisplayHelper::showImages(double timeout) {
    drawImages();
    pollEvents(timeout);
}

void DisplayHelper::pollEvents(double timeout) {
    auto start = getTickCount();
    while (double(getTickCount() - start)/getTickFrequency() < timeout) {
        bool should_exit = false;
        for (auto window: windows_) {
            if (glfwWindowShouldClose(window) || glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
                should_exit = true;
                break;
            }
        }
        if (should_exit)
            break;
        glfwPollEvents();
    }
}



