#ifndef __GL_SHADER_H__
#define __GL_SHADER_H__

#include <string>
#include <glad/glad.h>
#include "log.h"

struct GLShader {
    GLShader() = default;
    ~GLShader() { release(); }
    GLShader(const GLShader&) = delete;
    GLShader& operator=(const GLShader&) = delete;

    void use() const { glUseProgram(program_id_); }

    int location(const std::string& name) const {
        return glGetUniformLocation(program_id_, name.c_str());
    }

    void setInt(const std::string& name, int value) const {
        glUniform1i(glGetUniformLocation(program_id_, name.c_str()), value);
    }
    void setFloat(const std::string& name, float value) const {
        glUniform1f(glGetUniformLocation(program_id_, name.c_str()), value);
    }
    void setVec2(const std::string& name, float x, float y) const {
        glUniform2f(glGetUniformLocation(program_id_, name.c_str()), x, y);
    }
    void setVec3(const std::string& name, float x, float y, float z) const {
        glUniform3f(glGetUniformLocation(program_id_, name.c_str()), x, y, z);
    }
    // NOTES: mat4 given column-major, as OpenGL expects (16 floats).
    void setMat4(const std::string& name, const float* m) const {
        glUniformMatrix4fv(glGetUniformLocation(program_id_, name.c_str()), 1, GL_FALSE, m);
    }

    bool build(const std::string& vertexSrc, const std::string& fragmentSrc) {
        vertex_shader_ = glCreateShader(GL_VERTEX_SHADER);
        const char* vsSrc = vertexSrc.c_str();
        glShaderSource(vertex_shader_, 1, &vsSrc, nullptr);
        glCompileShader(vertex_shader_);
        if (!checkShader(vertex_shader_)) {
            release();
            return false;
        }
        fragment_shader_ = glCreateShader(GL_FRAGMENT_SHADER);
        const char* fsSrc = fragmentSrc.c_str();
        glShaderSource(fragment_shader_, 1, &fsSrc, nullptr);
        glCompileShader(fragment_shader_);
        if (!checkShader(fragment_shader_)) {
            release();
            return false;
        }
        program_id_ = glCreateProgram();
        glAttachShader(program_id_, vertex_shader_);
        glAttachShader(program_id_, fragment_shader_);
        glLinkProgram(program_id_);
        if (!checkProgram(program_id_)) {
            release();
            return false;
        }
        return true;
    }
    void release() {
        if (program_id_ != 0) {
            glDeleteProgram(program_id_);
            program_id_ = 0;
        }
        if (vertex_shader_) {
            glDeleteShader(vertex_shader_);
            vertex_shader_ = 0;
        }
        if (fragment_shader_) {
            glDeleteShader(fragment_shader_);
            fragment_shader_ = 0;
        }
    }

private:
    bool checkShader(unsigned shader) {
        int status = GL_FALSE;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &status);
        if (status == GL_FALSE) {
            char log[1024];
            glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
            LOGE("glGetShaderiv failed: %s", log);
            return false;
        }
        return true;
    }

    bool checkProgram(unsigned program) {
        int status = GL_FALSE;
        glGetProgramiv(program, GL_LINK_STATUS, &status);
        if (status == GL_FALSE) {
            char log[1024];
            glGetProgramInfoLog(program, sizeof(log), nullptr, log);
            LOGE("glGetProgramiv failed: %s", log);
            return false;
        }
        return true;
    }



private:
    unsigned program_id_ = 0;
    unsigned vertex_shader_ = 0;
    unsigned fragment_shader_ = 0;
};

#endif // __GL_SHADER_H__
