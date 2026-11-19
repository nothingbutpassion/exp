cmake_minimum_required(VERSION 3.10)
project(dual_projector)

find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

# Add GLAD (if you have glad.c in your src folder)
set(GLAD_ROOT "$ENV{HOME}/glad")
add_library(glad ${GLAD_ROOT}/src/glad.c)
target_include_directories(glad PUBLIC ${GLAD_ROOT}/include)

# dp_calib
add_executable(
    dp_calib
    dp_calib.cpp
    chessboard_calibrator.cpp
    display_helper.cpp
    calib_parameters.cpp
    dual_projection.cpp
)
target_include_directories(
    dp_calib PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(
    dp_calib PRIVATE
    ${OpenCV_LIBS}
    OpenGL::GL
    glad
    glfw
)

# dp_render
set(VIDEO_RENDER_EX ON)
set(RENDER_SRC
    dp_render.cpp
    calib_parameters.cpp
)
if (VIDEO_RENDER_EX)
    list(APPEND RENDER_SRC video_render_ex.cpp)
    add_executable(
        dp_render
        ${RENDER_SRC}
    )
    target_compile_definitions(
        dp_render PRIVATE
        -D VIDEO_RENDER_EX
        VideoRender=VideoRenderEx
    )
else ()
    list(APPEND RENDER_SRC video_render.cpp)
    add_executable(
        dp_render
        ${RENDER_SRC}
    )
endif()
target_include_directories(
    dp_render PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(
    dp_render PRIVATE
    ${OpenCV_LIBS}
    OpenGL::GL
    glad
    glfw
)

# af_pattern
add_executable(
    af_pattern
    autofocus_pattern.cpp
)
target_include_directories(
    af_pattern PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(
    af_pattern PRIVATE
    ${OpenCV_LIBS}
)

# af_emulator
add_executable(
    af_emulator
    af_emulator.cpp
    autofocus_emulator.cpp
    autofocus_utils.cpp
    display_helper.cpp
)
target_include_directories(
    af_emulator PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(
    af_emulator PRIVATE
    ${OpenCV_LIBS}
    OpenGL::GL
    glad
    glfw
)

# dp_player
add_executable(
    dp_player
    dp_player.cpp
    calib_parameters.cpp
    face_detector.cpp
    video_render.cpp
    yunet.cpp
)
target_include_directories(
    dp_player PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(
    dp_player PRIVATE
    ${OpenCV_LIBS}
    OpenGL::GL
    glad
    glfw
)