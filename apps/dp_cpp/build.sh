#!/bin/bash
set -e -x

mkdir -p build
rm -rf build/*
cd build
cmake   \
    -DOpenCV_DIR=$HOME/opencv/build/install/lib/cmake/opencv5   \
    -DCMAKE_BUILD_TYPE=Debug  \
    -G "Unix Makefiles"         \
    ..
cmake --build . --config Debug