#!/bin/bash
rm -rf "$NiftyMatch_BUILD_DIR"
mkdir -p "$NiftyMatch_BUILD_DIR"
cd "$NiftyMatch_BUILD_DIR"
cmake -D CMAKE_INSTALL_PREFIX="$INSTALL_DIR" -D CHECK_CUDA_HARDWARE=OFF "$NiftyMatch_SOURCE_DIR"
make -j