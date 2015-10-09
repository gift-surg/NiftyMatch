#!/bin/bash
cd "$NiftyMatch_BUILD_DIR"
make install
rm -rf "$cuSIFT_REPO_DIR"
git clone git@cmiclab.cs.ucl.ac.uk:GIFT-Surg/cuSIFT.git "$cuSIFT_REPO_DIR" --branch dev
rm -rf "$cuSIFT_BUILD_DIR"
mkdir -p "$cuSIFT_BUILD_DIR"
cd "$cuSIFT_BUILD_DIR"
cmake -D NiftyMatch_DIR="$NiftyMatch_DIR" -D BUILD_TESTS=ON "$cuSIFT_SOURCE_DIR"
make