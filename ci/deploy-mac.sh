#!/bin/bash
cd "$NiftyMatch_BUILD_DIR"
make install
rm -rf "$Test_REPO_DIR"
git clone git@cmiclab.cs.ucl.ac.uk:GIFT-Surg/NiftyMatch-Test.git "$Test_REPO_DIR" --branch 5-rename-project
rm -rf "$Test_BUILD_DIR"
mkdir -p "$Test_BUILD_DIR"
cd "$Test_BUILD_DIR"
cmake -D NiftyMatch_DIR="$NiftyMatch_DIR" -D BUILD_TESTS=ON "$Test_SOURCE_DIR"
make