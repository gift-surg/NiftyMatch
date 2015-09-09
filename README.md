# Summary
NiftyMatch is a library featuring GPU implementations of feature detection and matching algorithms.

# Build
Currently Mac and Linux only.
```
git clone <this-repo> NiftyMatch
mkdir NiftyMatch-build
cd NiftyMatch-build
cmake ../NiftyMatch/src
make -j
sudo make install
```

# Use
1. Specify `NiftyMatch_DIR` as `/usr/local/include/nm` for CMake.
1. Put `FIND_PACKAGE(NiftyMatch CONFIG REQUIRED)` into your project's CMake file.
1. See `src/cmake/NiftyMatchConfig.cmake` (installed in `/usr/local/include/nm/NiftyMatchConfig.cmake`) to see which CMake variables to use.