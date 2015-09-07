PROJECT(NiftyMatch)
CMAKE_MINIMUM_REQUIRED(VERSION 3.2.0)

# C++11
INCLUDE(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
if(COMPILER_SUPPORTS_CXX11)
    list(APPEND CMAKE_CXX_FLAGS "-std=c++11")
else()
    check_cxx_compiler_flag("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
    if(COMPILER_SUPPORTS_CXX0X)
        list(APPEND CMAKE_CXX_FLAGS "-std=c++0x")
    else()
        message(FATAL_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
    endif()
endif()

# CUDA
FIND_PACKAGE(CUDA)
MARK_AS_ADVANCED(CUDA_SDK_ROOT_DIR)
if(CUDA_FOUND)
    # Display the CUDA version
    MESSAGE(STATUS "The CUDA version you have is v${CUDA_VERSION_MAJOR}.${CUDA_VERSION_MINOR}")
    # Compile an executable to check if there is at least one suitable graphical card
    TRY_RUN(RUN_RESULT_VAR COMPILE_RESULT_VAR
            ${CMAKE_BINARY_DIR}
            ${CMAKE_SOURCE_DIR}/gpu/check/check_cuda_card.cpp
            CMAKE_FLAGS
            -DINCLUDE_DIRECTORIES:STRING=${CUDA_INCLUDE_DIRS}
            -DLINK_LIBRARIES:STRING=${CUDA_LIBRARIES}
            COMPILE_OUTPUT_VARIABLE COMPILE_OUTPUT_VAR
            RUN_OUTPUT_VARIABLE RUN_OUTPUT_VAR
    )
    # Check if the executable could not compile
    if(NOT COMPILE_RESULT_VAR)
        MESSAGE(STATUS "The code to check the presence of a CUDA-enabled card failed.")
        MESSAGE(FATAL_ERROR "CUDA not found. This application requires CUDA.")
    else(NOT COMPILE_RESULT_VAR)
        # Check if the executable return failure
        if(RUN_RESULT_VAR)
            MESSAGE(STATUS "No CUDA-enabled card has been detected.")
            MESSAGE(STATUS "Result message: ${RUN_RESULT_VAR}")
            MESSAGE(STATUS "Error message: ${RUN_OUTPUT_VAR}")
            MESSAGE(FATAL_ERROR "CUDA enabled card not found. This application requires CUDA.")
        else(RUN_RESULT_VAR)
            # Check if ptxas information should be displayed
            LIST(APPEND CUDA_NVCC_FLAGS "-arch=sm_30;-std=c++11;-rdc=true;-O2;-DVERBOSE")
            SET(CUDA_SEPARABLE_COMPILATION ON)
            SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
            ADD_DEFINITIONS(-DUSE_CUDA)
            MESSAGE(STATUS "Found CUDA and a CUDA-enabled card.")
        endif(RUN_RESULT_VAR)
    endif(NOT COMPILE_RESULT_VAR)
else(CUDA_FOUND)
    MESSAGE(FATAL_ERROR "CUDA not found. This application requires CUDA.")
endif(CUDA_FOUND)

INCLUDE_DIRECTORIES(${CUDA_INCLUDE_DIRS})
INCLUDE_DIRECTORIES(${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc) # helper_*
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/utils)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/gpu/kernels)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/gpu/utils)
INCLUDE_DIRECTORIES(${CMAKE_SOURCE_DIR}/gpu/sift)

SUBDIRS(${CMAKE_SOURCE_DIR}/gpu/kernels)
SUBDIRS(${CMAKE_SOURCE_DIR}/gpu/utils)
SUBDIRS(${CMAKE_SOURCE_DIR}/gpu/sift)

INSTALL(FILES
    utils/macros.h DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
