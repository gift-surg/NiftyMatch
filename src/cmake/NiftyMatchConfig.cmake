# Try to find NiftyMatch. Once done, this will define:
#
#  NiftyMatch_INCLUDE_DIR - the NiftyMatch include directories
#  NiftyMatch_LIBS - link these to use NiftyMatch
#
# If not using all libs:
#  NiftyMatch_gpuutils_LIB
#  NiftyMatch_kernels_LIB
#  NiftyMatch_sift_LIB

# to be kept in sync with CMakeLists.txt at top level
# allows defined suffix to be appended to all searched paths
SET(NiftyMatch_PATH_SUFFIX nm)

# Include dir
FIND_PATH(NiftyMatch_INCLUDE_DIR
	NAMES macros.h
	PATHS ${CMAKE_CURRENT_LIST_DIR}/../..
	PATH_SUFFIXES ${NiftyMatch_PATH_SUFFIX})

# And the modules of this library
FIND_LIBRARY(NiftyMatch_gpuutils_LIB
	NAMES gpuutils
	PATHS ${CMAKE_CURRENT_LIST_DIR}/../../lib
	PATH_SUFFIXES ${NiftyMatch_PATH_SUFFIX})
FIND_LIBRARY(NiftyMatch_kernels_LIB
	NAMES kernels
	PATHS ${CMAKE_CURRENT_LIST_DIR}/../../lib
	PATH_SUFFIXES ${NiftyMatch_PATH_SUFFIX})
FIND_LIBRARY(NiftyMatch_sift_LIB
	NAMES sift
	PATHS ${CMAKE_CURRENT_LIST_DIR}/../../lib
	PATH_SUFFIXES ${NiftyMatch_PATH_SUFFIX})

# Put them all into a var
SET(NiftyMatch_LIBS
	${NiftyMatch_gpuutils_LIB}
	${NiftyMatch_kernels_LIB}
	${NiftyMatch_sift_LIB})

# handle the QUIETLY and REQUIRED arguments and set NiftyMatch_FOUND
# if all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
	NiftyMatch DEFAULT_MSG
	NiftyMatch_LIBS NiftyMatch_INCLUDE_DIR)