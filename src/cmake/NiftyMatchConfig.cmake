# Try to find NiftyMatch. Once done, this will define:
#
#  NiftyMatch_FOUND - system has NiftyMatch
#  NiftyMatch_INCLUDE_DIR - the NiftyMatch include directories
#  NiftyMatch_LIBS - link these to use NiftyMatch

# Include dir
FIND_PATH(NiftyMatch_INCLUDE_DIR
	NAMES macros.h)

# And the modules of this library
FIND_LIBRARY(NiftyMatch_utils_LIB
	NAMES gpuutils)
FIND_LIBRARY(NiftyMatch_kernels_LIB
	NAMES kernels)
FIND_LIBRARY(NiftyMatch_sift_LIB
	NAMES sift)

# Put them all into a var
SET(NiftyMatch_LIBS
	${NiftyMatch_utils_LIB}
	${NiftyMatch_kernels_LIB}
	${NiftyMatch_sift_LIB})

# handle the QUIETLY and REQUIRED arguments and set NiftyMatch_FOUND
# if all listed variables are TRUE
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
	NiftyMatch DEFAULT_MSG
	NiftyMatch_LIBS NiftyMatch_INCLUDE_DIR)