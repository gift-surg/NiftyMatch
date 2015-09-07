SET(NAME sift)
SET(HEADERS
	siftparams.h
    pyramidata.h
    siftdata.h
    siftfunctions.h)
CUDA_ADD_LIBRARY(${NAME}
	${HEADERS}
    pyramidata.cu
    siftdata.cu
    siftfunctions.cu)
TARGET_LINK_LIBRARIES(${NAME} kernels)
INSTALL(FILES
    ${HEADERS} DESTINATION ${CMAKE_INSTALL_PREFIX}/include)
INSTALL(TARGETS ${NAME}
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
    LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
    ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)