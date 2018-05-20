if(DESKVOX_USE_CUDA)

find_package(CUDA)

deskvox_use_package(CUDA INCDIRS ${CUDA_TOOLKIT_INCLUDE})

endif(DESKVOX_USE_CUDA)

deskvox_link_libraries(virvo)
deskvox_link_libraries(virvo_fileio)

set(RAYREND_HEADERS
    ../../vvraycaster.h
)

add_definitions(-DHAVE_CONFIG_H)

if(DESKVOX_USE_CUDA)
add_definitions(-DVV_ARCH_CUDA=1)
deskvox_cuda_compiles(RAYREND_SCATTER_SOURCES
    ../../vvscattercaster.cu
)
else()
set(RAYREND_SCATTER_SOURCES
    ../../vvscattercaster.cpp
)
endif()

deskvox_add_library(rayrendscatter
    ${RAYREND_HEADERS}
    ${RAYREND_SCATTER_SOURCES}
)

set_target_properties(rayrendscatter PROPERTIES LINKER_LANGUAGE CXX)

# MSVC + CUDA: link with legacy stdio library

if (MSVC_VERSION GREATER 1800 AND DESKVOX_USE_CUDA AND CUDA_FOUND)
    target_link_libraries(rayrendscatter legacy_stdio_definitions.lib)
endif()
