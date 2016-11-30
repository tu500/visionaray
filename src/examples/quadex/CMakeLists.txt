find_package(GLUT REQUIRED)

visionaray_use_package(GLUT)
if (VSNRAY_ENABLE_CUDA)
find_package(CUDA)
endif()
if (VSNRAY_ENABLE_CUDA)
visionaray_use_package(CUDA)
endif()


if(CUDA_FOUND AND VSNRAY_ENABLE_CUDA)
    visionaray_cuda_compile(EX_QUAD_CUDA_SOURCES
        main.cu
    )
else()
    set(EX_QUAD_SOURCES
        main.cpp
    )
endif()

visionaray_add_executable(quadex
    ${EX_QUAD_SOURCES}
    ${EX_QUAD_CUDA_SOURCES}
)

add_custom_target(run-quadex quadex DEPENDS quadex WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ )
