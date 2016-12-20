find_package(GLUT REQUIRED)

visionaray_use_package(GLUT)
if (VSNRAY_ENABLE_CUDA)
find_package(CUDA)
endif()
if (VSNRAY_ENABLE_CUDA)
visionaray_use_package(CUDA)
endif()


### QUADEX ###

if(CUDA_FOUND AND VSNRAY_ENABLE_CUDA AND 1)
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


### SNEX ###

if(CUDA_FOUND AND VSNRAY_ENABLE_CUDA AND 1)
    visionaray_cuda_compile(EX_SNEX_CUDA_SOURCES
        snex.cu
    )
else()
    set(EX_SNEX_SOURCES
        snex.cpp
    )
endif()

visionaray_add_executable(snex
    ${EX_SNEX_SOURCES}
    ${EX_SNEX_CUDA_SOURCES}
)

add_custom_target(run-quadex quadex DEPENDS quadex WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ )
add_custom_target(run-snex snex DEPENDS snex WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ )
