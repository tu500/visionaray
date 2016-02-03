set(COMMON_DIR ${PROJECT_SOURCE_DIR}/src/common)


find_package(Boost COMPONENTS chrono filesystem iostreams system thread REQUIRED)
if (VSNRAY_ENABLE_CUDA)
find_package(CUDA)
endif()
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)
if (NOT APPLE AND NOT WIN32)
find_package(PTHREAD REQUIRED)
endif()

visionaray_use_package(Boost)
if (VSNRAY_ENABLE_CUDA)
visionaray_use_package(CUDA)
endif()
visionaray_use_package(GLEW)
visionaray_use_package(GLUT)
visionaray_use_package(OpenGL)
if (NOT APPLE AND NOT WIN32)
visionaray_use_package(PTHREAD)
endif()

#find_package(TBB)
#visionaray_use_package(TBB)

visionaray_link_libraries(visionaray)
visionaray_link_libraries(visionaray_common)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)

#if(CUDA_FOUND AND VSNRAY_ENABLE_CUDA)
#    visionaray_cuda_compile(FPSTEST_CUDA_SOURCES
#        fpstest.cu
#    )
#else()
    set(FPSTEST_SOURCES
        fpstest.cpp
        test_model.cpp
    )
#endif()

set(FPSTEST_PACKET_SIZE 1 CACHE INTEGER "Number of simultaneous rays to be used in fpstest")
add_definitions(-DFPSTEST_PACKET_SIZE=${FPSTEST_PACKET_SIZE})

set(FPSTEST_SCHEDULER SCHEDULER_TILED CACHE STRING "Scheduler to be used")
add_definitions(-DFPSTEST_SCHEDULER=${FPSTEST_SCHEDULER})

set(FPSTEST_ALGO ALGO_SIMPLE CACHE STRING "Algorithm to be used")
add_definitions(-DFPSTEST_ALGO=${FPSTEST_ALGO})


#--------------------------------------------------------------------------------------------------
# Add fpstest target
#

visionaray_add_executable(fpstest
    ${FPSTEST_SOURCES}
    ${FPSTEST_CUDA_SOURCES}
)

add_custom_target(run-fpstest fpstest ~/vis/teapot.obj DEPENDS fpstest WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ )


#--------------------------------------------------------------------------------------------------
# Install fpstest
#

install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/fpstest
    DESTINATION bin
    RENAME vsnray-fpstest
)
