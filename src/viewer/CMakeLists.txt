set(COMMON_DIR ${PROJECT_SOURCE_DIR}/src/common)
set(CMD_LINE_DIR ${PROJECT_SOURCE_DIR}/src/common/CmdLine)
set(CMD_LINE_INCLUDE_DIR ${CMD_LINE_DIR}/include)

if (NOT EXISTS ${CMD_LINE_DIR}/.git)
    message(SEND_ERROR "Git submodules not initialized.\nPlease run \"git submodule update --init --recursive\"")
    return()
endif()


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

visionaray_link_libraries(visionaray)
visionaray_link_libraries(visionaray_common)

include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/src)
include_directories(${CMD_LINE_INCLUDE_DIR})

if(CUDA_FOUND AND VSNRAY_ENABLE_CUDA)
    visionaray_cuda_compile(VIEWER_CUDA_SOURCES
        viewer.cu
    )
else()
    set(VIEWER_SOURCES
        viewer.cpp
    )
endif()


#--------------------------------------------------------------------------------------------------
# Add viewer target
#

visionaray_add_executable(precompile-header precompiled.cpp main.cpp empty.cpp)

add_custom_command(TARGET precompile-header
    PRE_LINK
    COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/precompile-header.dir/precompiled.cpp.o ${PROJECT_SOURCE_DIR}/src/viewer/precompiled.h.gch
#    COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/precompile-header.dir/precompiled.cpp.o ${CMAKE_CURRENT_BINARY_DIR}/precompiled.h.gch
    COMMAND cp ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/precompile-header.dir/empty.cpp.o ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/precompile-header.dir/precompiled.cpp.o
)

#add_custom_command(OUTPUT precompiled.h.gch
#    DEPENDS precompile-header
#    IMPLICIT_DEPENDS c++ precompiled.h
#)

set_source_files_properties(precompiled.cpp PROPERTIES COMPILE_FLAGS "-x c++-header")

visionaray_add_executable(viewer
    ${VIEWER_SOURCES}
    ${VIEWER_CUDA_SOURCES}
#    precompiled.h.gch
)

add_dependencies(viewer precompile-header)

target_compile_options(viewer PUBLIC -H)


#--------------------------------------------------------------------------------------------------
# Install viewer
#

install(PROGRAMS ${CMAKE_CURRENT_BINARY_DIR}/viewer
    DESTINATION bin
    RENAME vsnray-viewer
)
