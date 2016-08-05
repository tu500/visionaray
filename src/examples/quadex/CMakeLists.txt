find_package(GLUT REQUIRED)

visionaray_use_package(GLUT)

set(EX_QUAD_SOURCES
    main.cpp
)

visionaray_add_executable(quadex
    ${EX_QUAD_SOURCES}
)

add_custom_target(run-quadex quadex DEPENDS quadex WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ )
