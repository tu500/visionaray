find_package(GLUT REQUIRED)

visionaray_use_package(GLUT)

set(EX_SNEX_SOURCES
    main.cpp
    snex.h
)

visionaray_add_executable(snex
    ${EX_SNEX_SOURCES}
)

add_custom_target(run-snex snex DEPENDS snex WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/ )
