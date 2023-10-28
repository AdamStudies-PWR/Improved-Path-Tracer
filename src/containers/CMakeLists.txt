project(tracer-containers)

set(SOURCES
    Ray.cpp
    Vec.cpp
)

add_library(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE ${SOURCES})
target_include_directories(${PROJECT_NAME}
    PUBLIC
        include
)
