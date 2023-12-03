project(tracer-utils)

set(SOURCES
    Image.cpp
    InputParser.cpp
    Measurements.cpp
)

add_library(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE ${SOURCES})
target_include_directories(${PROJECT_NAME}
    PUBLIC
        include
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        tracer-containers
)
