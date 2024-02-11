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

find_package(ImageMagick REQUIRED COMPONENTS Magick++)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        tracer-containers
        ImageMagick::Magick++
)
