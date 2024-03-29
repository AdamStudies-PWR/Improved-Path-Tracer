project(tracer-renderer)

set(SOURCES
    Renderer.cu
    RenderController.cu
    helpers/Coordinates.cu
    helpers/ImageData.cpp
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
        tracer-scene
        tracer-utils
)
