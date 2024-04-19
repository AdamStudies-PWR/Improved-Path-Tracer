project(tracer-renderer)

set(SOURCES
    helpers/ImageData.cpp
    helpers/Range.cu
    helpers/RenderData.cu
    RenderController.cu
    Renderer.cu
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
