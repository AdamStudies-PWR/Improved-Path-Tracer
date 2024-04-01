project(tracer-renderer)

set(SOURCES
    helpers/LoopRange.cu
    helpers/SceneConstants.cpp
    RenderController.cu
    RendererCPU.cpp
    RendererGPU.cu
)

add_library(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE ${SOURCES})
target_include_directories(${PROJECT_NAME}
    PUBLIC
        include
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(${PROJECT_NAME}
        PUBLIC
            OpenMP::OpenMP_CXX)
endif()

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        tracer-containers
        tracer-scene
        tracer-utils
)
