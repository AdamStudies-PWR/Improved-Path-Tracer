project(tracer-scene)

set(SOURCES
    cuda/objects/Plane.cu
    cuda/objects/Sphere.cu
    objects/Camera.cpp
    objects/ObjectData.cpp
    SceneData.cpp
)

add_library(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE ${SOURCES})
target_include_directories(${PROJECT_NAME}
    PUBLIC
        include
        cuda
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        nlohmann_json::nlohmann_json
    PRIVATE
        tracer-containers
        tracer-utils
)
