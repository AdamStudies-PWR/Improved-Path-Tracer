project(tracer-scene)

set(SOURCES
    objects/AObject.cu
    objects/Camera.cpp
    objects/ObjectData.cpp
    SceneData.cu
)

add_library(${PROJECT_NAME})
target_sources(${PROJECT_NAME} PRIVATE ${SOURCES})
target_include_directories(${PROJECT_NAME}
    PUBLIC
        include
)

target_link_libraries(${PROJECT_NAME}
    PUBLIC
        nlohmann_json::nlohmann_json
    PRIVATE
        tracer-containers
)
