project(tracer-scene)

set(SOURCES
    objects/AObject.cpp
    objects/Camera.cpp
    objects/Plane.cpp
    objects/Sphere.cpp
    SceneData.cpp
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
