add_subdirectory(objects)
add_subdirectory(renderer)
add_subdirectory(scene)
add_subdirectory(utils)

add_executable(tracer
    main.cpp
)

target_link_libraries(tracer
    PRIVATE
        tracer-renderer
        tracer-scene
        tracer-utils
        tracer-objects
)
