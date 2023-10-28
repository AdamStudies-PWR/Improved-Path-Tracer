add_subdirectory(containers)
add_subdirectory(data)
add_subdirectory(objects)
add_subdirectory(renderer)
add_subdirectory(utils)

add_executable(tracer
    main.cpp
)

target_link_libraries(tracer
    PRIVATE
        tracer-containers
        tracer-renderer
        tracer-data
        tracer-utils
        tracer-objects
)
