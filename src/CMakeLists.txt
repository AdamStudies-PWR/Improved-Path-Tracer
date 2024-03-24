add_subdirectory(containers)
add_subdirectory(renderer)
add_subdirectory(scene)
add_subdirectory(utils)

add_executable(tracer
    main.cu
)

set_property(TARGET tracer PROPERTY CUDA_STANDARD 20)

if (CMAKE_BUILD_TYPE STREQUAL Debug)
    target_compile_options(tracer
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
        -G
        -v
    >)
else()
    target_compile_options(tracer
        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            -use_fast_math
            -Xptxas
            -O3
    >)
endif()

target_link_libraries(tracer
    PRIVATE
        tracer-containers
        tracer-renderer
        tracer-scene
        tracer-utils
)
