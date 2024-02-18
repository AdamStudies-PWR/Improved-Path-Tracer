cmake_minimum_required(VERSION 3.25.1)

if (NOT DEFINED IS_RELEASE)
    SET(IS_RELEASE TRUE)
endif()

if (${IS_RELEASE})
    set(CMAKE_BUILD_TYPE Release)
else()
    set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Werror")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

include(FetchContent)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz)
FetchContent_MakeAvailable(json)

project(tracer)

enable_language(CUDA)

add_subdirectory(src)
