cmake_minimum_required(VERSION 3.20)

project(tinyml VERSION 0.4.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)

# ----------- Options ----------
option(TINYML_BUILD_EXAMPLES "Build examples" ON)

add_library(tinyml STATIC library.cpp)

# ---------- Warnings ----------
if (MSVC)
    target_compile_options(tinyml PRIVATE /W4)
else()
    target_compile_options(tinyml PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---------- Examples ----------
if(TINYML_BUILD_EXAMPLES)
    add_executable(mnist examples/mnist/mnist.cpp)
    target_link_libraries(mnist PRIVATE tinyml)
endif()