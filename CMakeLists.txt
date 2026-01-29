cmake_minimum_required(VERSION 3.20)

project(tinyml VERSION 0.4.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ----------- Options ----------
option(TINYML_BUILD_TESTS "Build unit tests" ON)
option(TINYML_BUILD_EXAMPLES "Build examples" ON)
option(TINYML_ENABLE_SANITIZERS "Enable ASan/UBSan on supported platforms" OFF)

# ---------- Library ----------
add_library(tinyml
        src/internal/aligned_buffer.cpp

        src/model/dense.cpp
        src/model/relu.cpp
        src/model/tanh.cpp
        src/model/sequential.cpp

        src/train/context.cpp
        src/train/fit.cpp
        src/train/optimizer.cpp
        src/train/ce_loss.cpp

        src/dataset/dataset.cpp
)

target_include_directories(tinyml
        PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_compile_features(tinyml PUBLIC cxx_std_20)

# ---------- Warnings ----------
if (MSVC)
    target_compile_options(tinyml PRIVATE /W4)
else()
    target_compile_options(tinyml PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---------- Examples ----------
if(TINYML_BUILD_EXAMPLES)
    add_executable(mnist examples/mnist_mlp/main.cpp)
    target_link_libraries(mnist PRIVATE tinyml)
endif()

# ---------- Tests ----------
#if (TINYML_BUILD_TESTS)
#    include(CTest)
#    enable_testing()
#
#    include(FetchContent)
#
#    FetchContent_Declare(
#            googletest
#            URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
#    )
#
#    # Prevent MSVC runtime mismatch (safe on all platforms)
#    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
#
#    FetchContent_MakeAvailable(googletest)
#
#    add_executable(tinyml_tests
#            # core
#            tests/core/shape_test.cpp
##            tests/core/buffer_view_test.cpp
##
##            #tensor
##            tests/tensor/view_test.cpp
##            tests/tensor/tensor_test.cpp
##
##            #internal
##            tests/internal/aligned_buffer_test.cpp
##
##            #model
##            tests/model/dense_test.cpp
##            tests/model/relu_test.cpp
##            tests/model/sequential_test.cpp
##
##            #train
##            tests/train/context_tests.cpp
#
#            # add more test files here
#    )
#
#    target_link_libraries(tinyml_tests
#            PRIVATE
#            tinyml
#            GTest::gtest_main
#    )
#
#    include(GoogleTest)
#    gtest_discover_tests(tinyml_tests)

#endif()