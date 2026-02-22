cmake_minimum_required(VERSION 3.20)

if(POLICY CMP0135)
    cmake_policy(SET CMP0135 NEW)
endif()


project(tinyml VERSION 0.4.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ----------- Options ----------
option(TINYML_BUILD_TESTS "Build unit tests" ON)
option(TINYML_BUILD_EXAMPLES "Build examples" ON)
option(TINYML_ENABLE_SANITIZERS "Enable ASan/UBSan on supported platforms" ON)

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

        src/quant/qsequential.cpp
        src/quant/qdense.cpp
        src/quant/qrelu.cpp
        src/quant/qtanh.cpp
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

    add_executable(ex_mlp examples/ex_mlp/ex_mlp.cpp)
    target_link_libraries(ex_mlp PRIVATE tinyml)
endif()

# ---------- Tests ----------
if (TINYML_BUILD_TESTS)
    add_subdirectory(tests)

endif()

# ---------- Sanitizers (Debug on host) ----------
if (TINYML_ENABLE_SANITIZERS AND NOT MSVC)
    set(TINYML_SAN_FLAGS -fsanitize=address,undefined -fno-omit-frame-pointer)

    # Apply to library compile
    target_compile_options(tinyml PRIVATE ${TINYML_SAN_FLAGS})

    # Apply to any final link targets (executables)
    if (TINYML_BUILD_TESTS)
        target_compile_options(tinyml_tests PRIVATE ${TINYML_SAN_FLAGS})
        target_link_options(tinyml_tests PRIVATE ${TINYML_SAN_FLAGS})
    endif()

    if (TINYML_BUILD_EXAMPLES)
        target_compile_options(mnist_mlp PRIVATE ${TINYML_SAN_FLAGS})
        target_link_options(mnist_mlp PRIVATE ${TINYML_SAN_FLAGS})

        target_compile_options(ex_mlp PRIVATE ${TINYML_SAN_FLAGS})
        target_link_options(ex_mlp PRIVATE ${TINYML_SAN_FLAGS})
    endif()
endif()

