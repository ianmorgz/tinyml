    include(CTest)
    enable_testing()

    include(FetchContent)

    FetchContent_Declare(
            googletest
            URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
    )

    # Prevent MSVC runtime mismatch (safe on all platforms)
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(googletest)

    add_executable(tinyml_tests
            # core
            core/shape_test.cpp
            core/buffer_view_test.cpp

#            #tensor
#            tensor/view_test.cpp
#            tensor/tensor_test.cpp
#
#            #internal
#            internal/aligned_buffer_test.cpp
#
#            #model
#            model/dense_test.cpp
#            model/relu_test.cpp
#            model/sequential_test.cpp
#
#            #train
#            train/context_tests.cpp

            # add more test files here
    )

    target_link_libraries(tinyml_tests
            PRIVATE
            tinyml
            GTest::gtest_main
    )

    include(GoogleTest)
    gtest_discover_tests(tinyml_tests)
