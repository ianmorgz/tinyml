#include <gtest/gtest.h>
#include <tinyml/core/shape.hpp>
#include <tinyml/tensor/tensor.hpp>
#include <tinyml/tensor/tensor_view.hpp>


TEST(TensorView, Shape) {
    const tinyml::tensor::Tensor<float> t( tinyml::core::Shape{2, 3});
    const auto t_v = t.view();

    EXPECT_EQ(t_v.shape().rank, 2);
    EXPECT_EQ(t_v.shape()[0], 2);
    EXPECT_EQ(t_v.shape()[1], 3);
    EXPECT_EQ(t_v.shape().flat_size(), 6);
    EXPECT_FALSE(t_v.shape().empty());
}

TEST(TensorView, EmptyTensorView) {
    const tinyml::tensor::Tensor<float> t(tinyml::core::Shape{2, 3} );
    const auto t_v = t.view();

    ASSERT_NE(t.data(), nullptr);

    EXPECT_EQ(t_v.data(), t.data());
    EXPECT_EQ(t_v.size(), t.size());
    EXPECT_EQ(t_v.rank(), t.rank());
}

TEST(TensorView, TensorViewCreation) {
    const tinyml::tensor::Tensor<float> t({2, 3}, 1.0f);
    const auto t_v = t.view();

    ASSERT_NE(t_v.data(), nullptr);

    EXPECT_EQ(t_v.data(), t.data());
    EXPECT_EQ(t_v.size(), t.size());
    EXPECT_EQ(t_v.rank(), t.rank());
}

TEST(TensorView, TensorViewCopyable) {
    const tinyml::tensor::Tensor<float> t({2, 3}, 1.0f);
    const auto t_v1 = t.view();
    const auto t_v2 = t_v1;

    EXPECT_EQ(t_v1.size(), t_v2.size());
    EXPECT_EQ(t_v1.rank(), t_v2.rank());
    EXPECT_EQ(t_v1.data(), t_v2.data());
    EXPECT_EQ(t_v1.shape()[0], t.shape()[0]);
    EXPECT_EQ(t_v1.shape()[1], t.shape()[1]);
}

TEST(TensorView, Accessors) {
    tinyml::tensor::Tensor<float> t(tinyml::core::Shape{2, 4}, 3.0f);
    const auto t_v = t.view();

    ASSERT_EQ(t_v.size(), 8);
    for (size_t i = 0; i < t.size(); i++) {
        EXPECT_EQ(t_v[i], 3.0f);
    }

    EXPECT_EQ(t_v.at(0, 0), 3.0f);
    EXPECT_EQ(t_v.at(1, 0), 3.0f);
    EXPECT_ANY_THROW(t_v.at(2, 0));
    EXPECT_ANY_THROW(t_v.at(1, 4));
}

TEST(TensorView, TensorViewReferencesOwner) {
    tinyml::tensor::Tensor<float> t({2, 3}, 1.0f);
    const auto t_v = t.view();

    t.fill(2.0f);

    EXPECT_EQ(t_v.data(), t.data());
    EXPECT_EQ(t_v.size(), t.size());
    EXPECT_EQ(t_v.rank(), t.rank());
}

TEST(TensorView, TensorViewDestroyable) {
    tinyml::tensor::Tensor<float> t({2, 3}, 1.0f);
    if constexpr (true ) {
        const auto t_v = t.view();
        EXPECT_EQ(t_v.data(), t.data());
        EXPECT_EQ(t_v.size(), t.size());
        EXPECT_EQ(t_v.rank(), t.rank());
    }
    //t_v should be destroyed
    EXPECT_EQ(t.data()[3], 1.0f);
}

TEST(TensorView, TensorViewMutable) {
    tinyml::tensor::Tensor<float> t({2, 3}, 1.0f);
    const auto t_v = t.view();
    ASSERT_NO_THROW(t_v.data()[0] = 2.0f);
    EXPECT_EQ(t_v.data()[0], 2.0f);
    EXPECT_EQ(t.data()[0], 2.0f);
}

TEST(TensorView, MultipleTensorViewWriting) {
    tinyml::tensor::Tensor<float> t({2, 3}, 1.0f);
    const auto t_v1 = t.view();
    const auto t_v2 = t.view();

    t_v1.data()[1] = 2.0f;
    EXPECT_EQ(t_v2.data()[1], 2.0f);

    t_v2.data()[4] = 5.0f;
    EXPECT_EQ(t_v1.data()[4], 5.0f);
}


