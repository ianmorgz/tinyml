#include <gtest/gtest.h>
#include <tinyml/core/shape.hpp>
#include <tinyml/tensor/tensor.hpp>

TEST(TensorShape, NumDimAndFlatSize) {
    const tinyml::tensor::Tensor<float> t(tinyml::core::Shape{2, 3});

    EXPECT_EQ(t.shape().rank, 2);
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 3);
    EXPECT_EQ(t.shape().flat_size(), 6);
    EXPECT_FALSE(t.shape().empty());
}

TEST(TensorShape, EmptyTensor) {
    const tinyml::tensor::Tensor<float> t;

    EXPECT_EQ(t.shape().rank, 0);
    EXPECT_EQ(t.shape().flat_size(), 0);
    EXPECT_TRUE(t.shape().empty());
    EXPECT_ANY_THROW(t.shape()[0]);
}

TEST(TensorShape, LargeTensor) {
    const tinyml::tensor::Tensor<float> t({100, 100, 100});

    EXPECT_EQ(t.shape().rank, 3);
    EXPECT_EQ(t.shape()[0], 100);
    EXPECT_EQ(t.shape()[1], 100);
    EXPECT_EQ(t.shape()[2], 100);
    EXPECT_EQ(t.shape().flat_size(), 1e+6);
}

TEST(TensorShape, TensorThrowException) {
    const tinyml::tensor::Tensor<float> t(tinyml::core::Shape{2, 3});

    EXPECT_ANY_THROW(t.shape()[2]);
    EXPECT_ANY_THROW(t.shape()[100]);
}


