#include <gtest/gtest.h>
#include <tinyml/model/relu.hpp>
#include <tinyml/tensor/tensor.hpp>
#include <tinyml/tensor/tensor_view.hpp>

TEST(Relu, Shape) {
    const tinyml::model::Relu relu;
    EXPECT_EQ(relu.infer_output_shape({3u, 2u})[0], 3);
    EXPECT_EQ(relu.infer_output_shape({3u, 2u})[1], 2);
}

TEST(Relu, Forward) {
    const tinyml::model::Relu relu;
    const tinyml::tensor::Tensor<float> i1(tinyml::core::Shape({3u}), 2.0f);
    const tinyml::tensor::Tensor<float> i2(tinyml::core::Shape({3u}), -4.0f);
    tinyml::tensor::Tensor<float> o(tinyml::core::Shape({3u}), 1.0f);

    const tinyml::tensor::TensorView<const float> i1_v = i1.view();
    const tinyml::tensor::TensorView<const float> i2_v = i2.view();
    const tinyml::tensor::TensorView<float> o_v = o.view();

    EXPECT_NO_THROW(relu.forward(i1_v, o_v));

    for (std::size_t i =0; i < o.size(); ++i) {
        EXPECT_EQ(o[i], 2.0f);
    }

    EXPECT_NO_THROW(relu.forward(i2_v, o_v));

    for (std::size_t i =0; i < o.size(); ++i) {
        EXPECT_EQ(o[i], 0.0f);
    }

    const tinyml::tensor::Tensor<float> i3(tinyml::core::Shape({4u}), 2.0f);
    const tinyml::tensor::TensorView<const float> i3_v = i3.view();
    EXPECT_ANY_THROW(relu.forward(i3_v, o_v));
}