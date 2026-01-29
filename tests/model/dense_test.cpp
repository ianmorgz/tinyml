#include <gtest/gtest.h>
#include <tinyml/model/dense.hpp>
#include <tinyml/model/layer.hpp>
#include <tinyml/core/shape.hpp>
#include <tinyml/tensor/tensor_view.hpp>
#include <tinyml/tensor/tensor.hpp>

TEST(DenseLayer, Constructor) {
    EXPECT_ANY_THROW(tinyml::model::Dense(0, 10));
    EXPECT_ANY_THROW(tinyml::model::Dense(10, 0));

    const tinyml::model::Dense d_layer(4, 12);
    EXPECT_EQ(d_layer.in_features(), 4);
    EXPECT_EQ(d_layer.out_features(), 12);
    EXPECT_EQ(d_layer.type(), tinyml::model::LayerType::Dense);

    EXPECT_EQ(d_layer.params().size(), 2);
}

TEST(DenseLayer, inferOutputShape) {
    const tinyml::model::Dense d_layer(4, 12);

    EXPECT_EQ(d_layer.infer_output_shape(tinyml::core::Shape({4}))[0], 12);
    EXPECT_EQ(d_layer.infer_output_shape(tinyml::core::Shape({32, 4}))[1], 12);

    EXPECT_ANY_THROW(d_layer.infer_output_shape(tinyml::core::Shape({5})));
    EXPECT_ANY_THROW(d_layer.infer_output_shape(tinyml::core::Shape({32, 5})));
}

TEST(DenseLayer, Forward) {
    tinyml::model::Dense d_layer(3, 2);
    d_layer.init_zeros();

    const tinyml::tensor::Tensor<float> i(tinyml::core::Shape({3}), 1.0f);
    tinyml::tensor::Tensor<float> o(tinyml::core::Shape({2}), 1.0f);

    tinyml::tensor::TensorView<const float> i_v = i.view();
    tinyml::tensor::TensorView<float> o_v = o.view();
    EXPECT_NO_THROW(d_layer.forward(i_v, o_v));

    // since both weights and biases are initialized to 0, should output 0s in the output tensor_view
    ASSERT_EQ(o.size(), 2);
    EXPECT_EQ(o[0], 0.0f);
    EXPECT_EQ(o[1], 0.0f);

    const tinyml::tensor::Tensor<float> i_2(tinyml::core::Shape({4}), 1.0f);
    const auto i_v2 = i_2.view();
    EXPECT_ANY_THROW(d_layer.forward(i_v2, o_v));
}

TEST(DenseLayer, BatchedForward) {
    tinyml::model::Dense d_layer(3, 2);
    d_layer.init_zeros();

    const tinyml::tensor::Tensor<float> i(tinyml::core::Shape({5, 3}), 1.0f);
    tinyml::tensor::Tensor<float> o(tinyml::core::Shape({5, 2}), 1.0f);

    tinyml::tensor::TensorView<const float> i_v = i.view();
    tinyml::tensor::TensorView<float> o_v = o.view();
    EXPECT_NO_THROW(d_layer.forward(i_v, o_v));

    // since both weights and biases are initialized to 0, should output 0s in the output tensor_view
    ASSERT_EQ(o.size(), 10);
    for (size_t j = 0; j < 10; j++) {
        EXPECT_EQ(o[j], 0.0f);
    }
}

