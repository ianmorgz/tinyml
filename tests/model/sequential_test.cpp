#include <gtest/gtest.h>
#include <tinyml/model/sequential.hpp>

#include <iostream>

#include "tinyml/model/dense.hpp"
#include "tinyml/model/relu.hpp"
#include "../../include/tinyml/core/context.hpp"

TEST(SequentialModel, BuiltOutputsCorrectShape) {
    tinyml::model::Sequential m;
    ASSERT_NO_THROW(m.add(std::make_unique<tinyml::model::Dense>(4, 3)));
    ASSERT_NO_THROW(m.add(std::make_unique<tinyml::model::Relu>()));
    ASSERT_NO_THROW(m.add(std::make_unique<tinyml::model::Dense>(3, 2)));
    ASSERT_NO_THROW(m.build(4, 1));

    EXPECT_EQ(m.num_layers(), 3);
    EXPECT_EQ(m.layer(0).type(), tinyml::model::LayerType::Dense);
    EXPECT_EQ(m.layer(1).type(), tinyml::model::LayerType::ReLu);
    EXPECT_EQ(m.layer(2).type(), tinyml::model::LayerType::Dense);

    EXPECT_EQ(m.input_shape()[0], 4);
    EXPECT_EQ(m.output_shape()[0], 2);
}

TEST(SequentialModel, SequentialForward) {
    tinyml::model::Sequential m;
    ASSERT_NO_THROW(m.add(std::make_unique<tinyml::model::Dense>(4, 3)));
    ASSERT_NO_THROW(m.add(std::make_unique<tinyml::model::Relu>()));
    ASSERT_NO_THROW(m.add(std::make_unique<tinyml::model::Dense>(3, 2)));
    ASSERT_NO_THROW(m.build(4, 2));

    const tinyml::tensor::Tensor<float> input({2, 4}, 1.0f);
    tinyml::tensor::TensorView<const float> output;

    EXPECT_EQ(m.params().size(), 4);

    ASSERT_NO_THROW(output = m.forward(input.view()));
    ASSERT_EQ(output.shape()[0], 2);
    ASSERT_EQ(output.shape()[1], 2);
}

TEST(SequentialModel, SequentialForwardTrain) {
    tinyml::model::Sequential m;
    tinyml::train::TrainingContext ctx;
    ASSERT_NO_THROW(m.add(std::make_unique<tinyml::model::Dense>(4, 3)));
    ASSERT_NO_THROW(m.add(std::make_unique<tinyml::model::Relu>()));
    ASSERT_NO_THROW(m.add(std::make_unique<tinyml::model::Dense>(3, 2)));
    ASSERT_NO_THROW(m.build({2, 4}));

    const tinyml::tensor::Tensor<float> input({2, 4}, 1.0f);
    tinyml::tensor::TensorView<const float> output;

    EXPECT_EQ(m.params().size(), 4);

    ASSERT_NO_THROW(output = m.forward_train(input.view(), ctx));
    ASSERT_EQ(output.shape()[0], 2);
    ASSERT_EQ(output.shape()[1], 2);

    ASSERT_EQ(ctx.size(), 3);
    ASSERT_EQ(ctx.saved_input(0).shape(), tinyml::core::Shape({2, 4}));
    ASSERT_EQ(ctx.saved_input(1).shape(), tinyml::core::Shape({2, 3}));
    ASSERT_EQ(ctx.saved_input(2).shape(), tinyml::core::Shape({2, 3}));
}
