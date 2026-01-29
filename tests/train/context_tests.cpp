#include <../../include/tinyml/core/context.hpp>
#include <gtest/gtest.h>

#include "tinyml/model/dense.hpp"
#include "tinyml/model/relu.hpp"
#include "tinyml/model/sequential.hpp"

TEST(TrainingContext, ContextStoresLayerParams) {
    tinyml::tensor::Tensor<float> t1({4, 3}, 1.0f);
    tinyml::tensor::Tensor<float> t2({5, 10}, 2.0f);

    tinyml::train::TrainingContext ctx;
    ctx.resize(2);
    ASSERT_NO_THROW(ctx.save_input(0, t1.view().as_const()));
    ASSERT_NO_THROW(ctx.save_input(1, t2.view().as_const()));

    ASSERT_EQ(ctx.saved_input(0).shape(), t1.shape());
    ASSERT_EQ(ctx.saved_input(1).shape(), t2.shape());
}

