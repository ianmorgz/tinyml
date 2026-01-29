#pragma once

#include "tinyml/tensor/tensor.hpp"
#include "tinyml/tensor/tensor_view.hpp"

namespace tinyml::dataset {
struct Batch {
    tensor::Tensor<float> input;
    tensor::Tensor<float> label;
    std::size_t size;

    Batch() = default;
};

struct BatchView {
    tensor::TensorView<const float> input;
    tensor::TensorView<const float> label;
    std::size_t size;

    BatchView() = default;

    explicit BatchView(const Batch& b)
    : input(b.input.view()),
      label(b.label.view()),
      size(b.size) {}
};
}