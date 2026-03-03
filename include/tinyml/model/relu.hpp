#pragma once
#include "layer.hpp"
#include <span>

namespace tinyml::model {
class Relu final : public Layer {
public:
    // Constructor
    Relu() = default;

    // Getters
    LayerType type() const noexcept override { return LayerType::ReLu; }
    bool cache_input() const noexcept override { return false; }

    std::span<ParamRef> params() noexcept override { return { }; }
    std::span<const ParamRef> params() const noexcept override { return { }; }

    // Interface functions
    core::Shape infer_output_shape(const core::Shape &in) const override { return in;}
    void forward(tensor::TensorView<const float> in, tensor::TensorView<float> out) const override;
    void backward(tensor::TensorView<const float> grad_input, tensor::TensorView<const float> cached, tensor::TensorView<float> grad_output) override;
};
}
