#pragma once

#include <span>

#include "tinyml/model/layer.hpp"
#include "tinyml/tensor/tensor.hpp"
#include "tinyml/tensor/tensor_view.hpp"

namespace tinyml::model {
class Dense final : public Layer {

public:
    // Constructor
    Dense(std::uint32_t in_features, std::uint32_t out_features, std::size_t alignment = 64);

    // Getters
    LayerType type() const noexcept override { return LayerType::Dense; };
    core::Shape infer_output_shape(const core::Shape& in) const override;
    tensor::TensorView<const float> weights() const noexcept { return W_.view(); };
    tensor::TensorView<const float> biases() const noexcept { return B_.view(); };
    std::uint32_t in_features() const noexcept { return in_features_; }
    std::uint32_t out_features() const noexcept { return out_features_; }
    bool cache_input() const noexcept override { return true;}

    std::span<ParamRef> params() noexcept override { return params_; }
    std::span<const ParamRef> params() const noexcept override { return params_; }

    void forward(tensor::TensorView<const float> in, tensor::TensorView<float> out) const override;
    void backward(tensor::TensorView<const float> input_grad, tensor::TensorView<const float> input_cache, tensor::TensorView<float> output_grad) override;

    // Initialization functions
    void init_zeros() noexcept;
    void init_xavier(std::uint32_t seed) noexcept;

private:
    std::uint32_t in_features_;
    std::uint32_t out_features_;

    // Ownership of weights and biases
    tensor::Tensor<float> W_;
    tensor::Tensor<float> B_;

    // Weight and Bias gradients for Optimizer
    tensor::Tensor<float> dW_;
    tensor::Tensor<float> dB_;

    std::array<ParamRef, 2> params_;
};

using DensePtr = std::unique_ptr<Dense>;
}