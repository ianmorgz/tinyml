#pragma once

#include <span>

#include "tinyml/tensor/tensor_view.hpp"
#include "tinyml/tensor/tensor.hpp"
#include "tinyml/core/shape.hpp"

namespace tinyml::model {

// Param Ref allows for the Optimizer to directly access and update the weights and biases
struct ParamRef final {
    tensor::Tensor<float>* param;   // e.g. weights and biases
    tensor::Tensor<float>* grad;    // e.g. weight and bias gradients

    tensor::TensorView<float> param_view() noexcept { return param->view(); }
    tensor::TensorView<const float> param_view() const noexcept { return param->view().as_const(); }

    tensor::TensorView<float> grad_view() noexcept { return grad->view(); }
    tensor::TensorView<const float> grad_view() const noexcept { return grad->view().as_const(); }
};

// LayerType for checking layers and testing purposes.
enum class LayerType : std::uint8_t {
    Dense,
    ReLu,
    Tanh,
};

// Virutal layer class to be used and stored inside the model.
// All dense, mathmatical, pooling, and convolutional layers should be able to use this type
class Layer {
public:
    // Constructor
    virtual ~Layer() = default;

    // Getters
    virtual LayerType type() const noexcept = 0;
    virtual bool cache_input() const noexcept = 0;

    virtual std::span<ParamRef> params() noexcept = 0;
    virtual std::span<const ParamRef> params() const noexcept = 0;

    // Inference functions
    virtual core::Shape infer_output_shape(const core::Shape& in) const = 0;
    virtual void forward(tensor::TensorView<const float> in, tensor::TensorView<float> out) const = 0;
    virtual void backward(tensor::TensorView<const float> out_gradient, tensor::TensorView<const float> cached, tensor::TensorView<float> in_gradient) = 0;
};

// When creating a layer class unique_ptr must alwasy be used
using LayerPtr = std::unique_ptr<Layer>;

}
