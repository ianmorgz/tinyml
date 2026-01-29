#pragma once
#include "tinyml/train/context.hpp"

namespace tinyml::train {
class Loss {
public:
    virtual ~Loss() = default;
    virtual float forward(tensor::TensorView<const float> prediction, tensor::TensorView<const float> target) const = 0;
    virtual void backward(tensor::TensorView<const float> prediction, tensor::TensorView<const float> target, tensor::TensorView<float> gradient) const = 0;
};
}