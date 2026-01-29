#pragma once
#include "tinyml/train/loss.hpp"

namespace tinyml::train {
class CrossEntropyLoss final : public Loss {
public:
    CrossEntropyLoss() = default;

    float forward(tensor::TensorView<const float> prediction, tensor::TensorView<const float> target) const override;
    void backward(tensor::TensorView<const float> prediction, tensor::TensorView<const float> target, tensor::TensorView<float> gradient) const override;
};
}