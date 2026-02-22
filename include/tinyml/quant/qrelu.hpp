#pragma once

#include "tinyml/quant/qlayer.hpp"
#include "tinyml/model/relu.hpp"

namespace tinyml::quant {
class QRelu final : public QLayer {
public:
    QRelu() = default;

    void forward(tensor::TensorView<const int8_t> in, tensor::TensorView<int8_t> out) const override;
    void observe_fp32_output(const tensor::TensorView<const float> out) override { (void)out; /* do nothing */ }
    QParam finalize_callibration(const model::Layer& layer, QParam input_param) override;
    bool callibrated() const noexcept override { return callibrated_; }; // quantized Relu  does not require quantization
private:
    bool callibrated_ = false;
    QParam param_;
};

}
