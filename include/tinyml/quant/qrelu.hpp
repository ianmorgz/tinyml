#pragma once

#include "tinyml/quant/qlayer.hpp"
#include "tinyml/model/relu.hpp"

namespace tinyml::quant {
class QRelu final : public QLayer {
public:
    QRelu() = default;

    void forward(tensor::TensorView<const int8_t> in, tensor::TensorView<int8_t> out) const override;
    void observe_fp32_output(const tensor::TensorView<const float> out) override { (void)out; /* do nothing */ }
    QParam finalize_calibration(const model::Layer& layer, QParam input_param) override;
    bool calibrated() const noexcept override { return calibrated_; }; // quantized Relu  does not require quantization
    QLayer_Type type() const noexcept override { return QLayer_Type::QReLu; }

    // getter functions
    const QParam* param() const noexcept { return &param_; }
private:
    bool calibrated_ = false;
    QParam param_;
};

}
