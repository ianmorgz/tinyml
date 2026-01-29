#pragma once

#include "tinyml/quant/qlayer.hpp"
#include "tinyml/model/relu.hpp"

namespace tinyml::quant {
class QRelu final : public QLayer {
public:
    QRelu() = default;

    void forward(tensor::TensorView<const int8_t> in, tensor::TensorView<int8_t> out) const override;
    void observe_fp32_output(tensor::TensorView<const float> out) override { /* do nothing */ }
    void finalize_callibration(const model::Layer& layer) override { /* do nothing */ }
    bool callibrated() const noexcept override { return true; }; // The Quantized Relu layer does not require quantization
};

}
