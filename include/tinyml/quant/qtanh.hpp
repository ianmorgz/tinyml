#pragma once
#include "tinyml/quant/qlayer.hpp"
#include "tinyml/model/tanh.hpp"

namespace tinyml::quant {
class QTanh final : public QLayer {
public:
    QTanh() = default;

    void forward(tensor::TensorView<const int8_t> in, tensor::TensorView<int8_t> out) const override;
    void observe_fp32_output(tensor::TensorView<const float> out) override { /* do nothing */ }
    void finalize_callibration(const model::Layer& layer) override;// build and save the LUT
    bool callibrated() const noexcept override { return callibrated_; };

private:
    tensor::Tensor<int8_t> LUT;
    bool callibrated_;
};

}
