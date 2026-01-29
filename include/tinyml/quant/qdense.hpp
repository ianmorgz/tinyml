#pragma once
#include "tinyml/quant/qlayer.hpp"
#include "tinyml/model/dense.hpp"
#include "tinyml/core/shape.hpp"
#include "tinyml/quant/qparam.hpp"

namespace tinyml::quant {
class QDense final: public QLayer {
public:
    QDense() = default;
    explicit QDense(const model::Dense& d);

    tensor::TensorView<const std::int8_t> weights() const { return weights_.view().as_const(); }
    tensor::TensorView<const std::int32_t> biases() const { return biases_.view().as_const(); }
    tensor::TensorView<const float> weight_scales() const { return w_scales_.view().as_const(); }

    // qlayer overrides
    void forward(tensor::TensorView<const int8_t> in, tensor::TensorView<int8_t> out) const override;
    void observe_fp32_output(tensor::TensorView<const float> out) override;
    void finalize_callibration(const model::Layer& layer) override;
    bool callibrated() const noexcept override { return callibrated_; };

private:
    bool callibrated_ = false;
    std::size_t in_ = 0;
    std::size_t out_ = 0;
    QParam out_param_;

    tensor::Tensor<std::int8_t> weights_;
    tensor::Tensor<float> w_scales_;
    tensor::Tensor<std::int32_t> biases_;

    // helper functions
    void quantize_weights(const tensor::TensorView<const float>& fp32_weights);

    // calibration param methods
    float c_min = 0.0f;
    float c_max = 0.0f;
};
}