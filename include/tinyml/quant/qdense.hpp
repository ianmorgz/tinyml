#pragma once
#include "tinyml/quant/qlayer.hpp"
#include "tinyml/model/dense.hpp"

namespace tinyml::quant {
class QDense final: public QLayer {
public:
    explicit QDense(const model::Dense& d);

    // getters
    tensor::TensorView<const std::int8_t> weights() const { return weights_.view().as_const(); }
    tensor::TensorView<const std::int32_t> biases() const { return biases_.view().as_const(); }
    tensor::TensorView<const float> weight_scales() const { return w_scales_.view().as_const(); }

    // qlayer overrides
    void forward(tensor::TensorView<const int8_t> in, tensor::TensorView<int8_t> out) const override;
    void observe_fp32_output(tensor::TensorView<const float> out) override;
    QParam finalize_calibration(const model::Layer& layer, const QParam input_param) override;
    bool calibrated() const noexcept override { return calibrated_; };

private:
    bool calibrated_ = false;

    tensor::Tensor<std::int8_t> weights_;
    tensor::Tensor<float> w_scales_;
    tensor::Tensor<std::int32_t> biases_;

    std::size_t in_features_ = 0;
    std::size_t out_features_ = 0;

    QParam out_param_;
    QParam in_param_;

    // helper function
    void quantize_weights(const tensor::TensorView<const float>& fp32_weights);

    // calibration params
    float c_min = std::numeric_limits<float>::infinity();
    float c_max = -std::numeric_limits<float>::infinity();
};
}