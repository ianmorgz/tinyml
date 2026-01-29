#include "tinyml/quant/qdense.hpp"
#include "tinyml/core/config.hpp"
#include "tinyml/model/sequential.hpp"
#include "tinyml/core/shape.hpp"

namespace tinyml::quant {

QDense::QDense(const model::Dense& d) : in_(d.in_features()), out_(d.out_features()),
    weights_(core::Shape{static_cast<std::uint32_t>(out_), static_cast<std::uint32_t>(in_)}),
    w_scales_(core::Shape{static_cast<std::uint32_t>(out_)}), biases_((core::Shape{static_cast<std::uint32_t>(out_)})) {
        quantize_weights(d.weights());
}

void QDense::quantize_weights(const tensor::TensorView<const float>& fp32_weights) {
    if (fp32_weights.rank() != 2) { TINYML_EXCEPTION("Dense Layer Quantization: incorrect rank for fp32 weights"); }
    if (fp32_weights.shape()[0] != out_ || fp32_weights.shape()[1] != in_) { TINYML_EXCEPTION("Dense Layer Quantization: incorrect shape for fp32 weights"); }

    const float* w_fp32 = fp32_weights.data();
    int8_t* w_i8 = weights_.data();
    float* ws_fp32 = w_scales_.data();

    for (std::size_t i = 0; i < out_; i++) {
        // symetric signed quantize per channel and set weight and scale
        float abs_max = 0;
        for (std::size_t j = 0; j < in_; j++) {
            const float w = std::abs(w_fp32[i * in_ + j]);
            if (w > abs_max) { abs_max = w; }
        }

        float s = (abs_max > 0.0f) ? (abs_max / 127.0f) : 1.0f;
        ws_fp32[i] = s;

        for (std::size_t j = 0; j < in_; j++) {
            w_i8[i * in_ + j] = clamp_int8(w_fp32[i * in_ + j] / s);
        }
    }
}

void QDense::observe_fp32_output(const tensor::TensorView<const float> out) {
    if (out.rank() != 2) { TINYML_EXCEPTION("Quantized Dense observation float output tensor must be rank: 2"); }
    const float* o = out.data();
    const std::size_t n = out.size();

    for (std::size_t i = 0; i < n; i++) {
        if (o[i] > c_max) { c_max = o[i]; }
        if (o[i] < c_min) { c_min = o[i]; }
    }
}

void QDense::finalize_callibration(const model::Layer& layer) {
    if (layer.type() != model::LayerType::Dense) { TINYML_EXCEPTION("Quantized Dense FInalize Callibration float layer was not type dense"); }
    tensor::TensorView<const float> fp32_biases = static_cast<const model::Dense&>(layer).biases();

    // set the output param
    out_param_ = QParam(c_max, c_min, QType::Asymmetric);

    // quantize biases


    callibrated_ = true;

}

}
