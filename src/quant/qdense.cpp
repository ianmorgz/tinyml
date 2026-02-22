#include "tinyml/quant/qdense.hpp"
#include "tinyml/core/config.hpp"
#include "tinyml/model/sequential.hpp"
#include "tinyml/core/shape.hpp"

namespace tinyml::quant {

QDense::QDense(const model::Dense& d) : weights_(core::Shape{d.in_features(), d.out_features()}), w_scales_(core::Shape{d.out_features()}),
    biases_((core::Shape{d.out_features()})),
    in_features_(d.in_features()), out_features_(d.out_features()) {
        quantize_weights(d.weights());
}

void QDense::quantize_weights(const tensor::TensorView<const float>& fp32_weights) {
    if (fp32_weights.rank() != 2) {
        TINYML_EXCEPTION("Dense Layer Quantization: incorrect rank for fp32 weights");
    }
    // fp32_weights is [in_features_][out_features_]
    if (fp32_weights.shape()[0] != in_features_ || fp32_weights.shape()[1] != out_features_) {
        TINYML_EXCEPTION("Dense Layer Quantization: incorrect shape for fp32 weights");
    }

    const float* w_fp32 = fp32_weights.data();
    int8_t* w_i8 = weights_.data();      // in, out
    float* ws_fp32 = w_scales_.data();

    for (std::size_t o = 0; o < out_features_; ++o) {
        float abs_max = 0.0f;
        for (std::size_t i = 0; i < in_features_; ++i) {
            const float w = std::abs(w_fp32[i * out_features_ + o]);
            if (w > abs_max) abs_max = w;
        }

        const float s = (abs_max > 0.0f) ? (abs_max / 127.0f) : 1.0f;
        ws_fp32[o] = s;

        for (std::size_t i = 0; i < in_features_; ++i) {
            const float wf = w_fp32[i * out_features_ + o];
            const int32_t q = static_cast<int32_t>(std::lrintf(wf / s));
            w_i8[i * out_features_ + o] = QParam::clamp_int8(q);
        }
    }
}

// save minimum and maximum output to use as the layer's quantization bounds
void QDense::observe_fp32_output(const tensor::TensorView<const float> out) {
    if (out.rank() != 2) { TINYML_EXCEPTION("Quantized Dense observation float output tensor must be rank: 2"); }

    const float* o = out.data();
    const std::size_t n = out.size();

    for (std::size_t i = 0; i < n; i++) {
        if (o[i] > c_max) { c_max = o[i]; }
        if (o[i] < c_min) { c_min = o[i]; }
    }
}

QParam QDense::finalize_callibration(const model::Layer& layer, const QParam input_param) {
    if (layer.type() != model::LayerType::Dense) { TINYML_EXCEPTION("Quantized Dense FInalize Callibration float layer was not type dense"); }

    // set the params
    out_param_ = QParam(c_max, c_min, QType::Asymmetric);
    in_param_ = input_param;

    // fill out biases
    int32_t* b_i32 = biases_.data();
    const float* b_fp32 = static_cast<const model::Dense&>(layer).biases().data();
    const float* w_fp32 = w_scales_.data();

    for (std::size_t i = 0; i < out_features_; i++) {
        b_i32[i] = static_cast<int32_t>(
            std::round(b_fp32[i]/(in_param_.scale * w_fp32[i]))
        );
    }

    callibrated_ = true;
    return out_param_;
}

void QDense::forward(tensor::TensorView<const int8_t> in, tensor::TensorView<int8_t> out) const {
    tensor::Tensor<int32_t> acc(out.shape());

    const int8_t* w_i8 = weights_.data();
    const float* ws_fp32 = w_scales_.data();
    const int32_t* b_i32 = biases_.data();

    int32_t* a_i32 = acc.data();
    const int8_t* i_i8 = in.data();
    int8_t* o_i8 = out.data();

    const std::size_t n_o = out.size();
    const std::size_t n_i = in.size();

    if (n_o != out_features_) { TINYML_EXCEPTION("Quantized Dense forward input tensor incorectly sized"); }
    if (n_i != in_features_) { TINYML_EXCEPTION("Quantized Dense forward output tensor incorectly sized"); }

    for (std::size_t i = 0; i < n_o; i++) {
        a_i32[i] = 0;
        for (std::size_t j = 0; j < n_i; j++) {
            a_i32[i] += (i_i8[j] - in_param_.zero_point) * w_i8[i * n_o + j];
        }
        a_i32[i] += b_i32[i];

        const float M = (in_param_.scale * ws_fp32[i]) / out_param_.scale ;
        o_i8[i] = QParam::clamp_int8(std::round(a_i32[i] * M) + out_param_.zero_point);
    }
}

}
