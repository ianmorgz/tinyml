#include "tinyml/model/relu.hpp"

namespace tinyml::model {

void Relu::forward(const tensor::TensorView<const float> in, const tensor::TensorView<float> out) const {
    if (in.size() != out.size()) {
        TINYML_EXCEPTION("Relu output shape: features mismatch");
    }

    const float* x = in.data();
    float* y = out.data();

    const std::size_t n = in.size();
    for (std::size_t i = 0; i < n; ++i) {
        const float v = x[i];
        y[i] = (v > 0) ? v : 0;
    }
}

void Relu::backward(const tensor::TensorView<const float> grad_input, const tensor::TensorView<const float> cached, tensor::TensorView<float> grad_output) {
    if (grad_input.size() != grad_output.size() || grad_input.size() != cached.size()) {
        TINYML_EXCEPTION("Relu output shape: features mismatch");
    }

    const float* dY = grad_input.data();
    const float* Y = cached.data(); // output of forward
    float* dX = grad_output.data();

    const std::size_t n = grad_output.size();
    for (size_t i = 0; i < n; ++i) {
        dX[i] = Y[i] > 0.0f ? dY[i] : 0.0f;
    }
}

}