#include "tinyml/model/tanh.hpp"
#include <cmath> // std::tanh

namespace tinyml::model {

void Tanh::forward(const tensor::TensorView<const float> in,
                   const tensor::TensorView<float> out) const {
    if (in.size() != out.size()) {
        TINYML_EXCEPTION("Tanh forward: input/output size mismatch");
    }

    const float* x = in.data();
    float* y = out.data();

    const std::size_t n = in.size();
    for (std::size_t i = 0; i < n; ++i) {
        y[i] = std::tanh(x[i]);
    }
}

void Tanh::backward(const tensor::TensorView<const float> grad_input,
                    const tensor::TensorView<const float> cached,
                    tensor::TensorView<float> grad_output) {
    if (grad_input.size() != grad_output.size() || grad_input.size() != cached.size()) {
        TINYML_EXCEPTION("Tanh backward: size mismatch");
    }

    const float* dY = grad_input.data();
    const float* Y  = cached.data();   // FIX: use cached input, not grad_input
    float* dX       = grad_output.data();

    const std::size_t n = grad_output.size();
    for (std::size_t i = 0; i < n; ++i) {
        const float y = Y[i];
        dX[i] = dY[i] * (1.0f - y * y);
    }
}

} // namespace tinyml::model
