#include "tinyml/train/ce_loss.hpp"
#include <cmath>

#include <iostream>
#include <ostream>

namespace tinyml::train {

float CrossEntropyLoss::forward(const tensor::TensorView<const float> logits,
                                const tensor::TensorView<const float> target) const {
    if (logits.rank() != 2 || target.rank() != 2) {
        TINYML_EXCEPTION("SoftmaxCE forward: incorrect rank for input tensors");
    }
    if (logits.shape() != target.shape()) {
        TINYML_EXCEPTION("SoftmaxCE forward: logits shape does not match target shape");
    }

    const std::size_t B = logits.shape()[0];
    const std::size_t C = logits.shape()[1];
    if (B == 0 || C == 0) {
        TINYML_EXCEPTION("SoftmaxCE forward: batch size is zero");
    }

    const float* z = logits.data();
    const float* y = target.data();

    float loss_acc = 0.0f;

    for (std::size_t i = 0; i < B; ++i) {
        const std::size_t row = i * C;

        float max_z = z[row];
        for (std::size_t j = 1; j < C; ++j) {
            const float v = z[row + j];
            if (v > max_z) max_z = v;
        }

        float sum_exp = 0.0f;
        for (std::size_t j = 0; j < C; ++j) {
            sum_exp += std::exp(z[row + j] - max_z);
        }
        const float logsumexp = max_z + std::log(sum_exp);

        float yz = 0.0f;
        for (std::size_t j = 0; j < C; ++j) {
            yz += y[row + j] * z[row + j];
        }

        loss_acc += (logsumexp - yz);
    }

    return loss_acc / static_cast<float>(B);
}

void CrossEntropyLoss::backward(const tensor::TensorView<const float> logits,
                                const tensor::TensorView<const float> target,
                                const tensor::TensorView<float> gradient) const {
    if (logits.rank() != 2 || target.rank() != 2 || gradient.rank() != 2) {
        TINYML_EXCEPTION("SoftmaxCE backward: incorrect rank for input tensors");
    }
    if (logits.shape() != target.shape() || logits.shape() != gradient.shape()) {
        std::cout << "logits: " << logits.size() << std::endl;
        std::cout << "target: " << target.size() << std::endl;
        std::cout << "gradient: " << gradient.size() << std::endl;
        TINYML_EXCEPTION("SoftmaxCE backward: shape mismatch for input tensors");
    }

    const std::size_t B = logits.shape()[0];
    const std::size_t C = logits.shape()[1];
    if (B == 0 || C == 0) {
        TINYML_EXCEPTION("SoftmaxCE backward: batch size is zero");
    }

    const float* z = logits.data();
    const float* y = target.data();
    float* dz = gradient.data();

    const float invB = 1.0f / static_cast<float>(B);

    for (std::size_t i = 0; i < B; ++i) {
        const std::size_t row = i * C;

        float max_z = z[row];
        for (std::size_t j = 1; j < C; ++j) {
            const float v = z[row + j];
            if (v > max_z) max_z = v;
        }

        float denom = 0.0f;
        for (std::size_t j = 0; j < C; ++j) {
            denom += std::exp(z[row + j] - max_z);
        }
        const float inv_denom = 1.0f / denom;

        for (std::size_t j = 0; j < C; ++j) {
            const float p = std::exp(z[row + j] - max_z) * inv_denom; // softmax
            dz[row + j] = (p - y[row + j]) * invB;
        }
    }
}

} // namespace tinyml::train