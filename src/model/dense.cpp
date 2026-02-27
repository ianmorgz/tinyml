#include "tinyml/model/dense.hpp"

#include <iostream>
#include <random>

#include "tinyml/core/shape.hpp"
#include "tinyml/core/config.hpp"

namespace tinyml::model {

Dense::Dense(const std::uint32_t in_features, const std::uint32_t out_features, const std::size_t alignment)
        : in_features_(in_features), out_features_(out_features),
        W_(core::Shape{out_features, in_features}, alignment),
        B_(core::Shape{out_features}, alignment),
        dW_(core::Shape{out_features, in_features}, alignment),
        dB_(core::Shape{out_features}, alignment)
{
    if (in_features_ == 0 || out_features_ == 0) { TINYML_EXCEPTION("A Dense layer cannot contain input or output size 0"); }

    init_xavier(12345); // 12345 is just a placeholder seed

    dW_.fill(0);
    dB_.fill(0);

    params_[0] = model::ParamRef{ .param = &W_, .grad = &dW_ };
    params_[1] = model::ParamRef{ .param = &B_, .grad = &dB_ };
};

core::Shape Dense::infer_output_shape(const core::Shape& in) const {
    // only Shape ranks [in_features] or [batch, in_features] allowed
    if (in.rank == 1) {
        if (in.dims[0] != in_features_) {
            TINYML_EXCEPTION("Dense output shape: features mismatch");
        }
        return core::Shape{out_features_};
    }

    if (in.rank == 2) {
        if (in.dims[1] != in_features_) {
            TINYML_EXCEPTION("Dense output shape: features mismatch");
        }
        return core::Shape{in.dims[0], out_features_};
    }

    TINYML_EXCEPTION("Dense output shape: wrong shape");
}

void Dense::forward(const tensor::TensorView<const float> in, const tensor::TensorView<float> out) const {
    if (in.rank() > 2) {
        TINYML_EXCEPTION("Dense forward: wrong input shape");
    }

    const std::size_t batch_size = (in.rank() == 1) ? 1 : static_cast<std::size_t>(in.shape()[0]);
    const auto in_f = static_cast<std::size_t>(in_features_);
    const auto out_f = static_cast<std::size_t>(out_features_);;

    // validate sizes
    const std::size_t expected_in = batch_size * in_f;
    const std::size_t expected_out = batch_size * out_f;

    if (in.size() != expected_in) { TINYML_EXCEPTION("Dense forward: wrong input size"); }
    if (out.size() != expected_out) { TINYML_EXCEPTION("Dense forward: wrong output size"); }

    const float* x = in.data();
    float* y = out.data();

    const float* W = W_.data();
    const float* b = B_.data();

    for (std::size_t n = 0; n < batch_size; ++n) {
        const float* x_row = x + n * in_f;
        float* y_row = y + n * out_f;

        for (std::size_t o = 0; o < out_f; ++o) {
            float acc = b[o];
            // dot product over in_F
            for (std::size_t i = 0; i < in_f; ++i) {
                acc += x_row[i] * W[o * in_f + i];
            }
            y_row[o] = acc;
        }
    }
}
// gradient is the previous layer's output (layer_index + 1 because we are looping backwards) [batch_size, output_size]
// input_cache is the input that was given during the forward training loop (output of layer - 1)

void Dense::backward(const tensor::TensorView<const float> input_grad, const tensor::TensorView<const float> input_cache, tensor::TensorView<float> output_grad) {
    if (input_grad.rank() != 2 || input_cache.rank() != 2 || output_grad.rank() != 2) {
        TINYML_EXCEPTION("Dense backward: incorrect argument tensor shape");
    }

    if (input_grad.shape()[0] == 0 || input_grad.shape()[1] != out_features_) {
        TINYML_EXCEPTION("Dense backward: incorrect input gradient size");
    }

    const auto batch_size = static_cast<std::size_t>(input_grad.shape()[0]);

    if (input_cache.shape()[0] != batch_size || input_cache.shape()[1] != in_features_) {
        TINYML_EXCEPTION("Dense backward: incorrect input cache size");
    }

    if (output_grad.shape()[0] != batch_size || output_grad.shape()[1] != in_features_) {
        TINYML_EXCEPTION("Dense backward: incorrect output gradient size");
    }

    // get all the pointers for our math
    const float* dY     = input_grad.data();
    const float* X      = input_cache.data();
    float* dX     = output_grad.data();
    const float* W      = W_.data();
    float* dW           = dW_.data();
    float* dB           = dB_.data();

    // ---- compute dB and dW ----
    for (std::size_t b = 0; b < batch_size; ++b) {
        const float* dYb = dY + b * out_features_;
        const float* Xb  = X  + b * in_features_;

        for (std::size_t o = 0; o < out_features_; ++o) {
            const float dy = dYb[o];
            dB[o] += dy;
            for (std::size_t i = 0; i < in_features_; ++i) {
                dW[o * in_features_ + i] += dy * Xb[i];
            }
        }
    }

    // ---- compute dX ----
    // dX[b,i] = sum_o dY[b,o] * W[o,i]
    for (std::size_t b = 0; b < batch_size; ++b) {
        const float* dYb = dY + b * out_features_;
        float* dXb       = dX + b * in_features_;

        // init dX row
        for (std::size_t i = 0; i < in_features_; ++i) dXb[i] = 0.0f;

        for (std::size_t o = 0; o < out_features_; ++o) {
            const float dy = dYb[o];

            for (std::size_t i = 0; i < in_features_; ++i) {
                dXb[i] += dy * W[o * in_features_ + i];
            }
        }
    }
}

void Dense::init_zeros() noexcept {
    W_.fill(0);
    B_.fill(0);
}

// https://www.geeksforgeeks.org/deep-learning/xavier-initialization/
void Dense::init_xavier(const std::uint32_t seed) noexcept {
    const auto denom = static_cast<float>(in_features_ + out_features_);
    const float a = std::sqrt(6.0f / denom);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> distr(-a, a);

    float* w = W_.data();

    for (std::size_t i = 0; i < W_.size(); ++i) {
        w[i] = distr(rng);
    }

    B_.fill(0);

};

}