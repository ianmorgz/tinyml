#include "tinyml/core/config.hpp"
#include "tinyml/train/context.hpp"

namespace tinyml::core {
void Context::save_cache(const std::size_t layer_index, const tensor::TensorView<const float> &in) {
    if (layer_index >= cache_.size()) {
        cache_.resize(layer_index + 1);
    }

    auto& layer = cache_[layer_index];

    // resize the tensor if needed
    if (layer.shape() != in.shape()) {
        layer = tensor::Tensor<float>(in.shape());
    }

    // copies the tensor's values to avoid value corruption
    const float* src = in.data();
    float* dst = layer.data();
    for (std::size_t i = 0; i < in.size(); ++i) {
        dst[i] = src[i];
    }
}

tensor::TensorView<const float> Context::get_cache(const std::size_t layer_index) const {
    if (layer_index >= cache_.size()) {
        TINYML_EXCEPTION("Training Context: layer_index out of range");
    }

    const auto& layer = cache_[layer_index];
    return {layer.data(), layer.shape()};
}
}