#pragma once
#include "tinyml/tensor/tensor.hpp"
#include <vector>

namespace tinyml::core {

class Context final {
public:
    Context() = default;

    void resize(const std::size_t num_layers) { cache_.resize(num_layers); }
    std::size_t size() const noexcept { return cache_.size(); }

    void save_cache(std::size_t layer_index, const tensor::TensorView<const float> &in);     // save input tensor view to cache (forward)
    tensor::TensorView<const float> get_cache(size_t layer_index) const; // retrieve input tensor view from cache (backward)

private:
    std::vector<tensor::Tensor<float>> cache_;
};
}
