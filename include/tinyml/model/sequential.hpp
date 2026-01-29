#pragma once

#include "layer.hpp"
#include "tinyml/tensor/tensor.hpp"
#include "../train/context.hpp"

namespace tinyml::model {
class Sequential final {
public:
    Sequential() = default;

    // seuqneital model creation functions
    void add(std::unique_ptr<Layer> layer);
    void build(std::size_t input_size, std::size_t max_batch = 1);

    Sequential& dense(std::size_t in_features, std::size_t out_features);
    Sequential& relu();
    Sequential& tanh();

    // getter functions
    std::size_t num_layers() const noexcept { return layers_.size(); }
    const Layer& layer(const std::size_t i) const { return *layers_.at(i); }
    Layer& layer(const std::size_t i) { return *layers_.at(i); }

    bool is_built() const noexcept { return built_; }

    std::size_t input_features() const noexcept { return input_features_; };
    std::size_t output_features() const noexcept { return output_features_; };
    std::size_t max_features() const noexcept { return max_features_; };
    std::size_t max_batch() const noexcept { return max_batch_size_; };
    std::size_t layer_features(const std::size_t i) const noexcept {return layer_sizes.at(i); }
    const std::vector<ParamRef>& params() const noexcept { return model_params_; };

    void clear_param_gradients() const;

    tensor::TensorView<const float> forward(const tensor::TensorView<const float> &in);
    tensor::TensorView<const float> forward_train(const tensor::TensorView<const float> &in, core::Context& ctx);
    void backwards_train(const tensor::TensorView<const float> &output_gradient, const core::Context& ctx);

    // TODO save model and load model
private:
    std::vector<std::unique_ptr<Layer>> layers_;

    // layer metadata
    std::vector<ParamRef> model_params_;
    std::vector<std::size_t> layer_sizes;
    std::size_t input_features_ = 0;
    std::size_t output_features_ = 0;
    std::size_t max_features_;
    std::size_t max_batch_size_ = 0;
    bool built_ = false;

    // swapping activation buffers
    tensor::Tensor<float> act_a_;
    tensor::Tensor<float> act_b_;

    // internal helper function
    void throw_if_not_built() const;

};
}
