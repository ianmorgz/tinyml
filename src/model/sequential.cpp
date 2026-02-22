#include "tinyml/model/sequential.hpp"
#include "tinyml/core/config.hpp"
#include "tinyml/model/dense.hpp"
#include "tinyml/model/relu.hpp"
#include "tinyml/model/tanh.hpp"

namespace tinyml::model {
void Sequential::add(std::unique_ptr<Layer> layer) {
    if (!layer) { TINYML_EXCEPTION("Sequential model cannot add null layer"); }
    if (built_) { TINYML_EXCEPTION("Sequential model cannot add layer if built"); }
    layers_.push_back(std::move(layer));
}

// help
void Sequential::throw_if_not_built() const {
    if (!built_) {TINYML_EXCEPTION("Sequential model has not been built"); }
    if (layers_.empty()) {TINYML_EXCEPTION("Sequential model is empty"); }
}

void Sequential::build(const std::size_t input_size, const std::size_t max_batch) {
    if (layers_.empty()) { TINYML_EXCEPTION("Sequential model is empty"); }

    layer_sizes_.clear();
    model_params_.clear();
    layer_sizes_.reserve(layers_.size() + 1);
    layer_sizes_.push_back(input_size);

    core::Shape current_shape({static_cast<std::uint32_t>(max_batch), static_cast<std::uint32_t>(input_size)});
    std::size_t max_layer_elem = current_shape.flat_size();

    for (const auto& layer : layers_) {
        current_shape = layer->infer_output_shape(current_shape);
        if (current_shape.rank != 2) { TINYML_EXCEPTION("Model build: layer outputed incorrectly sized shape"); }
        if (const size_t current_shape_size = current_shape.flat_size(); current_shape_size > max_layer_elem) { max_layer_elem = current_shape_size; }
        layer_sizes_.push_back(current_shape[1]);

        // get and store the layer params
        auto lp = layer->params(); // span into layer-owned array
        model_params_.insert(model_params_.end(), lp.begin(), lp.end());
    }

    input_features_ = layer_sizes_[0];
    output_features_ = current_shape[1];
    max_features_ = max_layer_elem;
    max_batch_size_ = max_batch;

    // Initialzie activation tensors ( treated soley as raw buffers so shape doesn't matter only size)
    act_a_ = tensor::Tensor<float>(core::Shape{static_cast<std::uint32_t>(max_layer_elem)});
    act_b_ = tensor::Tensor<float>(core::Shape{static_cast<std::uint32_t>(max_layer_elem)});

    built_ = true;
}

// forward input through the model layer by layer
tensor::TensorView<const float> Sequential::forward(const tensor::TensorView<const float> &in) {
    throw_if_not_built();
    if (in.shape().rank != 2) {
        TINYML_EXCEPTION("Sequential model forward given invalid shape");
    }
    if (in.shape()[0] > max_batch_size_ || in.shape()[1] != input_features_) {
        TINYML_EXCEPTION("Sequential model forward given invalid size");
    }

    bool write_to_a = true;
    tensor::TensorView<const float> in_view = in;
    const std::size_t batch_size = in.shape()[0];


    for (std::size_t i = 0; i < layers_.size(); ++i) {
        core::Shape out_shape({static_cast<std::uint32_t>(batch_size), static_cast<std::uint32_t>(layer_sizes_[i+1])});

        tensor::TensorView<float> out_view = write_to_a
        ? tensor::TensorView(act_a_.data(), out_shape)
        : tensor::TensorView(act_b_.data(), out_shape);

        layers_[i]->forward(in_view, out_view);
        in_view = out_view.as_const();
        write_to_a = !write_to_a;
    }

    return in_view;
}

// Forward train an entire batch
tensor::TensorView<const float> Sequential::forward_train(const tensor::TensorView<const float> &in, core::Context& ctx) {
    throw_if_not_built();
    if (in.shape().rank != 2) { TINYML_EXCEPTION("Sequential model forward given invalid shape"); }
    if (in.shape()[0] > max_batch_size_ || in.shape()[1] != input_features_) { TINYML_EXCEPTION("Sequential model forward given invalid size"); }

    bool write_to_a = true;
    tensor::TensorView<const float> in_view = in;

    const std::size_t batch_size = in.shape()[0];
    for (std::size_t i = 0; i < layers_.size(); ++i) {
        core::Shape out_shape({static_cast<std::uint32_t>(batch_size), static_cast<std::uint32_t>(layer_sizes_[i+1])});

        tensor::TensorView<float> out_view =
            (write_to_a ? tensor::TensorView(act_a_.data(), out_shape) : tensor::TensorView(act_b_.data(), out_shape));

        layers_[i]->forward(in_view, out_view);

        if (layers_[i]->cache_input()) { ctx.save_cache(i, in_view); } else {
            ctx.save_cache(i, out_view.as_const());
        }

        in_view = out_view.as_const();
        write_to_a = !write_to_a;
    }

    return in_view;
}

void Sequential::backwards_train(const tensor::TensorView<const float>& output_gradient, const core::Context& ctx) {
    throw_if_not_built();

    if (output_gradient.shape().rank != 2) {
        TINYML_EXCEPTION("Sequential model backwards given invalid shape");
    }
    if (output_gradient.shape()[0] > max_batch_size_ ||
        output_gradient.shape()[1] != output_features_) {
        TINYML_EXCEPTION("Sequential model backwards given invalid size");
        }

    bool write_to_a = true;
    tensor::TensorView<const float> out_view = output_gradient;
    const auto batch_size = static_cast<std::size_t>(output_gradient.shape()[0]);

    // iterate layers in reverse safely
    for (std::size_t i = layers_.size(); i-- > 0; ) {
        core::Shape in_shape({
            static_cast<std::uint32_t>(batch_size),
            static_cast<std::uint32_t>(layer_sizes_[i])
        });

        const tensor::TensorView<const float> layer_input = ctx.get_cache(i);
        if (layer_input.shape() != in_shape) {
            TINYML_EXCEPTION("Sequential model backwards: incorrect saved input shape");
        }

        tensor::TensorView<float> input_gradient = write_to_a
            ? tensor::TensorView<float>(act_a_.data(), in_shape)
            : tensor::TensorView<float>(act_b_.data(), in_shape);

        layers_[i]->backward(out_view, layer_input, input_gradient);

        out_view = input_gradient.as_const();
        write_to_a = !write_to_a;
    }
}

void Sequential::clear_param_gradients() const {
    for (auto& param : params()) {
        float* g = param.grad->view().data();
        const std::size_t n = param.grad->size();;
        for (std::size_t i = 0; i < n; ++i) { g[i] = 0; }
    }
}

Sequential& Sequential::dense(std::size_t in_features, std::size_t out_features) {
    layers_.push_back(std::make_unique<Dense>(in_features, out_features));
    return *this;
}

Sequential& Sequential::relu() {
    layers_.push_back(std::make_unique<Relu>());
    return *this;
}

Sequential& Sequential::tanh() {
    layers_.push_back(std::make_unique<Tanh>());
    return *this;
}
}
