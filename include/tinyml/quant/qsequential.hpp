#pragma once

#include "tinyml/model/sequential.hpp"
#include "tinyml/dataset/dataset.hpp"
#include "tinyml/quant/qlayer.hpp"
#include "tinyml/quant/qparam.hpp"

namespace tinyml::quant {
class QSequential final {
public:
    QSequential() = default;

    QSequential(model::Sequential& net, dataset::Dataset& ds, std::size_t callibration_size);

    tensor::TensorView<const float> forward(const tensor::TensorView<const float> &in) const;

private:
    std::vector<std::unique_ptr<QLayer>> layers_;
    std::size_t num_layers;
    std::vector<std::size_t> layer_sizes_;
    std::size_t max_layer_size_;

    // 0th index is the input for the model, resulting in sizeof layers_size + 1
    QParam input_param_;
    QParam output_param_;

    float c_max;
    float c_min;

    // helper functions
    void make_layers(const model::Sequential& net);
    void callibrate(model::Sequential &net, dataset::Dataset &ds, const std::size_t callibration_size);
    void finalize(model::Sequential &net);
};
}