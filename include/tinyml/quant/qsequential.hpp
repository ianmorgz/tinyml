#pragma once

#include "tinyml/model/sequential.hpp"
#include "tinyml/dataset/dataset.hpp"
#include "tinyml/quant/qlayer.hpp"

namespace tinyml::quant {
class QSequential final {
public:
    QSequential() = default;
    QSequential(const model::Sequential& net, const dataset::Dataset& ds, const std::size_t callibration_size);

private:
    // helper functions
    void make_layers(const model::Sequential& net);
    void callibrate(model::Sequential &net, dataset::Dataset &ds, std::size_t callibration_size);

    std::vector<std::unique_ptr<QLayer>> layers_;
};
}