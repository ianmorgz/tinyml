#pragma once

#include "tinyml/model/sequential.hpp"
#include "tinyml/dataset/dataset.hpp"

namespace tinyml::train {

enum class Optimizer : std::uint8_t {
    SGD,
};

enum class LossFunction : std::uint8_t {
    CrossEntropy,
};

struct Config {
    std::size_t epochs{};
    std::size_t batch_size{};
    float learning_rate{};
    Optimizer optimizer = Optimizer::SGD;
    LossFunction loss_function = LossFunction::CrossEntropy;
};

void fit(model::Sequential& model, dataset::Dataset& dataset, const Config &config);
inline std::size_t argmax(const float* x, std::size_t n);
inline bool check_class(const float* logits_row, const float* onehot_row, std::size_t classes);
inline void print_progress(std::size_t epoch, std::size_t epochs, std::size_t step, std::size_t steps, float avg_loss);

}