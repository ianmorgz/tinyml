#include "tinyml/dataset/dataset.hpp"
#include "tinyml/dataset/loader.hpp"
#include "tinyml/core/config.hpp"
#include "tinyml/train/loss.hpp"

#include <random>
#include <algorithm>

namespace tinyml::dataset {

Dataset::Dataset(std::unique_ptr<Loader> loader, float training_split) {
    if (!loader) { TINYML_EXCEPTION("Dataset loader must not be null"); }
    if (training_split < 0.0f) { training_split = 0.0f; }
    if (training_split > 1.0f) { training_split = 1.0f; }

    loader_ = std::move(loader);

    Batch temp;
    loader_->load(temp);

    if (temp.input.rank() != 2) { TINYML_EXCEPTION("Dataset input rank must be 2"); }
    if (temp.label.rank() != 2) { TINYML_EXCEPTION("Dataset label rank must be 2"); }

    if (temp.size == 0) { TINYML_EXCEPTION("Dataset input is empty"); }

    if (temp.size != temp.input.shape()[0] || temp.size != temp.label.shape()[0]) {
        TINYML_EXCEPTION("Dataset loader does not match its own size");
    }

    const std::size_t batch_size = temp.size;
    const auto training_batch_size = static_cast<std::size_t>(static_cast<float>(batch_size) * training_split);
    const std::size_t testing_batch_size = batch_size - training_batch_size;

    training_order_.resize(training_batch_size);

    const std::size_t input_size = temp.input.shape()[1];
    const std::size_t label_size = temp.label.shape()[1];

    training_.size = training_batch_size;
    testing_.size = testing_batch_size;

    training_.input = tensor::Tensor<float>(core::Shape{static_cast<uint32_t>(training_batch_size), static_cast<uint32_t>(input_size)});
    training_.label = tensor::Tensor<float>(core::Shape{static_cast<uint32_t>(training_batch_size), static_cast<uint32_t>(label_size)});

    testing_.input = tensor::Tensor<float>(core::Shape{static_cast<uint32_t>(testing_batch_size), static_cast<uint32_t>(input_size)});
    testing_.label = tensor::Tensor<float>(core::Shape{static_cast<uint32_t>(testing_batch_size), static_cast<uint32_t>(label_size)});

    const float* tmp_in = temp.input.data();
    const float* tmp_lbl = temp.label.data();

    // Fill out training data
    float* tr_in = training_.input.data();
    float* tr_lbl = training_.label.data();

    for (std::size_t i = 0; i < training_batch_size; ++i) {
        training_order_[i] = i;
        for (std::size_t j = 0; j < input_size; ++j) {
            tr_in[i * input_size + j] = tmp_in[i * input_size + j];
        }
        for (std::size_t j = 0; j < label_size; ++j) {
            tr_lbl[i * label_size + j] = tmp_lbl[i * label_size + j];
        }
    }

    // Fill out testing data
    float* tst_in = testing_.input.data();
    float* tst_lbl = testing_.label.data();

    for (std::size_t i = training_batch_size; i < batch_size; ++i) {
        const std::size_t n = i - training_batch_size; // looping 0 -> testing_batch_size
        for (std::size_t j = 0; j < input_size; ++j) {
            tst_in[n * input_size + j] = tmp_in[i * input_size + j];
        }
        for (std::size_t j = 0; j < label_size; ++j) {
            tst_lbl[n * label_size + j] = tmp_lbl[i * label_size + j];
        }
    }
}

void Dataset::shuffle_training(const std::uint32_t seed) {
    std::mt19937 rng(seed);
    std::ranges::shuffle(training_order_, rng);
    training_cursor_ = 0;
}

bool Dataset::next_training_batch(std::size_t batch_size, BatchView& out) {
    if (batch_size == 0) { return false; }
    if (training_cursor_ >= training_order_.size()) { return false; }

    if (batch_size > training_.size) { batch_size = training_.size; }

    const std::size_t remaining = training_.size - training_cursor_;
    batch_size = (remaining >= batch_size) ? batch_size : remaining;

    const std::size_t in_dim  = training_.input.shape()[1];
    const std::size_t out_dim = training_.label.shape()[1];

    if (training_batch_.size != batch_size) {
        training_batch_.size = batch_size;
        training_batch_.input = tensor::Tensor<float>(core::Shape{static_cast<uint32_t>(batch_size), static_cast<uint32_t>(in_dim)});
        training_batch_.label = tensor::Tensor<float>(core::Shape{static_cast<uint32_t>(batch_size), static_cast<uint32_t>(out_dim)});
    }

    const float* src_input = training_.input.data();
    const float* src_label = training_.label.data();
    float* batch_input = training_batch_.input.data();
    float* batch_label = training_batch_.label.data();

    for (std::size_t i = 0; i < batch_size; ++i) {
        const std::size_t src_row = training_order_[training_cursor_ + i];
        for (std::size_t j = 0; j < in_dim; ++j) { batch_input[i * in_dim + j] = src_input[src_row * in_dim + j]; }
        for (std::size_t j = 0; j < out_dim; ++j) { batch_label[i * out_dim + j] = src_label[src_row * out_dim + j]; }
    }

    training_cursor_ += batch_size;
    out = BatchView(training_batch_);
    return true;
}

bool Dataset::next_testing_batch(std::size_t batch_size, BatchView& out) {
    if (batch_size == 0) { return false; }
    if (testing_cursor_ >= testing().label.shape()[0]) { return false; }

    if (batch_size > testing_.size) { batch_size = testing_.size; }

    const std::size_t remaining = testing_.size - testing_cursor_;
    batch_size = (remaining >= batch_size) ? batch_size : remaining;

    const std::size_t in_dim  = testing_.input.shape()[1];
    const std::size_t out_dim = testing_.label.shape()[1];

    if (testing_batch_.size != batch_size) {
        testing_batch_.size = batch_size;
        testing_batch_.input = tensor::Tensor<float>(core::Shape{static_cast<uint32_t>(batch_size), static_cast<uint32_t>(in_dim)});
        testing_batch_.label = tensor::Tensor<float>(core::Shape{static_cast<uint32_t>(batch_size), static_cast<uint32_t>(out_dim)});
    }

    const float* src_input = testing_.input.data();
    const float* src_label = testing_.label.data();
    float* batch_input = testing_batch_.input.data();
    float* batch_label = testing_batch_.label.data();

    for (std::size_t i = 0; i < batch_size; ++i) {
        const std::size_t src_row = testing_cursor_ + i;
        for (std::size_t j = 0; j < in_dim; ++j) { batch_input[i * in_dim + j] = src_input[src_row * in_dim + j]; }
        for (std::size_t j = 0; j < out_dim; ++j) { batch_label[i * out_dim + j] = src_label[src_row * out_dim + j]; }
    }

    testing_cursor_ += batch_size;
    out = BatchView(testing_batch_);
    return true;
}
}
