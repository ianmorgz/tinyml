#pragma once

#include "tinyml/dataset/batch.hpp"
#include "tinyml/dataset/loader.hpp"
#include <vector>

namespace tinyml::dataset {

class Dataset final {
public:
    Dataset() = default;
    Dataset(std::unique_ptr<Loader> loader, float training_split);

    // Temporary debugging feature
    BatchView training() const noexcept { return BatchView(training_); }
    BatchView testing() const noexcept { return BatchView(testing_); }

    void shuffle_training(std::uint32_t seed);
    bool next_training_batch(std::size_t batch_size, BatchView& out);
    bool next_testing_batch(std::size_t batch_size, BatchView& out);

private:
    Batch training_;
    Batch testing_;

    std::vector<std::size_t> training_order_;
    std::size_t training_cursor_ = 0;
    Batch training_batch_;

    std::size_t testing_cursor_ = 0;
    Batch testing_batch_;

    std::unique_ptr<Loader> loader_;
};

}