#pragma once
#include "tinyml/dataset/loader.hpp"
#include "tinyml/dataset/batch.hpp"
#include "tinyml/core/config.hpp"

#include <string>
#include <fstream>
#include <vector>

using namespace tinyml;

class XOR_loader : public dataset::Loader {
public:
    XOR_loader(std::size_t batch_iterations) : batch_iterations_(batch_iterations) {};

    void load(dataset::Batch& out) const override {
        const std::size_t batch_size = 8 * batch_iterations_;
        constexpr std::size_t input_size = 2;
        constexpr std::size_t label_size = 2;

        out.size = batch_size;

        out.input = tinyml::tensor::Tensor<float>(core::Shape {batch_size, input_size});
        out.label = tinyml::tensor::Tensor<float>(core::Shape {batch_size, label_size});

        float* i = out.input.data();
        float* l = out.label.data();

        for (std::size_t j = 0; j < batch_size; j += 8) {
            i[j + 0] = 0.0f;
            i[j + 1] = 0.0f;

            l[j + 0] = 1.0f;
            l[j + 1] = 0.0f;

            i[j + 2] = 1.0f;
            i[j + 3] = 0.0f;

            l[j + 2] = 0.0f;
            l[j + 3] = 1.0f;

            i[j + 4] = 0.0f;
            i[j + 5] = 1.0f;

            l[j + 4] = 0.0f;
            l[j + 5] = 1.0f;

            i[j + 6] = 1.0f;
            i[j + 7] = 1.0f;

            l[j + 6] = 1.0f;
            l[j + 7] = 0.0f;
        }
    }

private:
    std::size_t batch_iterations_;
};
