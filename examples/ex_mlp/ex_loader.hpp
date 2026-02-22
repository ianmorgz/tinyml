#pragma once
#include "tinyml/dataset/loader.hpp"
#include "tinyml/dataset/batch.hpp"
#include "tinyml/core/config.hpp"

#include <string>
#include <fstream>
#include <vector>

using namespace tinyml;

class EXloader : public dataset::Loader {
public:
    EXloader() {}

    void load(dataset::Batch& out) const override {
        out.size = 400;

        out.input = tinyml::tensor::Tensor<float>({400u, 2u});
        out.label = tensor::Tensor<float>({400u, 1u});

        float* X = out.input.data();
        float* Y = out.label.data();

        for (std::uint32_t i = 0; i < 399; i+=8) {
            X[i+0] = 1.0f;
            X[i+1] = 1.0f;

            X[i+2] = 1.0f;
            X[i+3] = 0.0f;

            X[i+4] = 0.0f;
            X[i+5] = 1.0f;

            X[i+6] = 0.0f;
            X[i+7] = 0.0f;
        }

        for (std::uint32_t i = 0; i < 399; i+=4) {
            Y[i+0] = 0.0f;
            Y[i+1] = 1.0f;
            Y[i+2] = 1.0f;
            Y[i+3] = 0.0f;
        }
    }

};
