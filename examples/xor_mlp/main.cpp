#include <iostream>
#include "tinyml/dataset/dataset.hpp"
#include "tinyml/quant/qsequential.hpp"
#include "tinyml/model/sequential.hpp"
#include "tinyml/train/fit.hpp"
#include "xor_loader.hpp"
#include "tinyml/quant/qtest.hpp"
#include "tinyml/codegen/codegen.hpp"

using namespace tinyml;

int main() {
    // Load and split all the data
    dataset::Dataset dataset(std::move(std::make_unique<XOR_loader>(125)), 0.8f);

    model::Sequential model;
    model.dense(2, 8).relu().dense(8, 2);
    model.build(2, 800);

    train::fit(model, dataset, {
        .epochs = 500,
        .batch_size = 800,
        .learning_rate = 0.01f,
        .optimizer = train::Optimizer::SGD,
        .loss_function = train::LossFunction::CrossEntropy,
    });


    quant::QSequential qnet(model, dataset, 200);
    quant::qtest(qnet, dataset);

    codegen::generate(qnet, "../include/tinyml/runtime/files", "../out");
};
