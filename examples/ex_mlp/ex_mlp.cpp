#include <iostream>
#include "tinyml/dataset/dataset.hpp"
#include "tinyml/quant/qsequential.hpp"
#include "tinyml/model/sequential.hpp"
#include "tinyml/train/fit.hpp"
#include "ex_loader.hpp"

using namespace tinyml;

int main() {
    // Load and split all the data
    dataset::Dataset dataset(std::move(std::make_unique<EXloader>()), 0.8f);

    model::Sequential model;
    model.dense(2, 6).relu().dense(6, 1);
    model.build(2, 320);

    train::fit(model, dataset, {
        .epochs = 1,
        .batch_size = 320,
        .learning_rate = 0.01f,
        .optimizer = train::Optimizer::SGD,
        .loss_function = train::LossFunction::CrossEntropy,
    });


    quant::QSequential qnet(model, dataset, 80);

    //DEBUG test the quantized forward method
    dataset.shuffle_training(17469196462);

    std::size_t num = 40;
    std::size_t correct = 0;

    for (std::size_t i = 0; i < num; ++i) {
        dataset::BatchView batch_view;
        dataset.next_training_batch(1, batch_view);

        auto in = batch_view.input;
        auto lbl = batch_view.label;;

        tensor::Tensor<const float> out(lbl.shape());
        out.view() = qnet.forward(in);

        if (out[0] >= 0.5f) { correct++; }
    }

    std::cout << "non zero: " << correct << "/" << num << std::endl;
};
