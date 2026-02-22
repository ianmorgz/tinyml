#include <iostream>
#include "mnist_loader.hpp"
#include "tinyml/dataset/dataset.hpp"
#include "tinyml/quant/qsequential.hpp"
#include "tinyml/model/sequential.hpp"
#include "tinyml/train/fit.hpp"

using namespace tinyml;

int main() {
    // Load and split all the data
    dataset::Dataset dataset(std::move(std::make_unique<MNISTLoader>(
        "../examples/mnist_mlp/data/train-images.idx3-ubyte",
        "../examples/mnist_mlp/data/train-labels.idx1-ubyte")),
        0.8f);

    model::Sequential model;
    model.dense(784, 128).relu().dense(128, 64).relu().dense(64, 10);
    model.build(784, 100);

    train::fit(model, dataset, {
        .epochs = 1,
        .batch_size = 100,
        .learning_rate = 0.01f,
        .optimizer = train::Optimizer::SGD,
        .loss_function = train::LossFunction::CrossEntropy,
    });

    quant::QSequential qnet(model, dataset, 10000);

    //DEBUG test the quantized forward method
    dataset.shuffle_training(42);
    dataset::BatchView batch_view;
    dataset.next_training_batch(1, batch_view);

    auto in = batch_view.input;
    auto lbl = batch_view.label;;


    tensor::Tensor<const float> out(lbl.shape());
    out.view() = qnet.forward(in);

    for (std::size_t i = 0; i < out.size(); ++i) {
        std::cout << "output: " << out[i] << " | expected: " << lbl[i] << std::endl;
    }
}
