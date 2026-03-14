#include <iostream>
#include "mnist_loader.hpp"
#include "tinyml/dataset/dataset.hpp"
#include "tinyml/quant/qsequential.hpp"
#include "tinyml/model/sequential.hpp"
#include "tinyml/quant/qtest.hpp"
#include "tinyml/train/fit.hpp"
#include "tinyml/codegen/codegen.hpp"
// MNIST example
// meant to show training and quantization.
// NOT meant to be used in embedded systems until
// some form of weight chunking is initazlized as
// quantized weights alone are ~100 Kb


using namespace tinyml;

int main() {
    // Load and split all the data
    dataset::Dataset dataset(std::move(std::make_unique<MNISTLoader>(
        "../examples/mnist_mlp/data/train-images.idx3-ubyte",
        "../examples/mnist_mlp/data/train-labels.idx1-ubyte")),
        0.8f);


    // create and build the model
    model::Sequential model;
    model.dense(784, 128).relu().dense(128, 64).relu().dense(64, 10);
    model.build(784, 100);

    // train the model
    train::fit(model, dataset, {
        .epochs = 1,
        .batch_size = 100,
        .learning_rate = 0.01f,
        .optimizer = train::Optimizer::SGD,
        .loss_function = train::LossFunction::CrossEntropy,
    });

    // quantize and test the model
    quant::QSequential qnet(model, dataset, 10000);
    quant::qtest(qnet, dataset);

};