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
        .epochs = 2,
        .batch_size = 100,
        .learning_rate = 0.01f,
        .optimizer = train::Optimizer::SGD,
        .loss_function = train::LossFunction::CrossEntropy,
    });

    quant::QSequential qnet(model, dataset, 10000);

    //DEBUG test the quantized forward method
    dataset.shuffle_training(42);
    dataset::BatchView batch_view;

    std::size_t testing_size = 20;
    int correct = 0;
    for (std::size_t i = 0; i < testing_size; ++i) {
        dataset.next_training_batch(1, batch_view);

        auto in = batch_view.input;
        auto lbl = batch_view.label;;

        tensor::Tensor<float> q_out(lbl.shape());
        auto qout_view = q_out.view();
        // tensor::Tensor<float> fp32_out(lbl.shape());

        auto o = model.forward(in);
        qnet.forward(in, qout_view);


        const float* q_o = q_out.data();
        const float* fp32_o = o.data();
        const float* l = lbl.data();
        std::size_t sz = lbl.size();

        // float max_out = -100;
        // float max_lbl = -100;
        //
        // std::size_t max_out_idx = 0;
        // std::size_t max_lbl_idx = 0;

        // if (i%100 == 0) {
            for (std::size_t j = 0; j < sz; ++j) {
                std::cout << (j+1) << ": fp32net: " << fp32_o[j] << ", quantnet: " << q_o[j] << " : Expected: " << l[j] << std::endl;
            }
            std::cout << "===============\n";
        // }


        // for (std::size_t j = 0; j < sz; ++j) {
        //     if (o[j] > max_out) {max_out = o[j]; max_out_idx = j;}
        //     if (l[j] > max_lbl) {max_lbl = l[j]; max_lbl_idx = j;}
        // }
        //
        // if (max_out_idx == max_lbl_idx) { correct++; }
    }

    std::cout << correct << "/" << testing_size << std::endl;

};