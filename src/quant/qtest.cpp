#include "iomanip"

#include "tinyml/quant/qtest.hpp"
#include "tinyml/train/fit.hpp"

namespace tinyml::quant {

void qtest(QSequential& qnet, dataset::Dataset& dataset) {
    std::cout << "\rTesting..." << std::flush;
    // dataset.shuffle_testing(142); TODO shuffle the testing dataset
    dataset::BatchView batch_view;

    std::size_t label_size = dataset.label_size();

    //output tennsor and view
    tensor::Tensor<float> q_out({label_size});
    auto qout_view = q_out.view();

    std::uint32_t total = 0;
    std::uint32_t correct = 0;
    while (dataset.next_testing_batch(1, batch_view)){
        auto in = batch_view.input;
        auto lbl = batch_view.label;

        qnet.forward(in, qout_view);

        if (train::argmax(qout_view.data(), label_size) == train::argmax(lbl.data(), label_size)) {
            correct++;
        }

        total++;
    }

    const float acc = ((total > 0) ? (static_cast<float>(correct) / static_cast<float>(total)) : 0.0f) * 100.0f;
    std::cout << "Testing accuracy: " << correct << "/" << total << " (" << std::fixed << std::setprecision(2) << acc << "%)\n" << std::flush;
}


}