#include "tinyml/train/fit.hpp"

#include <iomanip>

#include "tinyml/dataset/dataset.hpp"
#include "../../include/tinyml/train/context.hpp"
#include <iostream>

#include "tinyml/train/ce_loss.hpp"
#include "tinyml/train/loss.hpp"
#include "tinyml/train/optimizer.hpp"

namespace tinyml::train {
void fit(model::Sequential& model, dataset::Dataset& dataset, const Config &config) {
    std::cout << "===================== TINY ML TRAINING =====================\n";
    dataset::BatchView batch;
    core::Context ctx;
    tensor::Tensor<float> loss_gradient(core::Shape({ static_cast<std::uint32_t>(config.batch_size), static_cast<std::uint32_t>(model.output_features()) }));

    for (std::size_t e = 0; e < config.epochs; ++e){
        dataset.shuffle_training(123 + e);
        float loss_sum = 0.0f;
        std::size_t step = 0;

        while (dataset.next_training_batch(config.batch_size, batch)) {
            CrossEntropyLoss CELoss; // TODO
            const auto output = model.forward_train(batch.input, ctx);
            auto loss_gradient_view = loss_gradient.view();

            loss_sum += CELoss.forward(output, batch.label);    // TODO
            CELoss.backward(output, batch.label, loss_gradient_view); // TODO
            model.backwards_train(loss_gradient_view.as_const(), ctx);

            SGD_step(model.params(), config.learning_rate); // TODO
            model.clear_param_gradients();

            step++;
            const auto steps = static_cast<std::size_t>((static_cast<float>(dataset.training().size) / static_cast<float>(config.batch_size)) + 1);
            const float avg_loss = loss_sum / static_cast<float>(step);
            print_progress(e, config.epochs, step+1, steps, avg_loss);
        }
    }

    // testing loop
    std::cout << "\rTesting..." << std::flush;

    std::int32_t correct = 0;
    std::size_t total = 0;
    const std::size_t classes = model.output_features();

    while (dataset.next_testing_batch(config.batch_size, batch)) {
        auto output = model.forward(batch.input);
        const float* outp = output.data();
        const float* tgt  = batch.label.data();

        for (std::size_t i = 0; i < batch.size; ++i) {
            const float* pred_row   = outp + i * classes;
            const float* target_row = tgt  + i * classes;
            if (check_class(pred_row, target_row, classes)) {
                ++correct;
            }
        }
        total += batch.size;
    }

    std::cout << "\r\033[2K"; // clear the testing... placeholder
    const float acc = ((total > 0) ? (static_cast<float>(correct) / static_cast<float>(total)) : 0.0f) * 100.0f;
    std::cout << "Testing accuracy: " << correct << "/" << total << " (" << std::fixed << std::setprecision(2) << acc << "%)\n" << std::flush;
}

inline void print_progress(const std::size_t epoch, const std::size_t epochs, const std::size_t step, const std::size_t steps, const float avg_loss) {
    constexpr int bar_width = 30;
    const float frac = (steps == 0) ? 1.0f : static_cast<float>(step) / static_cast<float>(steps);
    const int filled = static_cast<int>(frac * bar_width);

    std::cout << "\r" << "Epoch " << (epoch + 1) << "/" << epochs << " [";

    for (int i = 0; i < bar_width; ++i) {
        std::cout << (i < filled ? '=' : ' ');
    }

    std::cout << "] "
              << std::setw(3) << static_cast<int>(frac * 100.0f) << "% "
              << "loss " << std::fixed << std::setprecision(4) << avg_loss
              << std::flush;

    // print newline at end of epoch
    if (step >= steps) {
        std::cout << "\n";
    }
}

inline std::size_t argmax(const float* x, const std::size_t n) {
    std::size_t idx = 0;
    float best = x[0];
    for (std::size_t i = 1; i < n; ++i) {
        if (x[i] > best) { best = x[i]; idx = i; }
    }
    return idx;
}

inline bool check_class(const float* logits_row, const float* onehot_row, std::size_t classes) {
    return argmax(logits_row, classes) == argmax(onehot_row, classes);
}
}
