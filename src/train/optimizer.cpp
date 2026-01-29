#include "tinyml/train/optimizer.hpp"

#include <iostream>

#include "tinyml/model/layer.hpp"
#include "tinyml/core/config.hpp"
#include <vector>

namespace tinyml::train {

void SGD_step(std::span<const model::ParamRef> params, float lr) {
    for (const auto& pr : params) {
        auto w = pr.param->view();     // view created fresh NOW
        auto g = pr.grad->view();      // view created fresh NOW

        if (w.shape() != g.shape()) TINYML_EXCEPTION("SGD: param/grad shape mismatch");

        float* wp = w.data();
        const float* gp = g.data();
        const size_t n = w.size();

        for (size_t i = 0; i < n; ++i) {
            wp[i] -= lr * gp[i];
        }
    }
}
}