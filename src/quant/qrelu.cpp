#include "tinyml/quant/qrelu.hpp"
#include <math.h>

namespace tinyml::quant {

// perform a relu sort with abs max
void QRelu::forward(const tensor::TensorView<const int8_t> in, tensor::TensorView<int8_t> out) const {
    const int8_t* i = in.data();
    int8_t* o = out.data();
    const std::size_t n = in.size();

    // std:: cout << "DEBUG: ";
    for (std::size_t j = 0; j < n; ++j) {
        o[j] = std::max(i[j], param_.zero_point);
        // std::cout << static_cast<int>(o[j]) << " ";
    }
    // std::cout << std::endl;


}

QParam QRelu::finalize_calibration(const model::Layer& layer, const QParam input_param) {
    (void) layer;

    param_ = input_param;
    calibrated_ = true;

    return input_param;
}
}