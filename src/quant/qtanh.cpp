#include "tinyml/quant/qtanh.hpp"

void tinyml::quant::QTanh::finalize_callibration(const model::Layer &layer) {
    LUT = tensor::Tensor<int8_t>({254u});
}
