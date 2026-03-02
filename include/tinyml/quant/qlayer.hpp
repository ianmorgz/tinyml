#pragma once

#include "qparam.hpp"
#include "tinyml/tensor/tensor_view.hpp"
#include "tinyml/model/layer.hpp"


/* A quantized layer is built with the following methods
 * 1. The layer observes the output of it's fp32 counterpart, saving necessary data to build input/output params and/or LUTs
 * 2. the layer then finalizes it's internal structure and is ready for the forward pass
 */

namespace tinyml::quant {

enum class QLayer_Type : std::uint8_t {
    QDense,
    QReLu,
    QTanh,
};

class QLayer {
public:
    virtual ~QLayer() = default;

    // the quantized forward pass
    virtual void forward(tensor::TensorView<const int8_t> in, tensor::TensorView<int8_t> out) const  = 0;

    // observe and output and adjust
    virtual void observe_fp32_output(tensor::TensorView<const float> out) = 0;

    // build the quantized layer with observed adjustments returns the output param for the next layer
    virtual QParam finalize_calibration(const model::Layer& layer, QParam input_param) = 0;

    // check if the layer has been calibrated
    virtual bool calibrated() const noexcept = 0;

    virtual QLayer_Type type() const noexcept = 0;
};

}
