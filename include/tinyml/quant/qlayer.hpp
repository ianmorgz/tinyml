#pragma once

#include "qparam.hpp"
#include "tinyml/tensor/tensor_view.hpp"
#include "tinyml/model/layer.hpp"

namespace tinyml::quant {
class QLayer {
public:
    virtual ~QLayer() = default;

    // the quantized forward pass
    virtual void forward(tensor::TensorView<const int8_t> in, tensor::TensorView<int8_t> out) const  = 0;

    // observe and output and adjust
    virtual void observe_fp32_output(tensor::TensorView<const float> out) = 0;

    // build the quantized layer with obsereved adjustments returns the output param for the next layer
    virtual QParam finalize_callibration(const model::Layer& layer, QParam input_param) = 0;

    // check if the layer has been callibrated
    virtual bool callibrated() const noexcept = 0;
};
}
