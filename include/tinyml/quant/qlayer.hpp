#pragma once

#include <cstdio>
#include "qparam.hpp"
#include "tinyml/tensor/tensor.hpp"
#include "tinyml/model/layer.hpp"

namespace tinyml::quant {
class QLayer {
public:
    virtual ~QLayer() = default;
    virtual void forward(tensor::TensorView<const int8_t> in, tensor::TensorView<int8_t> out) const  = 0;
    virtual void observe_fp32_output(tensor::TensorView<const float> out) = 0;
    virtual void finalize_callibration(const model::Layer& layer) = 0;
    virtual QParam output_param();
    virtual bool callibrated() const noexcept = 0;
};
}
