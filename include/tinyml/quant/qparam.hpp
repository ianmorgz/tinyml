#pragma once

#include <algorithm>
#include <valarray>
#include <cmath>
#include <iostream>

#include "tinyml/tensor/tensor_view.hpp"

namespace tinyml::quant {

enum class QType: std::uint8_t {
    Symmetric,
    Asymmetric,
};

struct QParam {
    float scale = 1.0f;
    int8_t zero_point = 0;
    QType type;

    QParam() : scale(1.0f), zero_point(0), type(QType::Symmetric) {};
    QParam(const float scale, const int8_t zero_point, const QType type) : scale(scale), zero_point(zero_point), type(type) {}
    QParam(const tensor::TensorView<const float> &tv, const QType qt) {
        type = qt;
        const float* data = tv.data();
        const std::size_t n = tv.size();

        float min = data[0];
        float max = data[0];
        for (std::size_t i = 1; i < n; i++) {
            if (data[i] > max) { max = data[i]; }
            if (data[i] < min) { min = data[i]; }
        }


        if (min == max) {
            scale = 1.0f;
            zero_point = 0;
            return;
        }

        switch (qt) {
            case QType::Symmetric: {
                const float abs_max = std::max(std::abs(max), std::abs(min));
                // scale is the step size: float_per_int
                scale = (abs_max < 1e-12f) ? 1.0f : (abs_max / 127.0f);
                zero_point = 0;
                break;
            }

            case QType::Asymmetric: {
                const float diff = max - min;
                scale = (diff < 1e-12f) ? 1.0f : (diff / 254.0f);
                zero_point = (diff < 1e-12f) ? 0 : clamp_int8(-127.0f - std::round((min / scale)));
                break;
            }

            default: {
                TINYML_EXCEPTION("Quantized Param Constructor undefined quantization type");
            }
        }
    }
    QParam(const float max, const float min, QType qt) {
        type = qt;
        if (min == max) {
            scale = 1.0f;
            zero_point = 0;
            return;
        }

        switch (qt) {
            case QType::Symmetric: {
                const float abs_max = std::max(std::abs(max), std::abs(min));
                // scale is the step size: float_per_int
                scale = (abs_max < 1e-12f) ? 1.0f : (abs_max / 127.0f);
                zero_point = 0;
                break;
            }
            case QType::Asymmetric: {
                const float diff = max - min;
                scale = (diff < 1e-12f) ? 1.0f : (diff / 254.0f);

                const int32_t zp = (diff < 1e-12f) ? 0 : static_cast<int32_t>(std::lrintf(-127.0f - (min / scale)));
                zero_point = clamp_int8(zp);
                break;
            }
            default: {
                TINYML_EXCEPTION("Quantized Param Constructor undefined quantization type");
            }
        }
    }

    static int8_t clamp_int8(const float val) {
        int32_t q = static_cast<int32_t>(std::lrintf(val));
        if (q > 127) { q = 127; }
        if (q < -127) { q = -127; }
        return static_cast<int8_t>(q);
    }

    static void quantize_i8(const tensor::TensorView<const float>& in,const tensor::TensorView<int8_t>& out, const QParam param) {
        if (in.size() != out.size()) {
            TINYML_EXCEPTION("Quantized Param int8 quantization input size does not match output size");
        }

        const float* i_fp32 = in.data();
        int8_t* o_i8 = out.data();
        const std::size_t n = in.size();;
        const float s = param.scale;
        const int32_t z = param.zero_point;

        for (std::size_t i = 0; i < n; i++) {
            o_i8[i] = clamp_int8(round((i_fp32[i] / s) + z)); // TODO here
        }
    }

    static void dequantize_i8(const tensor::TensorView<const int8_t>& in, const tensor::TensorView<float>& out, const QParam param) {
        if (in.size() != out.size()) {
            TINYML_EXCEPTION("Quantized Param int8 quantization input size does not match output size");
        }

        const int8_t* i_i8 = in.data();
        float* o_fp32 = out.data();
        const std::size_t n = in.size();
        const float s = param.scale;
        const int32_t z = param.zero_point;

        for (std::size_t i = 0; i < n; i++) {
            o_fp32[i] = (i_i8[i] - z) * s;
        }
    }
};


}
