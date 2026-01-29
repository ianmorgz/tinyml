#pragma once

#include <algorithm>
#include <cstdint>
#include <valarray>
#include <cmath>

#include "tinyml/tensor/tensor_view.hpp"

namespace tinyml::quant {

enum class QType: std::uint8_t {
    Symmetric,
    Asymmetric,
};

inline int8_t clamp_int8(const float val) {
    int32_t f = std::lrintf(val);
    if (f > 127) { f = 127; }
    if (f < -127) { f = -127; }
    return static_cast<int8_t>(f);
}

inline int8_t clamp_int8(int32_t val) {
    if (val > 127) { val = 127; }
    if (val < -127) { val = -127; }
    return static_cast<int8_t>(val);
}

struct QParam {
    float scale = 1.0f;
    int8_t zero_point = 0;
    QType type;

    QParam() = default;
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

                const int32_t zp = (diff < 1e-12f) ? 0 : static_cast<int32_t>(std::lrintf(-127.0f - (min / scale)));
                zero_point = clamp_int8(zp);
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
};


}
