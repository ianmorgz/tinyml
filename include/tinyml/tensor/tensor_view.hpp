#pragma once

#include "tinyml/core/shape.hpp"

namespace tinyml::tensor {

template <typename T>
struct TensorView {
public:
    using value_type = T;

    // ---- Constructors ----
    constexpr TensorView() noexcept = default;
    constexpr TensorView(T* data_p, const core::Shape &shape) noexcept : data_(data_p), shape_(shape) {}

    // ---- Getter Methods ----
    constexpr T* data() const noexcept { return data_; }
    constexpr const core::Shape& shape() const noexcept { return shape_; }
    constexpr std::size_t size() const noexcept { return shape_.flat_size(); }
    constexpr std::size_t rank() const noexcept { return shape_.rank; }

    // ---- Setter Methods ----
    void resize(const core::Shape& other) noexcept {
        if (other.flat_size() != shape_.flat_size()) {
            TINYML_EXCEPTION("Cannot resize with different sized shape, must match size");
        }
        shape_ = other;
    }

    constexpr const T& operator[](const std::size_t i) const {
        if (i >= size()) { TINYML_EXCEPTION("Shape index out of bounds"); }
        return data_[i];
    }

    constexpr const T& at(const std::size_t row, const std::size_t col) const {
        if (shape_.rank != 2){ TINYML_EXCEPTION("2-dimension Tensor shape required for .at(row, col) method"); }
        if (row >= shape_.dims[0] || col >= shape_.dims[1]){ TINYML_EXCEPTION("Shape index out of bounds"); }
        return data_[row * static_cast<std::size_t>(shape_.dims[1]) + col];
    }

    // read only TensorView
    constexpr TensorView<const std::remove_const_t<T>> as_const() const noexcept {
        using U = std::remove_const_t<T>;
        return TensorView<const U>(data_, shape_);
    }

private:
    T* data_ = nullptr;
    core::Shape shape_ { };
};


template<typename T>
constexpr TensorView<T> view(T* data, const core::Shape& s) noexcept{
    return TensorView<T>(data, s);
}
}