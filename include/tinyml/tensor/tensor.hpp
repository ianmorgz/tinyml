#pragma once

#include <stdexcept>
#include <cstddef>
#include <type_traits>

#include "tinyml/tensor/tensor_view.hpp"
#include "tinyml/core/config.hpp"
#include "tinyml/internal/aligned_buffer.hpp"
#include "tinyml/core/buffer_view.hpp"
#include "tinyml/core/shape.hpp"

namespace tinyml::tensor {

template <typename T>
class Tensor final {
public:
    // ensure T is a usable data type (int, float, etc)
    static_assert(std::is_trivially_copyable_v<T>, "Tensor<T> requires trivially copyable T");
    static_assert(std::is_trivially_destructible_v<T>, "Tensor<T> requires trivially destructible T");

    using value_type = T;

    // ---- Constructors ----
    Tensor() = default;

    explicit Tensor(const core::Shape &s, const std::size_t alignment = alignof(T)) : shape_(s), buf_(s.flat_size() * sizeof(T), alignment) {}

    Tensor(core::Shape s, const T& fill_value, std::size_t alignment = alignof(T)) : Tensor(s, alignment){
        fill(fill_value);
    }

    // ---- Copy and Move Methods ----
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&& other) noexcept : shape_(other.shape_), buf_(std::move(other.buf_)){ other.shape_ = core::Shape{}; };
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            shape_ = other.shape_;
            buf_   = std::move(other.buf_);
            other.shape_ = core::Shape{};
        }
        return *this;
    }

    // ---- Getter Methods ----
    const core::Shape& shape() const noexcept { return shape_; }
    size_t size() const noexcept { return shape_.flat_size(); }
    size_t rank() const noexcept { return shape_.rank; }

    T* data() noexcept { return reinterpret_cast<T*>(buf_.data()); }
    const T* data() const noexcept { return reinterpret_cast<const T*>(buf_.data()); }

    std::size_t n_bytes() const noexcept { return size() * sizeof(T); }

    TensorView<T> view()  noexcept { return TensorView<T>(data(), shape_); }
    TensorView<const T> view() const noexcept { return TensorView<const T>(data(), shape_); }

    const T& operator[](std::size_t i) const {
        if (i >= size()) TINYML_EXCEPTION("Tensor index out of bounds");
        return data()[i];
    }

    T& operator[](std::size_t i) {
        if (i >= size()) TINYML_EXCEPTION("Tensor index out of bounds");
        return data()[i];
    }

    const T& at(const std::size_t row, const size_t col) const {
        if (shape_.rank != 2) { TINYML_EXCEPTION("2-dim Tensor shape required for .at(row, col) method"); }
        if (row >= shape_.dims[0] || col >= shape_.dims[1]) { TINYML_EXCEPTION("Shape index out of bounds"); }
        return data()[row * static_cast<std::size_t>(shape_.dims[1]) + col];
    }

    T& at(const std::size_t row, const size_t col) {
        if (shape_.rank != 2) { TINYML_EXCEPTION("2-dim Tensor shape required for .at(row, col) method"); }
        if (row >= shape_.dims[0] || col >= shape_.dims[1]) { TINYML_EXCEPTION("Shape index out of bounds"); }
        return data()[row * static_cast<std::size_t>(shape_.dims[1]) + col];
    }

    core::BufferView buffer() noexcept {
        return core::BufferView{ static_cast<void*>(data()), n_bytes(), buf_.alignment() };
    }

    core::ConstBufferView buffer() const noexcept {
        return core::ConstBufferView{ static_cast<void*>(data()), n_bytes(), buf_.alignment() };
    }

    // ---- Setter Method ----
    void fill(const T& v) {
        for (size_t i = 0; i < size(); ++i) { data()[i] = v; }
    }

private:
    core::Shape shape_{};
    internal::AlignedBuffer buf_{};
};

}