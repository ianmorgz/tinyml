#pragma once

#include <span>

#include "tinyml/core/config.hpp"

namespace tinyml::core {

struct BufferView final {
    void* data = nullptr;
    std::size_t bytes = 0;
    std::size_t alignment = 0; // 0 if unknown

    constexpr bool empty() const noexcept { return bytes == 0 || data == nullptr; }
    constexpr explicit operator bool() const { return !empty(); }

    std::byte* byte_data() const noexcept {
        return reinterpret_cast<std::byte*>(data);
    }

    template <typename T>
    std::span<T> as_span() const {
        if (!data) TINYML_EXCEPTION("BufferView::as_span() called with null data");
        if (alignment != 0) {
            if (reinterpret_cast<std::uintptr_t>(data) % alignof(T) != 0) TINYML_EXCEPTION("BufferView::as_span() misaligned pointer");
        }
        if (bytes % sizeof(T) != 0) TINYML_EXCEPTION("BufferView::as_span() bytes not divisible by size of T");
        return { reinterpret_cast<T*>(data),  bytes / sizeof(T) };
    }
};

struct ConstBufferView final {
    const void* data = nullptr;
    std::size_t bytes = 0;
    std::size_t alignment = 0; // 0 if unknown

    constexpr bool empty() const noexcept { return bytes == 0 || data == nullptr; }
    constexpr explicit operator bool() const { return !empty(); }

    const std::byte* byte_data() const noexcept {
        return reinterpret_cast<const std::byte*>(data);
    }

    template <typename T>
    std::span<T> as_span() const {
        if (!data) TINYML_EXCEPTION("BufferView::as_span() called with null data");
        if (alignment != 0) {
            if (reinterpret_cast<std::uintptr_t>(data) % alignof(T) != 0) TINYML_EXCEPTION("BufferView::as_span() misaligned pointer");
        }
        if (bytes % sizeof(T) != 0) TINYML_EXCEPTION("BufferView::as_span() bytes not divisible by size of T");
        return { reinterpret_cast<T*>(data),  bytes / sizeof(T) };
    }
};

}
