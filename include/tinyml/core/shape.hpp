#pragma once

#include <cstdint>
#include <initializer_list>
#include <limits>

#include "config.hpp"

namespace tinyml::core {
struct Shape final {
    static constexpr std::size_t MAX_RANK = 4;
    std::uint32_t rank = 0;
    std::uint32_t dims[MAX_RANK] = {0, 0, 0, 0};

    // ---- Constructors ----
    constexpr Shape() = default;

    constexpr Shape(const std::initializer_list<std::uint32_t> d) {
        if (d.size() > MAX_RANK) { TINYML_EXCEPTION("Shape dimensions exceeds maximum number of allowed dimensions"); }
        rank = static_cast<std::uint32_t>(d.size());
        std::size_t i = 0;
        for (const std::uint32_t& d_i : d) {
            dims[i++] = d_i;
        }
    }

    constexpr Shape(const std::initializer_list<std::size_t> d) {
        if (d.size() > MAX_RANK) {
            TINYML_EXCEPTION("Shape dimensions exceeds maximum number of allowed dimensions");
        }

        rank = static_cast<std::uint32_t>(d.size());

        std::size_t i = 0;
        for (const std::size_t& d_i : d) {
            if (d_i > std::numeric_limits<std::uint32_t>::max()) {
                TINYML_EXCEPTION("Shape dimension overflow (size_t too large for uint32_t)");
            }
            dims[i++] = static_cast<std::uint32_t>(d_i);
        }
    }


    // ---- Getter Methods ----
    constexpr std::uint32_t operator[](const std::size_t i) const {
        if (i >= rank) { TINYML_EXCEPTION("Shape index out of bounds"); }
        return dims[i];
    }

    constexpr std::uint32_t& operator[](const std::size_t i) {
        if (i >= rank) { TINYML_EXCEPTION("Shape index out of bounds"); }
        return dims[i];
    }

    constexpr std::size_t flat_size() const {
        if (rank == 0) { return 0; }
        std::size_t n = 1;
        for (std::size_t i = 0; i < rank; ++i) {
            const auto d = static_cast<std::size_t>(dims[i]);
            if (d == 0 || n > (std::numeric_limits<std::uint32_t>::max() / d)) { TINYML_EXCEPTION("Shape flat size overflow"); }
            n *= d;
        }
        return n;
    }

    constexpr bool operator==(const Shape& other) const noexcept {
        if (rank != other.rank) return false;
        for (std::size_t i = 0; i < rank; ++i) {
            if (dims[i] != other.dims[i]) return false;
        }
        return true;
    }

    constexpr bool operator!=(const Shape& other) const noexcept {
        return !(*this == other);
    }

    constexpr bool empty() const { return flat_size() == 0; }
};
}
