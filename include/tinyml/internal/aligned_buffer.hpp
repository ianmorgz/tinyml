#pragma once

#include <cstdio>
#include <memory>

namespace tinyml::internal {
struct AlignedDeleter final {
    std::size_t alignment = 0;
    void operator()(void* p) const noexcept;
};

class AlignedBuffer final {
public:
    AlignedBuffer() = default;
    AlignedBuffer(std::size_t bytes, std::size_t alignment);

    // copy deleted
    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    // move defaulted
    AlignedBuffer(AlignedBuffer&&) noexcept = default;
    AlignedBuffer& operator=(AlignedBuffer&&) noexcept = default;

    void* data() const noexcept { return ptr_.get(); }
    std::size_t size_bytes() const noexcept { return bytes_; }
    std::size_t alignment() const noexcept {return alignment_; }
    explicit operator bool() const noexcept { return ptr_ != nullptr; }

private:
    std::unique_ptr<void, AlignedDeleter> ptr_ {nullptr, AlignedDeleter{0}};
    std::size_t bytes_ = 0;
    std::size_t alignment_ = 0;
};
}
