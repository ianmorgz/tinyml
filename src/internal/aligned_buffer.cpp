#include "tinyml/internal/aligned_buffer.hpp"
#include "../../include/tinyml/core/config.hpp"

namespace tinyml::internal {
static void* aligned_alloc_portable(std::size_t alignment, std::size_t size) { // allocation function
    // make sure that alignment is a power of two nd a multiple of sizeof()
    if (alignment < alignof(void*)) { alignment = alignof(void*); }
    if ((alignment & (alignment - 1)) != 0) {
        TINYML_EXCEPTION("Alignment must be power of 2");
    }

#if defined(_MSC_VER) // windows compilation
    void* p = aligned_malloc(size, alignment);
    if (!p) { TINYML_EXCEPTION("Aligned memory allocation failed"); }
    return p;
#elif defined(__APPLE__) || defined(__LINUX__) // apple or linux compilation
    void* p = nullptr;
    if (posix_memalign(&p, alignment, size) != 0 || !p) { TINYML_EXCEPTION("Aligned memory allocation failed"); }
    return p;
#else // fallback: could not determine system compilation or non-supported system
    void* p = std::malloc(size);
    if (!p) { TINYML_EXCEPTION("Aligned memory allocation failed"); }
    return p;
#endif
}

static void aligned_free_portable(void* p) noexcept { // free function
    if (!p) return;
#if defined(_MSC_VER)
    _aligned_free(p);
#else
    std::free(p);
#endif
}

void AlignedDeleter::operator()(void* p) const noexcept {
    aligned_free_portable(p);
}

AlignedBuffer::AlignedBuffer(const std::size_t bytes, const std::size_t alignment) : ptr_(nullptr, AlignedDeleter{ alignment }), bytes_(bytes), alignment_(alignment) {
    if (bytes_ == 0) {
        // empty buffers allowed
        return;
    }
    void* mem = aligned_alloc_portable(alignment, bytes);
    ptr_.reset(mem);
}
}