#pragma once

#include <stdint.h>
#include <stddef.h>

#ifndef ARENA_DEFAULT_ALIGNMENT
#define ARENA_DEFAULT_ALIGNMENT 4u
#endif

typedef struct {
    uint8_t* base;
    size_t size;
    size_t top;
} Arena;

static inline uintptr_t align_up_uintptr(uintptr_t x, size_t a) {
    return (x + (a - 1u)) & ~(uintptr_t)(a - 1u);
}

static inline void arena_init(Arena* a, void* mem, size_t bytes) {
    a->base = (uint8_t*)mem;
    a->size = bytes;
    a->top = 0u;
}

static inline void arena_reset_all(Arena* a) {
    a->top = 0u;
}

// allocate N bytes with alignment A (power of 2). Returns NULL on OOM.
static inline void* arena_alloc(Arena* a, size_t bytes, size_t align) {
    if(align == 0u) align = ARENA_DEFAULT_ALIGNMENT;

    uintptr_t base_addr = (uintptr_t)a->base;
    uintptr_t cur_addr  = base_addr + (uintptr_t)a->top;
    uintptr_t aligned  = align_up_uintptr(cur_addr, align);

    size_t new_top = (size_t)(aligned - base_addr) + bytes;
    if (new_top > a->size) return NULL;

    a->top = new_top;
    return (void*)aligned;
}

static inline size_t arena_mark(const Arena* a) { return a->top; }
static inline void arena_reset_to(Arena* a, size_t mark) {
    if(mark <= a->size) { a->top = mark; }
}