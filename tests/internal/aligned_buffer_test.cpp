#include <gtest/gtest.h>
#include <tinyml/internal/aligned_buffer.hpp>


TEST(AlignedBuffer, DefaultConstructIsEmpty) {
    const tinyml::internal::AlignedBuffer b;
    EXPECT_FALSE(b);
    EXPECT_EQ(b.size_bytes(), 0);
    EXPECT_EQ(b.alignment(), 0);
}

TEST(AlignedBuffer, AllocationIsAligned) {
    constexpr std::size_t k_align = 64;
    constexpr std::size_t k_size = 4096;

    tinyml::internal::AlignedBuffer b(k_size, k_align);
    ASSERT_TRUE(b);
    ASSERT_NE(b.data(), nullptr);

    EXPECT_EQ(b.size_bytes(), k_size);
    EXPECT_EQ(b.alignment(), k_align);
    EXPECT_NO_THROW(std::memset(b.data(), 0xAB, b.size_bytes()));
}

TEST(AlignedBuffer, MoveTransfersOwnership) {
    constexpr std::size_t k_size = 246;
    constexpr std::size_t k_align = 32;

    tinyml::internal::AlignedBuffer a(k_size, k_align);
    ASSERT_TRUE(a);

    const tinyml::internal::AlignedBuffer b(std::move(a));

    EXPECT_EQ(b.size_bytes(), k_size);
    EXPECT_EQ(b.alignment(), k_align);
}

TEST(AlignedBuffer, MoveReassigns) {
    constexpr std::size_t k_align = 32;

    tinyml::internal::AlignedBuffer a(512, k_align);
    tinyml::internal::AlignedBuffer b(1024, k_align);
    ASSERT_TRUE(a);
    ASSERT_TRUE(b);

    ASSERT_NE(b.data(), a.data());

    b = std::move(a);

    EXPECT_EQ(b.size_bytes(), 512);

    EXPECT_FALSE(a);
}