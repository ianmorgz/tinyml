#include <gtest/gtest.h>
#include <tinyml/core/buffer_view.hpp>

TEST(BufferView, DefaultConstructorIsEmpty) {
    constexpr tinyml::core::BufferView b;
    EXPECT_TRUE(b.empty());
    EXPECT_FALSE(b);
    EXPECT_EQ(b.data, nullptr);
    EXPECT_EQ(b.bytes, 0);
    EXPECT_EQ(b.alignment, 0);

    constexpr tinyml::core::ConstBufferView c_b;
    EXPECT_TRUE(c_b.empty());
    EXPECT_FALSE(c_b);
    EXPECT_EQ(c_b.data, nullptr);
    EXPECT_EQ(c_b.bytes, 0);
    EXPECT_EQ(c_b.alignment, 0);
}

TEST(BufferView, PopulatedBufferTruthfullness) {
    std::uint8_t buffer[16] = {};
    tinyml::core::BufferView b{ buffer, sizeof(buffer), alignof(uint8_t)};

    EXPECT_FALSE(b.empty());
    EXPECT_TRUE(b);

    std::uint8_t c_buffer[16] = {};
    tinyml::core::BufferView c_b{ c_buffer, sizeof(c_buffer), alignof(uint8_t)};

    EXPECT_FALSE(c_b.empty());
    EXPECT_TRUE(c_b);
}

TEST(BufferView, AsSpan) {
    std::vector<std::uint32_t> v(4);
    v[1] = 77;

    const tinyml::core::ConstBufferView b{ v.data(), v.size() * sizeof(std::int32_t), alignof(std::uint32_t) };
    const auto s = b.as_span<const std::int32_t>();

    ASSERT_EQ(s.size(), v.size());
    EXPECT_EQ(s[1], 77);
}

TEST(BufferView, AsSpanRejectsNull) {
    const tinyml::core::BufferView b{ nullptr, 16, 4};
    EXPECT_ANY_THROW((void)b.as_span<std::uint8_t>());
}

