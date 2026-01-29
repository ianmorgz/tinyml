#include <gtest/gtest.h>
#include <tinyml/tensor/tensor.hpp>


TEST(Tensor, TypeTraits) {
    // Tensor should not be copyable
    EXPECT_FALSE(std::is_copy_constructible_v<tinyml::tensor::Tensor<float>>);
    EXPECT_FALSE(std::is_copy_assignable_v<tinyml::tensor::Tensor<float>>);

    // Tensor should be moveable
    EXPECT_TRUE(std::is_move_constructible_v<tinyml::tensor::Tensor<float>>);
    EXPECT_TRUE(std::is_move_assignable_v<tinyml::tensor::Tensor<float>>);
}

TEST(Tensor, Accessors) {
    tinyml::tensor::Tensor<float> t(tinyml::core::Shape{2, 4}, 3.0f);

    ASSERT_EQ(t.size(), 8);
    for (size_t i = 0; i < t.size(); i++) {
        EXPECT_EQ(t[i], 3.0f);
    }

    EXPECT_EQ(t.at(0, 0), 3.0f);
    EXPECT_EQ(t.at(1, 0), 3.0f);
    EXPECT_ANY_THROW(t.at(2, 0));
    EXPECT_ANY_THROW(t.at(1, 4));
}

TEST(Tensor, DefaultConstructorIsEmpty) {
    tinyml::tensor::Tensor<float> t;

    EXPECT_EQ(t.rank(), 0);
    EXPECT_EQ(t.size(), 0);
    EXPECT_EQ(t.shape().rank, 0);
    EXPECT_EQ(t.data(), nullptr);
}

TEST(Tensor, ConsructorShapeMatchesMetadata) {
    constexpr tinyml::core::Shape s{2, 3};
    const tinyml::tensor::Tensor<float> t(s);

    EXPECT_EQ(t.rank(), 2);
    EXPECT_EQ(t.size(), 6);
    EXPECT_EQ(t.shape()[0], 2);
    EXPECT_EQ(t.shape()[1], 3);
}

TEST(Tensor, FillWritesAllElements) {
    tinyml::tensor::Tensor<int8_t> t(tinyml::core::Shape{2, 3});
    t.fill(7);

    for (size_t i = 0; i < t.size(); i++) {
        EXPECT_EQ(t[i], 7);
        EXPECT_EQ(t.data()[i], 7);
    }
}

TEST(Tensor, ConstViewIsReadOnly) {
    tinyml::tensor::Tensor<float> temp(tinyml::core::Shape{2, 3});
    temp.fill(1.0f);

    const tinyml::tensor::Tensor<float>& t = temp;
    auto t_v = t.view();

    EXPECT_TRUE((std::is_const_v<typename decltype(t_v)::value_type>));

    ASSERT_NE(t_v.data(), nullptr);
}

TEST(Tensor, MoveLeavesSourceInSafeEmptyState) {
    tinyml::tensor::Tensor<float> a(tinyml::core::Shape{2, 2}, 9.0f, /*alignment=*/64);

    void* p_a = a.data();
    ASSERT_NE(p_a, nullptr);

    tinyml::tensor::Tensor<float> b(std::move(a));

    // Destination should have taken ownership
    EXPECT_EQ(b.size(), 4);
    EXPECT_EQ(b.rank(), 2);
    EXPECT_EQ(b.data(), p_a);
    EXPECT_FLOAT_EQ(b[0], 9.0f);

    // Source should be safe-to-use as "empty"
    // (This is a strong invariant you should enforce.)
    EXPECT_EQ(a.data(), nullptr);
    EXPECT_EQ(a.size(), 0);
    EXPECT_EQ(a.rank(), 0);
}
