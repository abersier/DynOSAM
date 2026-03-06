#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "dynosam_common/Types.hpp"

using namespace dyno;

struct MeterTag {};
struct SecondTag {};
struct IdTag {};

using Meter = dyno::Strong<double, MeterTag>;
using Second = dyno::Strong<double, SecondTag>;
using Id = dyno::Strong<int, IdTag>;

TEST(StrongTypeTest, VerifyCompileTimeSafety) {
  static_assert(!std::is_convertible_v<Meter, Second>);
  static_assert(!std::is_convertible_v<Second, Meter>);
}

// ------------------------------------------------------------
// Construction
// ------------------------------------------------------------

TEST(StrongTypeTest, DefaultConstruction) {
  Meter m;
  EXPECT_DOUBLE_EQ(static_cast<double>(m), 0.0);
}

TEST(StrongTypeTest, ValueConstruction) {
  Meter m{5.5};
  EXPECT_DOUBLE_EQ(static_cast<double>(m), 5.5);
}

// ------------------------------------------------------------
// Accessor
// ------------------------------------------------------------

TEST(StrongTypeTest, GetAccessor) {
  Meter m{3.2};

  EXPECT_DOUBLE_EQ(m.get(), 3.2);

  m.get() = 7.4;

  EXPECT_DOUBLE_EQ(m.get(), 7.4);
}

// ------------------------------------------------------------
// Arithmetic Operators
// ------------------------------------------------------------

TEST(StrongTypeTest, Addition) {
  Meter a{3.0};
  Meter b{2.0};

  Meter c = a + b;

  EXPECT_DOUBLE_EQ(static_cast<double>(c), 5.0);
}

TEST(StrongTypeTest, Subtraction) {
  Meter a{7.0};
  Meter b{2.0};

  Meter c = a - b;

  EXPECT_DOUBLE_EQ(static_cast<double>(c), 5.0);
}

TEST(StrongTypeTest, Multiplication) {
  Meter a{3.0};
  Meter b{4.0};

  Meter c = a * b;

  EXPECT_DOUBLE_EQ(static_cast<double>(c), 12.0);
}

TEST(StrongTypeTest, Division) {
  Meter a{8.0};
  Meter b{2.0};

  Meter c = a / b;

  EXPECT_DOUBLE_EQ(static_cast<double>(c), 4.0);
}

// ------------------------------------------------------------
// Raw Type Arithmetic
// ------------------------------------------------------------

TEST(StrongTypeTest, RawAddition) {
  Meter m{3.0};

  Meter result = m + 2.0;

  EXPECT_DOUBLE_EQ(static_cast<double>(result), 5.0);
}

TEST(StrongTypeTest, RawMultiplication) {
  Meter m{4.0};

  Meter result = m * 2.0;

  EXPECT_DOUBLE_EQ(static_cast<double>(result), 8.0);
}

TEST(StrongTypeTest, CompareWithRaw) {
  Meter t{5.0};

  EXPECT_TRUE(t == 5.0);
  EXPECT_TRUE(t != 3.0);
  EXPECT_TRUE(t < 6.0);
  EXPECT_TRUE(t <= 5.0);
  EXPECT_TRUE(t > 4.0);
  EXPECT_TRUE(t >= 5.0);

  EXPECT_TRUE(5.0 == t);
  EXPECT_TRUE(3.0 != t);
  EXPECT_TRUE(4.0 < t);
  EXPECT_TRUE(5.0 <= t);
  EXPECT_TRUE(6.0 > t);
  EXPECT_TRUE(5.0 >= t);
}

// ------------------------------------------------------------
// Compound Assignment
// ------------------------------------------------------------

TEST(StrongTypeTest, CompoundAdd) {
  Meter m{3.0};

  m += Meter{2.0};

  EXPECT_DOUBLE_EQ(static_cast<double>(m), 5.0);
}

TEST(StrongTypeTest, CompoundRawAdd) {
  Meter m{3.0};

  m += 2.0;

  EXPECT_DOUBLE_EQ(static_cast<double>(m), 5.0);
}

// ------------------------------------------------------------
// Comparison Operators
// ------------------------------------------------------------

TEST(StrongTypeTest, ComparisonOperators) {
  Meter a{3.0};
  Meter b{5.0};
  Meter c{3.0};

  EXPECT_TRUE(a < b);
  EXPECT_TRUE(b > a);
  EXPECT_TRUE(a == c);
  EXPECT_TRUE(a != b);
}

// ------------------------------------------------------------
// Type Safety
// ------------------------------------------------------------

TEST(StrongTypeTest, DifferentTypesAreDistinct) {
  Meter m{5.0};
  Second s{5.0};

  EXPECT_NE(typeid(m), typeid(s));
}

// ------------------------------------------------------------
// Hash Support
// ------------------------------------------------------------

TEST(StrongTypeTest, HashSupport) {
  std::unordered_map<Id, int> map;

  map[Id{5}] = 42;

  EXPECT_EQ(map[Id{5}], 42);
}

// ------------------------------------------------------------
// Stream Operator
// ------------------------------------------------------------

TEST(StrongTypeTest, StreamOperator) {
  Meter m{4.5};

  std::stringstream ss;
  ss << m;

  EXPECT_EQ(ss.str(), "4.5");
}

// ------------------------------------------------------------
// STL Container Compatibility
// ------------------------------------------------------------

TEST(StrongTypeTest, STLContainers) {
  std::vector<Id> ids;

  ids.push_back(Id{1});
  ids.push_back(Id{2});

  EXPECT_EQ(static_cast<int>(ids[0]), 1);
  EXPECT_EQ(static_cast<int>(ids[1]), 2);
}

// ------------------------------------------------------------
// Size Check (Zero-overhead guarantee)
// ------------------------------------------------------------

TEST(StrongTypeTest, SizeMatchesUnderlying) {
  EXPECT_EQ(sizeof(Meter), sizeof(double));
  EXPECT_EQ(sizeof(Id), sizeof(int));
}

// ------------------------------------------------------------
// Constexpr Support
// ------------------------------------------------------------

TEST(StrongTypeTest, ConstexprSupport) {
  constexpr Meter m{3.0};

  constexpr double value = static_cast<double>(m);

  EXPECT_DOUBLE_EQ(value, 3.0);
}

// ------------------------------------------------------------
// Eigen Compatibility
// ------------------------------------------------------------

TEST(StrongTypeTest, EigenVector) {
  Eigen::Matrix<Meter, 3, 1> v;

  v << Meter{1.0}, Meter{2.0}, Meter{3.0};

  EXPECT_DOUBLE_EQ(static_cast<double>(v(0)), 1.0);
  EXPECT_DOUBLE_EQ(static_cast<double>(v(1)), 2.0);
  EXPECT_DOUBLE_EQ(static_cast<double>(v(2)), 3.0);
}

TEST(StrongTypeTest, EigenArithmetic) {
  Eigen::Matrix<Meter, 2, 1> a;
  Eigen::Matrix<Meter, 2, 1> b;

  a << Meter{1}, Meter{2};
  b << Meter{3}, Meter{4};

  auto c = a + b;

  EXPECT_DOUBLE_EQ(static_cast<double>(c(0)), 4.0);
  EXPECT_DOUBLE_EQ(static_cast<double>(c(1)), 6.0);
}
