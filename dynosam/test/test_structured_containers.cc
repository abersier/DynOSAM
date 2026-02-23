/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris
 (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a
 copy
 *   of this software and associated documentation files (the "Software"), to
 deal
 *   in the Software without restriction, including without limitation the
 rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in
 all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE
 *   SOFTWARE.
 */

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <exception>
#include <iterator>
#include <vector>

#include "dynosam_common/StructuredContainers.hpp"
#include "dynosam_common/Types.hpp"

using namespace dyno;
using namespace dyno::internal;

TEST(FilterIteratorTest, FiltersEvenNumbers) {
  std::vector<int> v = {1, 2, 3, 4, 5, 6};

  FilterView view(v, [](int x) { return x % 2 == 0; });

  std::vector<int> result;
  for (int x : view) result.push_back(x);

  std::vector<int> expected = {2, 4, 6};
  EXPECT_EQ(result, expected);
}

TEST(FilterIteratorTest, NoMatches) {
  std::vector<int> v = {1, 3, 5};

  FilterView view(v, [](int x) { return x % 2 == 0; });

  auto it = view.begin();
  EXPECT_EQ(it, view.end());
}

TEST(FilterIteratorTest, EmptyContainer) {
  std::vector<int> v;

  FilterView view(v, [](int) { return true; });

  EXPECT_EQ(view.begin(), view.end());
}

TEST(FilterIteratorTest, PreAndPostIncrement) {
  std::vector<int> v = {1, 2, 3, 4};

  FilterView view(v, [](int x) { return x > 1; });

  auto it = view.begin();

  EXPECT_EQ(*it, 2);

  auto old = it++;
  EXPECT_EQ(*old, 2);
  EXPECT_EQ(*it, 3);

  ++it;
  EXPECT_EQ(*it, 4);

  ++it;
  EXPECT_EQ(it, view.end());
}

TEST(FilterIteratorTest, MultiPass) {
  std::vector<int> v = {1, 2, 3, 4};

  FilterView view(v, [](int x) { return x > 1; });

  auto it1 = view.begin();
  auto it2 = it1;

  EXPECT_EQ(*it1, 2);
  EXPECT_EQ(*it2, 2);

  ++it1;
  EXPECT_EQ(*it1, 3);
  EXPECT_EQ(*it2, 2);  // independent
}

TEST(FilterIteratorTest, Equality) {
  std::vector<int> v = {1, 2, 3, 4};

  FilterView view(v, [](int x) { return x > 2; });

  auto it1 = view.begin();
  auto it2 = view.begin();

  EXPECT_TRUE(it1 == it2);

  ++it1;

  EXPECT_FALSE(it1 == it2);
}

TEST(FilterIteratorTest, ConstContainer) {
  const std::vector<int> v = {1, 2, 3, 4, 5};

  FilterView view(v, [](int x) { return x % 2 == 1; });

  std::vector<int> result;
  for (int x : view) result.push_back(x);

  std::vector<int> expected = {1, 3, 5};
  EXPECT_EQ(result, expected);
}

TEST(FilterIteratorTest, MutationThroughIterator) {
  std::vector<int> v = {1, 2, 3, 4};

  FilterView view(v, [](int x) { return x % 2 == 0; });

  for (int& x : view) x *= 10;

  std::vector<int> expected = {1, 20, 3, 40};
  EXPECT_EQ(v, expected);
}

TEST(FilterIteratorTest, WorksWithStdAlgorithm) {
  std::vector<int> v = {1, 2, 3, 4, 5, 6};

  FilterView view(v, [](int x) { return x % 2 == 0; });

  int sum = std::accumulate(view.begin(), view.end(), 0);

  EXPECT_EQ(sum, 12);  // 2 + 4 + 6
}

TEST(FilterIteratorTest, StdDistanceCountsFilteredElements) {
  std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8};

  FilterView view(v, [](int x) { return x % 2 == 0; });

  auto begin = view.begin();
  auto end = view.end();

  auto dist = std::distance(begin, end);

  EXPECT_EQ(dist, 4);  // 2,4,6,8
}

TEST(FilterIteratorTest, StdDistanceDoesNotCorruptOtherIterators) {
  std::vector<int> v = {1, 2, 3, 4, 5, 6};

  FilterView view(v, [](int x) { return x > 2; });

  auto it1 = view.begin();
  auto it2 = it1;  // copy

  auto d = std::distance(it1, view.end());

  EXPECT_EQ(d, 4);     // 3,4,5,6
  EXPECT_EQ(*it2, 3);  // original iterator still valid
}

/////////// OLD
