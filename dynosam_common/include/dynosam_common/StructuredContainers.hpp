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

#pragma once

#include <glog/logging.h>
#include <gtsam/base/FastDefaultAllocator.h>

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/Numerical.hpp"  //for hash pair

template <class T1, class T2>
using TemplatedPair = std::pair<T1, T2>;

template <class T1, class T2>
struct std::hash<TemplatedPair<T1, T2>> {
  using TP = TemplatedPair<T1, T2>;

  inline std::size_t operator()(const TP& k) const { return dyno::hashPair(k); }
};

namespace dyno {

/**
 * @brief Internal iterator type allowing iteration over a map directly as if it
 * were a vector. This type satisfies constraints for a filter_iterator_base as
 * well as an std::iterator
 *
 * @tparam MapIterator The internal iterator to use.
 * @tparam MappedType The type we are iterating over.
 */
template <typename MapIterator, typename MappedType>
struct vector_iterator_base {
  using iterator_type = MapIterator;

  using value_type = MappedType;
  using reference = value_type&;
  using pointer = value_type*;

  iterator_type it_;
  vector_iterator_base(iterator_type it) : it_(it) {}

  reference operator*() { return it_->second; }
  reference operator->() { return it_->second; }

  bool operator==(const vector_iterator_base& other) const {
    return it_ == other.it_;
  }
  bool operator!=(const vector_iterator_base& other) const {
    return it_ != other.it_;
  }

  bool operator==(const iterator_type& other) const { return it_ == other; }
  bool operator!=(const iterator_type& other) const { return it_ != other; }

  vector_iterator_base& operator++() {
    ++it_;
    return *this;
  }
};

namespace internal {

// T is the pointer to the type that is expected to be iterated over
// such that we have struct iterator_traits<T*> or struct iterator_traits<const
// T*>; see https://en.cppreference.com/w/cpp/iterator/iterator_traits

// usually defined as filter_iterator_detail<Container::pointer> could also be
// <filter_iterator_detail<Iter::point>
template <typename T>
struct filter_iterator_detail {
  //! naming conventions to match those required by iterator traits
  using value_type = typename std::iterator_traits<T>::value_type;
  using reference_type = typename std::iterator_traits<T>::reference;
  using pointer = typename std::iterator_traits<T>::pointer;
  using difference_type = typename std::iterator_traits<T>::difference_type;
  using iterator_category =
      std::forward_iterator_tag;  // i guess? only forward is defined (++iter)
                                  // right now
};

// container is the actual container type and Iter is the iterator type (eg
// container::iterator or container::const_iterator) in this way,
// filter_iterator can act as either a regular iterator or a const iterator we
// expect container to also be iterable ref type should also be
// BaseDetail::reference_type? check to ensure container types and iter types
// are the same? container must at least contain (const and non const versions
// of end and begin which must return actual iterators which satisfy the forward
// iterator category) container must also at least have the definitions for
// Container::iterator, Container::const_iterator, Container::const_reference
// and Iter must be a properly defined iterator such that it satisfies all the
// conditions for a forward_iterator (e.f Iter::value_type, Iter::reference
// etc...)
template <typename Container, typename Iter,
          typename FilterFunction =
              std::function<bool(typename Container::const_reference)>>
struct filter_iterator_base
    : public filter_iterator_detail<typename Iter::pointer> {
 public:
  using BaseDetail = filter_iterator_detail<typename Iter::pointer>;
  using iterator = Iter;
  using container = Container;
  using typename BaseDetail::difference_type;
  using typename BaseDetail::iterator_category;
  using typename BaseDetail::pointer;
  using typename BaseDetail::reference_type;
  using typename BaseDetail::value_type;

  filter_iterator_base(Container& container, const FilterFunction& filter_func)
      : filter_iterator_base(container, filter_func, container.begin()) {}

  reference_type operator*() { return *it_; }
  reference_type operator->() { return *it_; }

  bool operator==(const filter_iterator_base& other) const {
    return it_ == other.it_;
  }
  bool operator!=(const filter_iterator_base& other) const {
    return it_ != other.it_;
  }

  bool operator==(const iterator& other) const { return it_ == other; }
  bool operator!=(const iterator& other) const { return it_ != other; }

  Container& getContainer() { return container_; }
  const Container& getContainer() const { return container_; }

  bool operator()(typename Container::const_reference arg) const {
    return filter_func_(arg);
  }

  // preincrement (++iter)
  filter_iterator_base& operator++() {
    do {
      ++it_;
    } while (is_invalid());
    return *this;
  }

  // TODO??
  filter_iterator_base& operator++(int x) {
    do {
      it_ += x;
    } while (is_invalid());
    return *this;
  }

  // allows the iterator to be used as a enhanced for loop
  filter_iterator_base begin() {
    return filter_iterator_base(container_, filter_func_, container_.begin());
  }
  filter_iterator_base end() {
    return filter_iterator_base(container_, filter_func_, container_.end());
  }

  const filter_iterator_base begin() const {
    return filter_iterator_base(container_, filter_func_, container_.begin());
  }
  const filter_iterator_base end() const {
    return filter_iterator_base(container_, filter_func_, container_.end());
  }

 private:
  bool is_invalid() const {
    return it_ != container_.end() && !filter_func_(*it_);
  }

  void find_next_valid() {
    while (is_invalid()) {
      this->operator++();
    }
  }

 protected:
  filter_iterator_base(Container& container, const FilterFunction& filter_func,
                       iterator it)
      : container_(container), filter_func_(filter_func), it_(it) {
    find_next_valid();
  }

 protected:
  Container& container_;  //! reference to container
  FilterFunction filter_func_;
  mutable iterator it_;
};

template <typename Container>
using filter_iterator =
    filter_iterator_base<Container, typename Container::iterator>;

template <typename Container>
using filter_const_iterator =
    filter_iterator_base<Container, typename Container::const_iterator>;

}  // namespace internal

struct FrameRangeBase {
  FrameId start;
  FrameId end;
  //! indicates that this is the current keyframe range for the latest frame id
  //! and therefore the end value is not valid
  bool is_active = false;

  FrameRangeBase() {}
  FrameRangeBase(FrameId start_, FrameId end_, bool is_active_ = false)
      : start(start_), end(end_), is_active(is_active_) {}

  bool contains(FrameId frame_id) const;
  bool operator<(const FrameRangeBase& rhs) const { return start < rhs.start; }
};

template <typename T>
struct FrameRange : public FrameRangeBase {
  using This = FrameRange<T>;
  DYNO_POINTER_TYPEDEFS(This)

  FrameRange() {}
  FrameRange(FrameId start_, FrameId end_, const T& data_,
             bool is_active_ = false)
      : FrameRangeBase(start_, end_, is_active_), data(data_) {}

  T data;
  std::pair<FrameId, T> dataPair() const { return {start, data}; }
};

template <typename T>
class FrameRangeData {
 public:
  using This = FrameRangeData<T>;
  using FrameRangeT = FrameRange<T>;
  using FrameRangeTVector = std::set<typename FrameRangeT::Ptr>;

  const typename FrameRangeT::ConstPtr find(FrameId frame_id) const {
    typename FrameRangeT::Ptr active_range = getActiveRange();
    if (!active_range) {
      return nullptr;
    }

    // sanity check
    CHECK(active_range->is_active);
    if (active_range->contains(frame_id)) {
      return active_range;
    } else {
      CHECK_GE(ranges.size(), 1u);
      // iterate over ranges
      for (const typename FrameRangeT::Ptr& range : ranges) {
        if (range->contains(frame_id)) {
          return range;
        }
      }
    }
    return nullptr;
  }

  const typename FrameRangeT::ConstPtr startNewActiveRange(FrameId frame_id,
                                                           const T& data) {
    typename FrameRangeT::Ptr old_active_range = getActiveRange();
    auto new_range = std::make_shared<FrameRangeT>();
    new_range->start = frame_id;
    // dont set end (yet) but make active
    new_range->is_active = true;
    new_range->data = data;

    return updateRanges(new_range, old_active_range);
  }

  auto begin() { return ranges.begin(); }
  auto end() { return ranges.end(); }

  auto begin() const { return ranges.begin(); }
  auto end() const { return ranges.end(); }

  typename FrameRangeT::Ptr getActiveRange() const {
    if (!active_range) {
      // no range means no data
      CHECK_EQ(ranges.size(), 0u);
      return nullptr;
    }
    CHECK(active_range->is_active);
    return active_range;
  }

 private:
  typename FrameRangeT::Ptr updateRanges(
      typename FrameRangeT::Ptr new_range,
      typename FrameRangeT::Ptr old_active_range = nullptr) {
    CHECK_NOTNULL(new_range);
    if (old_active_range) {
      // modify existing range so that the end is the start of the next (new
      // range)
      old_active_range->end = new_range->start;
      old_active_range->is_active = false;
    }

    // ranges.push_back(new_range);
    ranges.insert(new_range);
    // set new active range
    active_range = new_range;
    return new_range;
  }

 private:
  FrameRangeTVector ranges;
  typename FrameRangeT::Ptr active_range;
};

template <typename V, typename T>
class MultiFrameRangeData {
 public:
  using This = MultiFrameRangeData<V, T>;
  using FrameRangeT = FrameRange<T>;
  using FrameRangeDataTVector = FrameRangeData<T>;

  MultiFrameRangeData() {}

  const typename FrameRangeT::ConstPtr find(V query, FrameId frame_id) const {
    if (!ranges.exists(query)) {
      return nullptr;
    }

    return ranges.at(query).find(frame_id);
  }
  const typename FrameRangeT::ConstPtr startNewActiveRange(V query,
                                                           FrameId frame_id,
                                                           const T& data) {
    if (!ranges.exists(query)) {
      ranges.insert2(query, FrameRangeDataTVector{});
    }

    return ranges.at(query).startNewActiveRange(frame_id, data);
  }

  bool exists(V query) const { return ranges.exists(query); }

  const FrameRangeDataTVector& at(V query) const {
    if (!ranges.exists(query)) {
      DYNO_THROW_MSG(std::out_of_range) << "Query " << query << " out of range";
    }

    return ranges.at(query);
  }

 private:
  gtsam::FastMap<V, FrameRangeDataTVector> ranges;
};

/**
 * FastUnorderedMap is a thin wrapper around std::unordered that uses the boost
 * fast_pool_allocator instead of the default STL allocator.  This is just a
 * convenience to avoid having lengthy types in the code.  Through timing,
 * we've seen that the fast_pool_allocator can lead to speedups of several
 * percent.
 * @ingroup base
 */
template <typename KEY, typename VALUE>
class FastUnorderedMap
    : public std::unordered_map<KEY, VALUE, std::hash<KEY>, std::equal_to<KEY>,
                                typename gtsam::internal::FastDefaultAllocator<
                                    std::pair<const KEY, VALUE>>::type> {
 public:
  typedef std::unordered_map<KEY, VALUE, std::hash<KEY>, std::equal_to<KEY>,
                             typename gtsam::internal::FastDefaultAllocator<
                                 std::pair<const KEY, VALUE>>::type>
      Base;

  /** Default constructor */
  FastUnorderedMap() {}

  /** Constructor from a range, passes through to base class */
  template <typename INPUTITERATOR>
  explicit FastUnorderedMap(INPUTITERATOR first, INPUTITERATOR last)
      : Base(first, last) {}

  /** Copy constructor from another FastUnorderedMap */
  FastUnorderedMap(const FastUnorderedMap<KEY, VALUE>& x) : Base(x) {}

  /** Copy constructor from the base map class */
  FastUnorderedMap(const Base& x) : Base(x) {}

  /** Move constructor */
  FastUnorderedMap(FastUnorderedMap&& x) noexcept = default;

  /** Copy assignment */
  FastUnorderedMap& operator=(const FastUnorderedMap& x) = default;

  /** Move assignment */
  FastUnorderedMap& operator=(FastUnorderedMap&& x) noexcept = default;

  /** Conversion to a standard STL container */
  operator std::unordered_map<KEY, VALUE>() const {
    return std::unordered_map<KEY, VALUE>(this->begin(), this->end());
  }

  /** Handy 'insert' function for Matlab wrapper */
  bool insert2(const KEY& key, const VALUE& val) {
    return Base::insert(std::make_pair(key, val)).second;
  }

  /** Handy 'exists' function */
  bool exists(const KEY& e) const { return this->find(e) != this->end(); }

 private:
};

class TrajectoryEntryAlreadyExists : public DynosamException {
 public:
  TrajectoryEntryAlreadyExists(const FrameId& frame_id)
      : DynosamException("Trajectory entry already exists at k=" +
                         std::to_string(frame_id)) {}
};

template <typename TData>
class TrajectoryBase {
 public:
  using This = TrajectoryBase<TData>;

  struct Entry {
    FrameId frame_id;
    Timestamp timestamp;
    //! User defined data
    TData data;
  };

  //! Alias for internal trajectory map
  using TrajectoryImpl = gtsam::FastMap<FrameId, Entry>;
  using EntryIterator = vector_iterator_base<TrajectoryImpl, Entry>;

  struct Segment {
    const FrameId start_frame{0};
    const FrameId end_frame{0};
    TrajectoryBase trajectory;

    Segment(FrameId s_frame, FrameId e_frame,
            const TrajectoryBase& trajectory_segment)
        : start_frame(s_frame),
          end_frame(e_frame),
          trajectory(trajectory_segment) {}

    static Segment fromTrajectory(const TrajectoryBase& trajectory_segment) {
      if (trajectory_segment.empty()) {
        return Segment{0, 0, {}};
      }

      return Segment(trajectory_segment.first().frame_id,
                     trajectory_segment.last().frame_id, trajectory_segment);
    }

    bool check() const {
      if (trajectory.first().frame_id != start_frame) {
        return false;
      }
      if (trajectory.last().frame_id != end_frame) {
        return false;
      }

      if (trajectory.empty()) {
        return true;
      }

      // Check increasing trajectory
      auto it = trajectory.begin();
      auto prev_it = it;
      ++it;

      bool is_continuous = true;
      for (; it != trajectory.end(); ++it) {
        // Check continuity
        if (it->frame_id != prev_it->frame_id + 1) {
          is_continuous = false;
        }
        prev_it = it;
      }
      return is_continuous;
    }
  };

  TrajectoryBase() : trajectory_() {}
  virtual ~TrajectoryBase() = default;

  TrajectoryBase& insert(FrameId frame_id, Timestamp timestamp,
                         const TData& data) {
    return insert(Entry{frame_id, timestamp, data});
  }

  bool update(FrameId frame_id, const TData& data) {
    if (!exists(frame_id)) {
      trajectory_.at(frame_id).data = data;
      return true;
    }
    return false;
  }

  bool empty() const { return trajectory_.empty(); }
  size_t size() const { return trajectory_.size(); }
  void clear() { trajectory_.clear(); }

  const TData& at(FrameId frame_id) const {
    return trajectory_.at(frame_id).data;
  }
  TData& at(FrameId frame_id) { return trajectory_.at(frame_id).data; }
  Timestamp timeAt(FrameId frame_id) const {
    return trajectory_.at(frame_id).timestamp;
  }

  bool exists(FrameId frame_id) const { return trajectory_.exists(frame_id); }

  const Entry& first() const { return trajectory_.begin()->second; }
  const Entry& last() const { return trajectory_.rbegin()->second; }

  /**
   * @brief Get the previous entry in the trajectory.
   *
   * Entry may not be frame_id - 1 if trajectory has multiple segments
   *
   * @param previous
   * @param query
   * @return true
   * @return false
   */
  bool getPrevious(Entry& previous, const Entry& query) const {
    auto it = trajectory_.find(query.frame_id);
    if (it == trajectory_.end()) {
      return false;
    }

    auto prev_it = it;
    std::prev(prev_it);

    if (prev_it == trajectory_.end()) {
      return false;
    }

    previous = *prev_it;
    return true;
  }

  std::vector<Entry> toVector() const {
    std::vector<Entry> destination;
    std::transform(begin(), end(), std::back_inserter(destination),
                   [](auto entry) { return entry; });
    return destination;
  }

  std::vector<Segment> segments() const {
    std::vector<Segment> segments;

    if (trajectory_.empty()) return segments;

    auto it = trajectory_.begin();
    This current_segment;
    // Initalise from entry
    current_segment.insert(it->second);

    auto prev_it = it;
    ++it;

    for (; it != trajectory_.end(); ++it) {
      // Check continuity
      if (it->first == prev_it->first + 1) {
        current_segment.insert(*it);
      } else {
        // Break in trajectory
        segments.push_back(Segment::fromTrajectory(current_segment));
        current_segment.clear();
        current_segment.insert(it->second);
      }
      prev_it = it;
    }

    // Push final segment
    segments.push_back(Segment::fromTrajectory(current_segment));
    return segments;
  }

  EntryIterator begin() const { return EntryIterator(trajectory_.begin()); }
  EntryIterator end() const { return EntryIterator(trajectory_.end()); }

 protected:
  TrajectoryBase(const TrajectoryImpl& trajectory) : trajectory_(trajectory) {}
  TrajectoryBase& insert(const Entry& entry) {
    if (exists(entry.frame_id)) {
      throw TrajectoryEntryAlreadyExists(entry.frame_id);
    }

    trajectory_.insert2(entry.frame_id, entry);
    return *this;
  }
  // for inserting from iterator
  TrajectoryBase& insert(const std::pair<FrameId, Entry>& entry_pair) {
    CHECK_EQ(entry_pair.first, entry_pair.second.frame_id);
    return this->insert(entry_pair.second);
  }

 private:
  TrajectoryImpl trajectory_;
};

}  // namespace dyno

// allow convenience tuple like getters for FrameRange
namespace std {
template <typename T>
struct tuple_size<dyno::FrameRange<T>> : std::integral_constant<size_t, 2> {};

template <typename T>
struct tuple_element<0, dyno::FrameRange<T>> {
  using type = dyno::FrameId;
};

template <typename T>
struct tuple_element<1, dyno::FrameRange<T>> {
  using type = T;
};
}  // namespace std

namespace dyno {
template <size_t N, typename T>
auto get(const FrameRange<T>& p) {
  if constexpr (N == 0)
    return p.dataPair().first;
  else if constexpr (N == 1)
    return p.dataPair().second;
}
}  // namespace dyno
