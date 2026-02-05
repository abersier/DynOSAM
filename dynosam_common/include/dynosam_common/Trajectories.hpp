#pragma once

#include "dynosam_common/StructuredContainers.hpp"
#include "dynosam_common/Types.hpp"

namespace dyno {

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
  using Data = TData;

  struct Entry {
    FrameId frame_id;
    Timestamp timestamp;
    //! User defined data
    Data data;
  };

  //! Alias for internal trajectory map
  using TrajectoryImpl = gtsam::FastMap<FrameId, Entry>;
  using EntryIterator =
      vector_iterator_base<typename TrajectoryImpl::iterator, Entry>;
  using ConstEntryIterator =
      vector_iterator_base<typename TrajectoryImpl::const_iterator,
                           const Entry>;

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

  TrajectoryBase& insert_or_update(FrameId frame_id, Timestamp timestamp,
                                   const TData& data) {
    if (exists(frame_id)) {
      CHECK(update(frame_id, data));
      return *this;
    }
    return insert(frame_id, timestamp, data);
  }

  bool update(FrameId frame_id, const TData& data) {
    if (exists(frame_id)) {
      trajectory_.at(frame_id).data = data;
      return true;
    }
    return false;
  }

  bool empty() const { return trajectory_.empty(); }
  size_t size() const { return trajectory_.size(); }
  void clear() { trajectory_.clear(); }

  const Entry& get(FrameId frame_id) const { return trajectory_.at(frame_id); }

  Entry& get(FrameId frame_id) { return trajectory_.at(frame_id); }

  const TData& at(FrameId frame_id) const { return get(frame_id).data; }

  TData& at(FrameId frame_id) { return get(frame_id).data; }

  Timestamp timeAt(FrameId frame_id) const { return get(frame_id).timestamp; }

  bool exists(FrameId frame_id) const { return trajectory_.exists(frame_id); }

  const Entry& first() const { return trajectory_.begin()->second; }
  const Entry& last() const { return trajectory_.rbegin()->second; }

  FrameId minFrame() const { return first().frame_id; }
  FrameId maxFrame() const { return last().frame_id; }

  Timestamp startTime() const { return first().timestamp; }
  Timestamp endTime() const { return last().timestamp; }

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
    // use the internal iterator of trajecrory becuase this->begin() returns a
    // forward only iterator
    auto it = trajectory_.find(query.frame_id);
    if (it == trajectory_.end()) {
      return false;
    }

    auto prev_it = std::prev(it);

    if (prev_it == trajectory_.end()) {
      return false;
    }

    previous = prev_it->second;
    return true;
  }

  std::vector<Entry> toVector() const {
    std::vector<Entry> destination;
    std::transform(begin(), end(), std::back_inserter(destination),
                   [](auto entry) { return entry; });
    return destination;
  }

  std::vector<Data> toDataVector() const {
    std::vector<Data> destination;
    std::transform(begin(), end(), std::back_inserter(destination),
                   [](auto entry) { return entry.data; });
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

  EntryIterator begin() { return EntryIterator(trajectory_.begin()); }
  EntryIterator end() { return EntryIterator(trajectory_.end()); }

  ConstEntryIterator begin() const {
    return ConstEntryIterator(trajectory_.begin());
  }
  ConstEntryIterator end() const {
    return ConstEntryIterator(trajectory_.end());
  }

  ConstEntryIterator cbegin() const {
    return ConstEntryIterator(trajectory_.cbegin());
  }
  ConstEntryIterator cend() const {
    return ConstEntryIterator(trajectory_.cend());
  }

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

class PoseTrajectory : public TrajectoryBase<gtsam::Pose3> {
 public:
  PoseTrajectory() : TrajectoryBase<gtsam::Pose3>() {}
};

struct PoseWithMotion {
  gtsam::Pose3 pose;
  Motion3ReferenceFrame motion;
};

class PoseWithMotionTrajectory : public TrajectoryBase<PoseWithMotion> {
 public:
  using TrajectoryBase<PoseWithMotion>::Entry;
  PoseWithMotionTrajectory() : TrajectoryBase<PoseWithMotion>() {}
};

class MultiObjectTrajectories
    : public gtsam::FastMap<ObjectId, PoseWithMotionTrajectory> {
 public:
  using Base = gtsam::FastMap<ObjectId, PoseWithMotionTrajectory>;
  using Base::at;
  using Base::Base;  // all the stl map stuff
  using Base::exists;
  using Base::insert2;

  using Entry = PoseWithMotionTrajectory::Entry;
  using EntryMap = gtsam::FastMap<ObjectId, Entry>;

  MultiObjectTrajectories() = default;

  // this is basically copying everything over and is therefore slow
  // only needed for backwards compatability
  MultiObjectTrajectories(const ObjectPoseMap& poses,
                          const ObjectMotionMap& motion,
                          const FrameIdTimestampMap& times);

  void insert(ObjectId object_id, FrameId frame_id, Timestamp timestamp,
              const PoseWithMotion& data);

  bool hasObject(ObjectId object_id) const;
  size_t numObjects() const;
  ObjectIds objectIds() const;

  bool hasFrame(ObjectId object_id, FrameId frame_id) const;
  EntryMap atFrame(FrameId frame_id) const;

  // Slow - mostly needed for backwards compatability
  ObjectPoseMap toObjectPoseMap() const;
  ObjectMotionMap toObjectMotionMap() const;
};

}  // namespace dyno
