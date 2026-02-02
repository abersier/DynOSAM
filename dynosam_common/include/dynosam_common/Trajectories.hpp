#pragma once

#include "dynosam_common/StructuredContainers.hpp"
#include "dynosam_common/Types.hpp"

namespace dyno {

class PoseTrajectory : public TrajectoryBase<gtsam::Pose3> {
 public:
  PoseTrajectory() : TrajectoryBase<gtsam::Pose3>() {}
};

struct PoseWithMotion {
  gtsam::Pose3 pose;
  MotionReferenceFrame motion;
};

class PoseWithMotionTrajectory : TrajectoryBase<PoseWithMotion> {
 public:
  PoseWithMotionTrajectory() : TrajectoryBase<PoseWithMotion>() {}
};

}  // namespace dyno
