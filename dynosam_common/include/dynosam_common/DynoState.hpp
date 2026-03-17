#pragma once
#include "dynosam_common/Trajectories.hpp"

namespace dyno {

struct DynoStateTrajectories {
  // trajectories
  PoseTrajectory camera_trajectory;
  MultiObjectTrajectories object_trajectories;
};

struct DynoState : public DynoStateTrajectories {
  DYNO_POINTER_TYPEDEFS(DynoState)

  //! Must be filled or we get weird behaviour in visualisation
  FrameId frame_id;
  Timestamp timestamp;

  // if nothing in the trajectory this segfaults :/
  gtsam::Pose3 cameraPose() const;
  MotionEstimateMap objectMotions() const;

  // maps
  StatusLandmarkVector local_static_map;
  StatusLandmarkVector global_static_map;
  StatusLandmarkVector dynamic_map;
};

}  // namespace dyno
