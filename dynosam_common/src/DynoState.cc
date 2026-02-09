#include "dynosam_common/DynoState.hpp"

namespace dyno {
gtsam::Pose3 DynoState::cameraPose() const {
  return camera_trajectory.at(this->frame_id);
}
MotionEstimateMap DynoState::objectMotions() const {
  LOG(FATAL) << "Not implemented!";
}

}  // namespace dyno
