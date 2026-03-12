#pragma once

#include "dynosam_common/Types.hpp"

/**
 * Common definitions for the PoseChange/MotionKeyFrame modules
 *
 */

namespace dyno {

enum class ObjectKeyFrameStatus {
  NonKeyFrame = 0,
  RegularKeyFrame = 1,
  AnchorKeyFrame = 2
};

struct ObjectPoseChangeInfo {
  FrameId frame_id;

  StatusLandmarkVector initial_object_points;
  //! Associated keyframe
  //! if keyframe then this value is NEW (ie changed from the previous one)
  //! and the initial motion should be identity
  gtsam::Pose3 L_W_KF;
  //! This is the preintegrated motion immediately before the current
  //! keyframe at k
  Motion3ReferenceFrame H_W_KF_k;
  gtsam::Pose3 L_W_k;

  ObjectKeyFrameStatus keyframe_status{ObjectKeyFrameStatus::NonKeyFrame};

  // // make intermediate keyframe to optimise w.r.t to the same anchor point
  // // ie. indicates if a motion variable should be added this frame
  // bool regular_keyframe{false};
  // // make a new anchor point for the object
  // // this happens when the object is new or has re-appeared (and therefore
  // // has no contuous tracks) in this case a regular keyframe MUST also be
  // // made a motion will added this frame AND the anchor pose will be updated
  // bool anchor_keyframe{false};

  bool isKeyFrame() const {
    return keyframe_status != ObjectKeyFrameStatus::NonKeyFrame;
  }
};

using ObjectPoseChangeInfoMap = gtsam::FastMap<ObjectId, ObjectPoseChangeInfo>;

std::ostream& operator<<(std::ostream& os, const ObjectKeyFrameStatus& status);

}  // namespace dyno
