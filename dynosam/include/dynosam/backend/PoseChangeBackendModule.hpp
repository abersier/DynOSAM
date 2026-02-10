#pragma once

#include <gtsam/nonlinear/ISAM2.h>

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/backend/BackendModule.hpp"
#include "dynosam/backend/Formulation.hpp"
#include "dynosam/backend/rgbd/HybridEstimator.hpp"

namespace dyno {

struct PoseChangeInput {
  DYNO_POINTER_TYPEDEFS(PoseChangeInput)

  gtsam::Values new_values;
  gtsam::NonlinearFactorGraph new_factors;
};

struct ObjectPoseChangeInfo {
  FrameId frame_id;

  ObjectTrackingStatus motion_track_status;

  StatusLandmarkVector initial_object_points;
  //! Associated keyframe
  //! if keyframe then this value is NEW (ie changed from the previous one)
  //! and the initial motion should be identity
  gtsam::Pose3 L_W_KF;
  //! This is the preintegrated motion immediately before the current
  //! keyframe at k
  Motion3ReferenceFrame H_W_KF_k;
  gtsam::Pose3 L_W_k;

  // make intermediate keyframe to optimise w.r.t to the same anchor point
  // ie. indicates if a motion variable should be added this frame
  bool regular_keyframe{false};
  // make a new anchor point for the object
  // this happens when the object is new or has re-appeared (and therefore
  // has no contuous tracks) in this case a regular keyframe MUST also be
  // made a motion will added this frame AND the anchor pose will be updated
  bool anchor_keyframe{false};

  bool isKeyFrame() const { return regular_keyframe || anchor_keyframe; }
};

using ObjectPoseChangeInfoMap = gtsam::FastMap<ObjectId, ObjectPoseChangeInfo>;

class PoseChangeVIBackendModule : public BackendModuleV1<PoseChangeInput> {
 public:
  using Base = BackendModuleV1<PoseChangeInput>;
  DYNO_POINTER_TYPEDEFS(PoseChangeVIBackendModule)

  PoseChangeVIBackendModule(const BackendParams& params, Camera::Ptr camera,
                            HybridFormulationKeyFrame::Ptr formulation);

  std::pair<gtsam::Values, gtsam::NonlinearFactorGraph> getActiveOptimisation()
      const override {
    LOG(FATAL) << "Not implemented!";
  }

  Accessor::Ptr getAccessor() const override {
    return formulation_->getAsVIOAccessor();
  }
  HybridFormulationKeyFrame::Ptr getFormulation() const { return formulation_; }

 private:
  using SpinReturn = Base::SpinReturn;

  // just call spin once
  SpinReturn boostrapSpin(PoseChangeInput::ConstPtr input) override {
    return {State::Nominal, spinOnce(input)};
  }
  SpinReturn nominalSpin(PoseChangeInput::ConstPtr input) override {
    return {State::Nominal, spinOnce(input)};
  }

  DynoState::Ptr spinOnce(PoseChangeInput::ConstPtr input);

 private:
  HybridFormulationKeyFrame::Ptr formulation_;
  ErrorHandlingHooks error_hooks_;

  std::unique_ptr<gtsam::ISAM2> smoother_;
};

}  // namespace dyno
