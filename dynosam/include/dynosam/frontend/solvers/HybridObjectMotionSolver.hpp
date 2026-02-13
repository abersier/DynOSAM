#pragma once

#include "dynosam/frontend/Frontend-Definitions.hpp"
#include "dynosam/frontend/solvers/HybridObjectMotionSRIF.hpp"
#include "dynosam/frontend/solvers/ObjectMotionSolver.hpp"
#include "dynosam/frontend/solvers/OpticalFlowAndPoseSolver.hpp"
#include "dynosam/frontend/solvers/PnPRansac.hpp"
#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam_common/GroundTruthPacket.hpp"
#include "dynosam_cv/Camera.hpp"

namespace dyno {

struct HybridObjectMotionSolverParams {
  PnPRansacSolverParams pnp_ransac_params;
  OpticalFlowAndPoseSolverParams optical_flow_solver_params;
};

class HybridObjectMotionSolver : public ObjectMotionSolver {
 public:
  DYNO_POINTER_TYPEDEFS(HybridObjectMotionSolver)

  HybridObjectMotionSolver(const HybridObjectMotionSolverParams& params,
                           const CameraParams& camera_params,
                           const SharedGroundTruth& shared_ground_truth = {});

  MultiObjectTrajectories solve(Frame::Ptr frame_k, Frame::Ptr frame_km1,
                                bool parallel_solve = true) override;

  ObjectTrackingStatus getTrackingStatus(ObjectId object_id) const {
    return object_statuses_.at(object_id);
  }

  ObjectKeyFrameStatus getKeyFrameStatus(ObjectId object_id) const {
    return object_keyframe_statuses_.at(object_id);
  }

  const gtsam::FastMap<ObjectId, HybridObjectMotionSRIF::Ptr>& getFilters()
      const {
    return filters_;
  }

 protected:
  bool solveImpl(Frame::Ptr frame_k, Frame::Ptr frame_km1, ObjectId object_id,
                 Motion3ReferenceFrame& motion_estimate) override;

  void updateTrajectories(MultiObjectTrajectories& object_trajectories,
                          const MotionEstimateMap& motion_estimates,
                          Frame::Ptr frame_k, Frame::Ptr frame_km1) override;

 private:
  bool filterNeedsReset(ObjectId object_id);

  gtsam::Pose3 constructPoseFromCentroid(const Frame::Ptr frame,
                                         const TrackletIds& tracklets) const;

  HybridObjectMotionSRIF::Ptr createAndInsertFilter(
      ObjectId object_id, Frame::Ptr frame, const TrackletIds& tracklets);

 private:
  HybridObjectMotionSolverParams params_;
  PnPRansacSolver pnp_ransac_solver_;
  OpticalFlowAndPoseSolver<Camera::CalibrationType> optical_flow_pose_solver_;
  const SharedGroundTruth shared_ground_truth_;

  MultiObjectTrajectories object_trajectories_;

  gtsam::FastMap<ObjectId, HybridObjectMotionSRIF::Ptr> filters_;

 private:
  gtsam::FastMap<ObjectId, ObjectTrackingStatus> object_statuses_;
  //! If filter needs resetting from last frame
  gtsam::FastMap<ObjectId, bool> filter_needs_reset_;
  gtsam::FastMap<ObjectId, ObjectKeyFrameStatus> object_keyframe_statuses_;
};

}  // namespace dyno

// void declare_config(OpticalFlowAndPoseOptimizer::Params& config);
// void declare_config(MotionOnlyRefinementOptimizer::Params& config);

// void declare_config(EgoMotionSolver::Params& config);
// void declare_config(ConsecutiveFrameObjectMotionSolver::Params& config);
