#pragma once

#include "dynosam/backend/rgbd/HybridEstimator.hpp"  //for pose change info
#include "dynosam/frontend/Frontend-Definitions.hpp"
#include "dynosam/frontend/solvers/HybridObjectMotionSRIF.hpp"
#include "dynosam/frontend/solvers/HybridObjectMotionSmoother.hpp"
#include "dynosam/frontend/solvers/HybridObjectMotionSolver-Impl.hpp"
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

  void solve(Frame::Ptr frame_k, Frame::Ptr frame_km1,
             MultiObjectTrajectories& trajectories_out,
             MotionEstimateMap& motion_estimate_out,
             bool parallel_solve = true) override;

  bool getObjectStructureinL(ObjectId object_id,
                             StatusLandmarkVector& object_points) const;
  bool getObjectStructureinW(ObjectId object_id,
                             StatusLandmarkVector& object_points) const;

  const auto& getFilters() const { return solvers_; }

  const ObjectPoseChangeInfoMap& poseChangeInfoMap() const {
    return pose_change_info_;
  }

 protected:
  bool solveImpl(Frame::Ptr frame_k, Frame::Ptr frame_km1, ObjectId object_id,
                 Motion3ReferenceFrame& motion_estimate) override;

  void updateTrajectories(MultiObjectTrajectories& object_trajectories,
                          const MotionEstimateMap& motion_estimates,
                          Frame::Ptr frame_k, Frame::Ptr frame_km1) override;

 private:
  bool filterNeedsReset(ObjectId object_id);

  // may be from centroid or gronud truth depending on availablility
  gtsam::Pose3 constructObjectPose(const ObjectId object_id,
                                   const Frame::Ptr frame,
                                   const TrackletIds& tracklets) const;

  std::optional<gtsam::Pose3> objectPoseFromGroundTruth(
      ObjectId object_id, const Frame::Ptr frame) const;

  gtsam::Pose3 objectPoseFromCentroid(const ObjectId object_id,
                                      const Frame::Ptr frame,
                                      const TrackletIds& tracklets) const;

  HybridObjectMotionSolverImpl::Ptr createAndInsertFilter(
      ObjectId object_id, Frame::Ptr frame, const TrackletIds& tracklets);
  // HybridObjectMotionSmoother::Ptr createAndInsertFilter(
  //     ObjectId object_id, Frame::Ptr frame, const TrackletIds& tracklets);

  void deleteObject(ObjectId object_id);

  HybridObjectMotionSolverImpl::Ptr threadSafeFilterAccess(
      ObjectId object_id) const;

  // std::optional<int> getNumKeyFramesPerObject(ObjectId object_id) const;
  // void setObjectKeyFrameStatus(ObjectId object_id, ObjectKeyFrameStatus
  // status);

 private:
  HybridObjectMotionSolverParams params_;
  PnPRansacSolver pnp_ransac_solver_;
  OpticalFlowAndPoseSolver<Camera::CalibrationType> optical_flow_pose_solver_;
  const SharedGroundTruth shared_ground_truth_;

  MultiObjectTrajectories object_trajectories_;

  gtsam::FastMap<ObjectId, HybridObjectMotionSolverImpl::Ptr> solvers_;
  // Info from the last frame. ONly stores change info with keyframes
  gtsam::FastMap<ObjectId, ObjectPoseChangeInfo> pose_change_info_;

 private:
  gtsam::FastMap<ObjectId, ObjectTrackingStatus> object_statuses_;
  // //! If filter needs resetting from last frame
  gtsam::FastMap<ObjectId, bool> filter_needs_reset_;
  gtsam::FastMap<ObjectId, gtsam::Pose3> filter_needs_reset_init_KF_;
  // Need this only for previous state!
  gtsam::FastMap<ObjectId, ObjectKeyFrameStatus> object_keyframe_statuses_;
  gtsam::FastMap<ObjectId, int> num_kfs_per_object_;

  mutable std::mutex object_status_mutex_;
  mutable std::mutex keyframe_status_mutex_;
  mutable std::mutex num_kfs_per_object_mutex_;
  mutable std::mutex solvers_mutex_;
};

}  // namespace dyno

// void declare_config(OpticalFlowAndPoseOptimizer::Params& config);
// void declare_config(MotionOnlyRefinementOptimizer::Params& config);

// void declare_config(EgoMotionSolver::Params& config);
// void declare_config(ConsecutiveFrameObjectMotionSolver::Params& config);
