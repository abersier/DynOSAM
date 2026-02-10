#pragma once

#include "dynosam/backend/PoseChangeBackendModule.hpp"
#include "dynosam/backend/rgbd/HybridEstimator.hpp"
#include "dynosam/frontend/VIFrontend.hpp"
#include "dynosam/frontend/vision/MotionSolver.hpp"
#include "dynosam_cv/RGBDCamera.hpp"

namespace dyno {

using PoseChangeBackendSink =
    std::function<void(const PoseChangeInput::ConstPtr&)>;

class PoseChangeVIFrontend : public VIFrontend {
 public:
  DYNO_POINTER_TYPEDEFS(PoseChangeVIFrontend)
  PoseChangeVIFrontend(const DynoParams& params, Camera::Ptr camera,
                       HybridFormulationKeyFrame::Ptr formulation,
                       ImageDisplayQueue* display_queue = nullptr,
                       const SharedGroundTruth& shared_ground_truth = {});

  void addPoseChangeOutputSink(const PoseChangeBackendSink& func) {
    pose_change_backend_sink_ = func;
  };

 private:
  SpinReturn boostrapSpin(FrontendInputPacketBase::ConstPtr input) override;
  SpinReturn nominalSpin(FrontendInputPacketBase::ConstPtr input) override;

  void solveObjectMotions(MultiObjectTrajectories& trajectories,
                          ObjectPoseChangeInfoMap& infos, Frame::Ptr frame_k,
                          Frame::Ptr frame_km1);

 private:
  HybridFormulationKeyFrame::Ptr formulation_;
  MapVision::Ptr map_;
  std::unique_ptr<ObjectMotionSolverFilter> object_motion_solver_;

  gtsam::NavState nav_state_km1_;
  gtsam::NavState nav_state_lkf_;
  //! The relative camera pose (T_k_1_k) from the previous frame
  //! this is used as a constant velocity model when VO tracking fails and the
  //! IMU is not available!
  gtsam::Pose3 T_km1_k_;
  gtsam::Pose3 T_lkf_k_;

  //! Current trajectories. Copied to the DynoState output
  DynoStateTrajectories dyno_state_;

  PoseChangeBackendSink pose_change_backend_sink_;
};

}  // namespace dyno
