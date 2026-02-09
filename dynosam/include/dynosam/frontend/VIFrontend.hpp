#pragma once

#include "dynosam/frontend/FrontendInputPacket.hpp"
#include "dynosam/frontend/RGBDInstance-Definitions.hpp"
#include "dynosam/frontend/imu/ImuFrontend.hpp"
#include "dynosam/frontend/vision/FeatureTracker.hpp"
#include "dynosam/pipeline/PipelineParams.hpp"
#include "dynosam_common/DynoState.hpp"
#include "dynosam_common/ModuleBase.hpp"
#include "dynosam_common/Trajectories.hpp"

namespace dyno {

// TODO: Original is in FrontendModule which can be deleted once refactoring
struct InvalidImageContainerException1 : public DynosamException {
  InvalidImageContainerException1(const ImageContainer& container,
                                  const std::string& what)
      : DynosamException("Image container with config: " +
                         container.toString() + "\n was invalid - " + what) {}
};

struct RealtimeOutput {
  DYNO_POINTER_TYPEDEFS(RealtimeOutput)
  //! Current state data
  DynoState state;
  //! Possible dense point cloud (with label and RGB) in camera frame
  PointCloudLabelRGB::Ptr dense_labelled_cloud;
  //! Debug/visualiation imagery for this frame. Internal data may be empty
  DebugImagery debug_imagery;
  //! Optional ground truth information for this frame
  GroundTruthInputPacket::Optional ground_truth;
};

class Frontend : public ModuleBase<FrontendInputPacketBase, RealtimeOutput> {
 public:
  using Base = ModuleBase<FrontendInputPacketBase, RealtimeOutput>;

  DYNO_POINTER_TYPEDEFS(Frontend)
  Frontend(const std::string& name, const DynoParams& params,
           ImageDisplayQueue* display_queue);
  virtual ~Frontend() = default;

 protected:
  bool pushImageToDisplayQueue(const std::string& wname, const cv::Mat& image);

 protected:
  virtual void validateInput(
      const FrontendInputPacketBase::ConstPtr& input) const;

  // TODO: log state

  const DynoParams dyno_params_;
  ImageDisplayQueue* display_queue_;
  RGBDFrontendLogger::UniquePtr logger_;
};

class VIFrontend : public Frontend {
 public:
  DYNO_POINTER_TYPEDEFS(VIFrontend)
  VIFrontend(const std::string& name, const DynoParams& params,
             Camera::Ptr camera, ImageDisplayQueue* display_queue);
  virtual ~VIFrontend() = default;

 protected:
  Frame::Ptr featureTrack(const FrontendInputPacketBase::ConstPtr input,
                          std::optional<gtsam::Rot3> R_km1_k = std::nullopt);

  std::optional<gtsam::NavState> tryPropogateImu(
      const FrontendInputPacketBase::ConstPtr input,
      const gtsam::NavState& nav_state_lIMU, ImuFrontend::PimPtr& pim_out);

  bool tryStereoMatch(Frame::Ptr frame, ImageContainer::Ptr image_container,
                      FeaturePtrs& stereo_features_out,
                      FeatureContainer& left_features_in_out);

  bool tryStereoMatchStaticFeatures(Frame::Ptr frame,
                                    ImageContainer::Ptr image_container,
                                    FeaturePtrs& stereo_features_out);

  /**
   * @brief Solve the visual odometry (k-1 to k) using PnP + Refinement with
   * Optical Flow
   *
   * The input T_km1_k should be the best known relative camera motion (usually
   * the relative motion from the previous frame) to act as constant motion
   * model if the tracking fails.
   *
   * @param frame_k Frame::Ptr current frame at k
   * @param frame_km1 Frame::Ptr previous frame at k-1
   * @param nav_state_km1 const gtsam::NavState& previous nav state at k-1
   * @param T_km1_k const gtsam::Pose3& relative camera motion to act as a
   * constant velocity model.
   * @param propogated_nav_state_k std::optional<gtsam::NavState> nav state at k
   * as propogated by the IMU
   * @param R_km1_k std::optional<gtsam::Rot3> relative camera rotation from k-1
   * to k
   * @return true
   * @return false
   */
  bool solveAndRefineEgoMotion(
      Frame::Ptr frame_k, const Frame::Ptr& frame_km1,
      const gtsam::NavState& nav_state_km1, const gtsam::Pose3& T_km1_k,
      std::optional<gtsam::NavState> propogated_nav_state_k = std::nullopt,
      std::optional<gtsam::Rot3> R_km1_k = std::nullopt);

  // TODO: add back object poses?
  void fillDebugImagery(DebugImagery& debug_imagery, const Frame::Ptr& frame_k,
                        const Frame::Ptr& frame_km1) const;

 protected:
  Camera::Ptr camera_;
  EgoMotionSolver ego_motion_solver_;
  ImuFrontend imu_frontend_;
  FeatureTracker::UniquePtr tracker_;
};

// maybe a better name is VIModule or something? Depends on what we mean by
// regular!!!
using RegularBackendSink =
    std::function<void(const VisionImuPacket::ConstPtr&)>;

// TODO: include sinks (callback) to backend with VisionImuOutput
class RegularVIFrontend : public VIFrontend {
 public:
  DYNO_POINTER_TYPEDEFS(RegularVIFrontend)
  RegularVIFrontend(const DynoParams& params, Camera::Ptr camera,
                    ImageDisplayQueue* display_queue = nullptr);

  void addVIOutputSink(const RegularBackendSink& func) {
    regular_backend_output_sink_ = func;
  };

 private:
  SpinReturn boostrapSpin(FrontendInputPacketBase::ConstPtr input) override;
  SpinReturn nominalSpin(FrontendInputPacketBase::ConstPtr input) override;

  void fillOutputPacketWithTracks(
      VisionImuPacket::Ptr vision_imu_packet, const Frame& frame,
      const gtsam::Pose3 X_W_k, const gtsam::Pose3& T_k_1_k,
      const MultiObjectTrajectories& object_trajectories) const;

 private:
  ConsecutiveFrameObjectMotionSolver::UniquePtr object_motion_solver_;

  gtsam::NavState nav_state_km1_;
  //! The relative camera pose (T_k_1_k) from the previous frame
  //! this is used as a constant velocity model when VO tracking fails and the
  //! IMU is not available!
  gtsam::Pose3 T_km1_k_;

  //! Current trajectories. Copied to the DynoState output
  DynoStateTrajectories dyno_state_;

  RegularBackendSink regular_backend_output_sink_;
};

class PoseChangeVIFrontend : public VIFrontend {
 public:
  DYNO_POINTER_TYPEDEFS(PoseChangeVIFrontend)
  PoseChangeVIFrontend(const DynoParams& params, Camera::Ptr camera,
                       ImageDisplayQueue* display_queue = nullptr);

 private:
  SpinReturn boostrapSpin(FrontendInputPacketBase::ConstPtr input) override;
  SpinReturn nominalSpin(FrontendInputPacketBase::ConstPtr input) override;
};

}  // namespace dyno
