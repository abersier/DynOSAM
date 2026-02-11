#pragma once

#include "dynosam/frontend/FrontendInputPacket.hpp"
#include "dynosam/frontend/imu/ImuFrontend.hpp"
#include "dynosam/frontend/vision/FeatureTracker.hpp"
#include "dynosam/pipeline/PipelineParams.hpp"
#include "dynosam_common/DynoState.hpp"
#include "dynosam_common/ModuleBase.hpp"
#include "dynosam_common/RealtimeOutput.hpp"
#include "dynosam_common/Trajectories.hpp"

namespace dyno {

// TODO: Original is in FrontendModule which can be deleted once refactoring
struct InvalidImageContainerException1 : public DynosamException {
  InvalidImageContainerException1(const ImageContainer& container,
                                  const std::string& what)
      : DynosamException("Image container with config: " +
                         container.toString() + "\n was invalid - " + what) {}
};

class RGBDFrontendLogger : public EstimationModuleLogger {
 public:
  DYNO_POINTER_TYPEDEFS(RGBDFrontendLogger)
  RGBDFrontendLogger();
  virtual ~RGBDFrontendLogger();

  void logTrackingLengthHistogram(const Frame::Ptr frame);

 private:
  std::string tracking_length_hist_file_name_;
  json tracklet_length_json_;
};

class Frontend : public ModuleBase<FrontendInputPacketBase, RealtimeOutput> {
 public:
  using Base = ModuleBase<FrontendInputPacketBase, RealtimeOutput>;

  DYNO_POINTER_TYPEDEFS(Frontend)
  Frontend(const std::string& name, const DynoParams& params,
           ImageDisplayQueue* display_queue,
           const SharedGroundTruth& shared_ground_truth);
  virtual ~Frontend() = default;

 protected:
  bool pushImageToDisplayQueue(const std::string& wname, const cv::Mat& image);

  void logRealTimeOutput(const RealtimeOutput::Ptr& output);

 protected:
  virtual void validateInput(
      const FrontendInputPacketBase::ConstPtr& input) const;

  const DynoParams dyno_params_;
  ImageDisplayQueue* display_queue_;
  const SharedGroundTruth shared_ground_truth_;

  RGBDFrontendLogger::UniquePtr logger_;
};

class VIFrontend : public Frontend {
 public:
  DYNO_POINTER_TYPEDEFS(VIFrontend)
  VIFrontend(const std::string& name, const DynoParams& params,
             Camera::Ptr camera, ImageDisplayQueue* display_queue,
             const SharedGroundTruth& shared_ground_truth);
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

  // helper functions
  void fillMeasurementsFromFeatureIterator(
      CameraMeasurementStatusVector* measurements, FeatureFilterIterator it,
      FrameId frame_id, Timestamp timestamp, const gtsam::Vector2& pixel_sigmas,
      double depth_sigma, StatusLandmarkVector* landmarks = nullptr) const;

 protected:
  Camera::Ptr camera_;

  EgoMotionSolver ego_motion_solver_;
  ImuFrontend imu_frontend_;
  FeatureTracker::UniquePtr tracker_;

  //! Cached rgbd-camera
  std::shared_ptr<RGBDCamera> rgbd_camera_;

  //! Cached sigmas for static feature measurements
  double static_point_sigma_{0};
  gtsam::Vector2 static_pixel_sigmas_;
  //! Cached sigmas for dynamic feature measurements
  double dynamic_point_sigma_{0};
  gtsam::Vector2 dynamic_pixel_sigmas_;
};

}  // namespace dyno
