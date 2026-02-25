#pragma once

#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam_common/Trajectories.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_cv/RGBDCamera.hpp"

namespace dyno {

// RENAME for keyframe?
class HybridObjectMotionSolverImpl {
 public:
  DYNO_POINTER_TYPEDEFS(HybridObjectMotionSolverImpl)

  HybridObjectMotionSolverImpl(ObjectId object_id, Camera::Ptr camera)
      : object_id_(object_id),
        logger_prefix_("hybrid_motion_smoother_j" + std::to_string(object_id)) {
    rgbd_camera_ = CHECK_NOTNULL(camera)->safeGetRGBDCamera();
    CHECK(rgbd_camera_);
    stereo_calibration_ = rgbd_camera_->getFakeStereoCalib();
  }

  ~HybridObjectMotionSolverImpl() = default;

  virtual FrameId keyFrameId() const = 0;
  virtual FrameId frameId() const = 0;
  virtual Timestamp timestamp() const = 0;

  virtual gtsam::Pose3 keyFrameMotion() const = 0;
  virtual gtsam::Pose3 keyFramePose() const = 0;
  virtual gtsam::Pose3 frameToFrameMotion() const = 0;

  virtual Motion3ReferenceFrame keyFrameMotionReference() {
    return Motion3ReferenceFrame(
        keyFrameMotion(), Motion3ReferenceFrame::Style::KF,
        ReferenceFrame::GLOBAL, keyFrameId(), frameId());
  }

  virtual gtsam::Pose3 pose() const {
    return keyFrameMotion() * keyFramePose();
  }

  virtual void update(const gtsam::Pose3& H_w_km1_k_predict, Frame::Ptr frame,
                      const TrackletIds& tracklets) = 0;

  virtual void createNewKeyedMotion(const gtsam::Pose3& L_KF, Frame::Ptr frame,
                                    const TrackletIds& tracklets) = 0;

  virtual gtsam::FastMap<TrackletId, gtsam::Point3> getObjectPoints() const = 0;
  virtual PoseWithMotionTrajectory trajectory() const = 0;
  /**
   * @brief Construct local trajectory representing the object path since
   * (inclusive) the last keyframe (ie KF_k -> k)
   *
   * @return PoseWithMotionTrajectory
   */
  virtual PoseWithMotionTrajectory localTrajectory() const = 0;

 protected:
  const ObjectId object_id_;
  const std::string logger_prefix_;

  std::shared_ptr<RGBDCamera> rgbd_camera_;
  gtsam::Cal3_S2Stereo::shared_ptr stereo_calibration_;
};

}  // namespace dyno
