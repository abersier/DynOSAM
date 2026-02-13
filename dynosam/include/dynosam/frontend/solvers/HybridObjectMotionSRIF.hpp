#pragma once

#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/StereoPoint2.h>

#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_cv/RGBDCamera.hpp"

namespace dyno {

struct HybridObjectMotionSRIFResult {
  double error{0.0};
  double reweighted_error{0.0};
};

// TODO: rename to use anchor KF or equivalent rather than KF or e!

/**
 * @brief Hybrid Object motion Square-Root Information Filter
 *
 */
class HybridObjectMotionSRIF {
 public:
  HybridObjectMotionSRIF(const gtsam::Pose3& initial_state_H,
                         const gtsam::Pose3& L_e, const FrameId& frame_id_e,
                         const gtsam::Matrix66& initial_P,
                         const gtsam::Matrix66& Q, const gtsam::Matrix33& R,
                         Camera::Ptr camera, double huber_k = 1.23);

  const gtsam::Pose3& getKeyFramePose() const;
  const gtsam::Pose3& lastCameraPose() const;
  FrameId getKeyFrameId() const;
  FrameId getFrameId() const;

  const gtsam::FastMap<TrackletId, gtsam::Point3>& getCurrentLinearizedPoints()
      const;

  gtsam::Pose3 getPose() const;

  void predictAndUpdate(const gtsam::Pose3& H_w_km1_k_predict, Frame::Ptr frame,
                        const TrackletIds& tracklets,
                        const int num_irls_iterations = 1);

  /**
   * @brief Recovers the state perturbation delta_w by solving R * delta_w = d.
   */
  gtsam::Vector6 getStatePerturbation() const;

  // this is H_W_e_k
  const gtsam::Pose3& getCurrentLinearization() const;

  // this is H_W_e_k
  // calculate best estimate!!
  gtsam::Pose3 getKeyFramedMotion() const;

  Motion3ReferenceFrame getKeyFramedMotionReference() const;

  /**
   * @brief Recovers the full state pose W by applying the perturbation
   * to the linearization point.
   *
   * LIES: thie is H_W_km1_k
   */
  gtsam::Pose3 getF2FMotion() const;

  /**
   * @brief Recovers the state covariance P by inverting the information matrix.
   * @note This is a slow operation (O(N^3)) and should only be called
   * for inspection, not inside the filter loop.
   */
  gtsam::Matrix66 getCovariance() const;

  /**
   * @brief Recovers the information matrix Lambda = R^T * R.
   */
  gtsam::Matrix66 getInformationMatrix() const;

  /**
   * @brief Resets information d_info_ and R_info.
   * d_inifo is set to zero and R_info is constructed from the initial
   * covariance P. L_e_ is updated with new value and previous_H_ reset to
   * identity
   *
   * @param L_e
   * @param frame_id_e
   */
  void resetState(const gtsam::Pose3& L_e, FrameId frame_id_e);

 private:
  /**
   * @brief EKF Prediction Step (Trivial motion model for W)
   * @note Prediction is the hard/slow part of an Information Filter.
   * This implementation is a "hack" that converts to covariance,
   * adds noise, and converts back. A "pure" SRIF predict is complex.
   */
  void predict(const gtsam::Pose3& H_W_km1_k);
  /**
   * @brief SRIF Update Step using Iteratively Reweighted Least Squares (IRLS)
   * with QR decomposition to achieve robustness.
   */
  HybridObjectMotionSRIFResult update(Frame::Ptr frame,
                                      const TrackletIds& tracklets,
                                      const int num_irls_iterations = 1);

 private:
  //! Nominal state (linearization point)
  gtsam::Pose3 H_linearization_point_;
  //! Process Noise Covariance (for prediction step)
  const gtsam::Matrix66 Q_;
  //! 3x3 Measurement Noise
  const gtsam::Matrix33 R_noise_;
  //! Cached R_noise inverse
  const gtsam::Matrix33 R_inv_;
  const gtsam::Matrix66 initial_P_;

  gtsam::Pose3 L_e_;
  // Frame Id for the reference KF
  FrameId frame_id_e_;
  //! Last camera pose used within predict
  gtsam::Pose3 X_K_;
  //! Frame id used for last update
  FrameId frame_id_;

  //! R (6x6) - Upper triangular Cholesky factor of Info Matrix
  gtsam::Matrix66 R_info_;
  // ! d (6x1) - Transformed information vector
  gtsam::Vector6 d_info_;

  // --- System Parameters ---
  std::shared_ptr<RGBDCamera> rgbd_camera_;
  gtsam::Cal3_S2Stereo::shared_ptr stereo_calibration_;

  //! Points in L (current linearization)
  gtsam::FastMap<TrackletId, gtsam::Point3> m_linearized_;

  //! should be from e to k-1. Currently set in predict
  gtsam::Pose3 previous_H_;
  double huber_k_{1.23};

  constexpr static int StateDim = gtsam::traits<gtsam::Pose3>::dimension;
  constexpr static int ZDim = gtsam::traits<gtsam::StereoPoint2>::dimension;
};

}  // namespace dyno
