#include "dynosam/frontend/solvers/HybridObjectMotionSRIF.hpp"

#include "dynosam/factors/HybridFormulationFactors.hpp"
#include "dynosam_common/logger/Logger.hpp"
#include "dynosam_common/utils/TimingStats.hpp"

namespace dyno {

/**
 * @brief Compute the 2-norm condition number of the SRIF matrix R.
 *
 * R must be the upper-triangular matrix obtained after QR decomposition
 * of the *whitened* Jacobian stack:
 *
 *     H = Q R
 *
 * where H is already whitened by measurement covariance.
 *
 * The condition number is:
 *
 *     cond(R) = sigma_max / sigma_min
 *
 * where sigma are the singular values of R.
 *
 * Interpretation guidelines (double precision):
 *
 *   cond(R) < 1e3      → Very healthy system
 *   1e4 – 1e6          → Mild degeneracy
 *   1e7 – 1e9          → Serious warning
 *   > 1e10             → Numerically dangerous
 *
 * Why it matters:
 *   - Large cond(R) means the information ellipsoid is highly stretched.
 *   - Small singular values indicate weakly constrained directions.
 *   - cond(R) ≈ ∞ implies rank deficiency (unobservable directions).
 *
 * In SLAM / SRIF systems, large cond(R) often indicates:
 *   - Gauge freedom not fixed
 *   - Poor motion excitation (e.g., pure rotation)
 *   - Underconstrained object states
 *   - Marginalization corruption
 *   - Incorrect Jacobians
 *
 * Important:
 *   - R must come from whitened Jacobians.
 *   - cond(R) == cond(H)
 *   - Using SRIF avoids squaring the condition number
 *     (since cond(H^T H) = cond(H)^2).
 *
 * @param R Upper triangular SRIF matrix
 * @return Condition number (2-norm)
 */
double conditionNumberR(const Eigen::MatrixXd& R) {
  if (R.rows() == 0 || R.cols() == 0)
    throw std::invalid_argument("R must be non-empty.");

  // Compute singular values of R
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(
      R, Eigen::ComputeThinU | Eigen::ComputeThinV);

  const auto& singular_values = svd.singularValues();

  const double sigma_max = singular_values(0);
  const double sigma_min = singular_values(singular_values.size() - 1);

  // Handle rank-deficient case
  if (sigma_min <= std::numeric_limits<double>::epsilon())
    return std::numeric_limits<double>::infinity();

  return sigma_max / sigma_min;
}

HybridObjectMotionSRIF::HybridObjectMotionSRIF(
    const ObjectId object_id, const gtsam::Pose3& initial_state_H,
    const gtsam::Pose3& L_e, const FrameId& frame_id_e,
    const Timestamp& timestamp_e, const gtsam::Matrix66& initial_P,
    const gtsam::Matrix66& Q, const gtsam::Matrix33& R, Camera::Ptr camera,
    double huber_k)
    : object_id_(object_id),
      H_linearization_point_(initial_state_H),
      Q_(Q),
      R_noise_(R),
      R_inv_(R.inverse()),
      initial_P_(initial_P),
      huber_k_(huber_k),
      logger_(CsvHeader("timestamp", "frame_id", "residual", "initial_error",
                        "weighted_error", "new_features", "existing_tracks",
                        "cond_R")) {
  // set camera
  rgbd_camera_ = CHECK_NOTNULL(camera)->safeGetRGBDCamera();
  CHECK(rgbd_camera_);
  stereo_calibration_ = rgbd_camera_->getFakeStereoCalib();

  resetState(L_e, frame_id_e, timestamp_e);
}

HybridObjectMotionSRIF::~HybridObjectMotionSRIF() {
  const std::string file_out =
      "hybrid_srif_j_" + std::to_string(object_id_) + ".csv";
  OfstreamWrapper::WriteOutCsvWriter(logger_, file_out);
}

const gtsam::Pose3& HybridObjectMotionSRIF::getKeyFramePose() const {
  return L_e_;
}
const gtsam::Pose3& HybridObjectMotionSRIF::lastCameraPose() const {
  return X_K_;
}
FrameId HybridObjectMotionSRIF::getKeyFrameId() const { return frame_id_e_; }
FrameId HybridObjectMotionSRIF::getFrameId() const { return frame_id_; }

Timestamp HybridObjectMotionSRIF::getKeyFrameTimestamp() const {
  return timestamp_e_;
}
Timestamp HybridObjectMotionSRIF::getTimestamp() const { return timestamp_; }

const gtsam::FastMap<TrackletId, gtsam::Point3>&
HybridObjectMotionSRIF::getCurrentLinearizedPoints() const {
  return m_linearized_;
}

gtsam::Pose3 HybridObjectMotionSRIF::getPose() const {
  return getKeyFramedMotion() * getKeyFramePose();
}

void HybridObjectMotionSRIF::predictAndUpdate(
    const gtsam::Pose3& H_w_km1_k_predict, Frame::Ptr frame,
    const TrackletIds& tracklets, const int num_irls_iterations) {
  utils::ChronoTimingStats timer("hybrid_object_motion_srif.predictAndUpdate");
  predict(H_w_km1_k_predict);

  const FrameId frame_id_before_update = frame_id_;
  update(frame, tracklets, num_irls_iterations);
  const FrameId frame_id_after_update = frame_id_;
}

/**
 * @brief Recovers the state perturbation delta_w by solving R * delta_w = d.
 */
gtsam::Vector6 HybridObjectMotionSRIF::getStatePerturbation() const {
  // R is upper triangular, so this is a fast back-substitution
  return R_info_.triangularView<Eigen::Upper>().solve(d_info_);
}

const gtsam::Pose3& HybridObjectMotionSRIF::getCurrentLinearization() const {
  return H_linearization_point_;
}

gtsam::Pose3 HybridObjectMotionSRIF::getKeyFramedMotion() const {
  return H_linearization_point_.retract(getStatePerturbation());
}

Motion3ReferenceFrame HybridObjectMotionSRIF::getKeyFramedMotionReference()
    const {
  return Motion3ReferenceFrame(
      getKeyFramedMotion(), Motion3ReferenceFrame::Style::KF,
      ReferenceFrame::GLOBAL, getKeyFrameId(), getFrameId());
}

gtsam::Pose3 HybridObjectMotionSRIF::getF2FMotion() const {
  gtsam::Pose3 H_W_e_k = H_linearization_point_.retract(getStatePerturbation());
  gtsam::Pose3 H_W_e_km1 = previous_H_;
  return H_W_e_k * H_W_e_km1.inverse();
}

gtsam::Matrix66 HybridObjectMotionSRIF::getCovariance() const {
  gtsam::Matrix66 Lambda = R_info_.transpose() * R_info_;
  return Lambda.inverse();
}

gtsam::Matrix66 HybridObjectMotionSRIF::getInformationMatrix() const {
  return R_info_.transpose() * R_info_;
}

void HybridObjectMotionSRIF::predict(const gtsam::Pose3& H_W_km1_k) {
  // 1. Get current mean state and covariance
  gtsam::Vector6 delta_w = getStatePerturbation();
  gtsam::Pose3 H_current_mean = H_linearization_point_.retract(delta_w);
  gtsam::Matrix66 P_current = getCovariance();  // Slow step

  // call before updating the previous_H_
  // previous motion -> assume constant!
  // gtsam::Pose3 H_W_km1_k = getF2FMotion();

  // gtsam::Pose3 L_km1 = previous_H_ * L_e_;
  // gtsam::Pose3 L_k = H_current_mean * L_e_;

  // //calculate motion relative to object
  // gtsam::Pose3 local_motion = L_km1.inverse() * L_k;
  // // forward predict pose using constant velocity model
  // gtsam::Pose3 predicted_pose = L_k * local_motion;
  // // convert pose back to motion
  // gtsam::Pose3 predicted_motion = predicted_pose * L_e_.inverse();

  previous_H_ = H_current_mean;

  // 2. Perform EKF prediction (add process noise)
  // P_k = P_{k-1} + Q
  gtsam::Matrix66 P_predicted = P_current + Q_;

  // 3. Re-linearize: Set new linearization point to the current mean
  H_linearization_point_ = H_W_km1_k * H_current_mean;

  // 4. Recalculate R and d based on new P and new linearization point
  // The new perturbation is 0 relative to the new linearization point.
  d_info_ = gtsam::Vector6::Zero();
  gtsam::Matrix66 Lambda_predicted = P_predicted.inverse();
  R_info_ =
      Lambda_predicted.llt().matrixU();  // R_info_ = L^T where L*L^T = Lambda
}

HybridObjectMotionSRIFResult HybridObjectMotionSRIF::update(
    Frame::Ptr frame, const TrackletIds& tracklets,
    const int num_irls_iterations) {
  const size_t num_measurements = tracklets.size();
  const size_t total_rows = num_measurements * ZDim;

  const gtsam::Pose3& X_W_k = frame->getPose();
  const gtsam::Pose3 X_W_k_inv = X_W_k.inverse();
  X_K_ = X_W_k;
  frame_id_ = frame->getFrameId();
  timestamp_ = frame->getTimestamp();

  return HybridObjectMotionSRIFResult{};

  // 1. Calculate Jacobians (H) and Linearized Residuals (y_lin)
  // These are calculated ONCE at the linearization point and are fixed
  // for all IRLS iterations.
  gtsam::Matrix H_stacked = gtsam::Matrix ::Zero(total_rows, StateDim);
  gtsam::Vector y_linearized = gtsam::Vector::Zero(total_rows);

  const gtsam::Pose3 A = X_W_k_inv * H_linearization_point_;
  // const gtsam::Pose3 A = X_W_k_inv * H_linearization_point_;
  // G will project the point from the object frame into the camera frame
  const gtsam::Pose3 G_w = A * L_e_;
  const gtsam::Matrix6 J_correction = -A.AdjointMap();

  gtsam::StereoCamera gtsam_stereo_camera(G_w.inverse(), stereo_calibration_);
  // whitened error
  gtsam::Vector initial_error = gtsam::Vector::Zero(num_measurements);
  gtsam::Vector re_weighted_error = gtsam::Vector::Zero(num_measurements);

  size_t num_new_features = 0u, num_tracked_features = 0u,
         num_good_measurements = 0u;
  double average_residual = 0;

  for (size_t i = 0; i < num_measurements; ++i) {
    const TrackletId& tracklet_id = tracklets.at(i);
    const Feature::Ptr feature = frame->at(tracklet_id);
    CHECK(feature);

    if (!m_linearized_.exists(tracklet_id)) {
      // this is initalising from a predicted motion (bad!) Should use previous
      // linearization (ie k-1)
      const gtsam::Point3 m_X_k = frame->backProjectToCamera(tracklet_id);
      Landmark m_init = HybridObjectMotion::projectToObject3(
          X_W_k, getKeyFramedMotion(), L_e_, m_X_k);
      m_linearized_.insert2(tracklet_id, m_init);
      // VLOG(10) << "Initalising new point i=" << tracklet_id << " Le " <<
      // L_e_;
      num_new_features++;
    } else {
      num_tracked_features++;
    }

    gtsam::Point3 m_L = m_linearized_.at(tracklet_id);

    const auto [stereo_keypoint_status, stereo_measurement] =
        rgbd_camera_->getStereo(feature);
    if (!stereo_keypoint_status) {
      continue;
    }

    const auto& z_obs = stereo_measurement;

    // const Point2& z_obs = feature->keypoint();
    gtsam::Matrix36 J_pi;  // 2x6
    gtsam::StereoPoint2 z_pred;
    try {
      // Project using the *linearization point*
      // z_pred = camera.project(m_L, J_pi);
      z_pred = gtsam_stereo_camera.project2(m_L, J_pi);
    } catch (const gtsam::StereoCheiralityException& e) {
      LOG(WARNING) << "Warning: Point " << i << " behind camera. Skipping.";
      continue;  // Skip this measurement
    }

    // Calculate linearized residual y_lin = z - h(x_nom)
    gtsam::Vector3 y_i_lin = (z_obs - z_pred).vector();
    Eigen::Matrix<double, ZDim, StateDim> H_ekf_i = J_pi * J_correction;

    size_t row_idx = i * ZDim;

    H_stacked.block<ZDim, StateDim>(row_idx, 0) = H_ekf_i;
    y_linearized.segment<ZDim>(row_idx) = y_i_lin;

    double error_sq = y_i_lin.transpose() * R_inv_ * y_i_lin;
    double error = std::sqrt(error_sq);
    initial_error(i) = error;

    average_residual += y_i_lin.norm();
    num_good_measurements++;
  }

  average_residual /= (double)num_good_measurements;

  VLOG(30) << "Feature stats. New features: " << num_new_features << "/"
           << num_measurements << " Tracked features " << num_tracked_features
           << "/" << num_measurements;

  // 2. Store the prior information
  gtsam::Matrix6 R_info_prior = R_info_;
  gtsam::Vector6 d_info_prior = d_info_;

  // 3. --- Start Iteratively Reweighted Least Squares (IRLS) Loop ---
  // cout << "\n--- SRIF Robust Update (IRLS) Started ---" << endl;
  for (int iter = 0; iter < num_irls_iterations; ++iter) {
    // 3a. Get current state estimate (from previous iteration)
    const gtsam::Vector6 delta_w =
        R_info_.triangularView<Eigen::Upper>().solve(d_info_);
    const gtsam::Pose3 H_current_mean = H_linearization_point_.retract(delta_w);
    // Need to validate this:
    //  We intentionally do not recalculate the Jacobian block (H_stacked).
    //  to solve for the single best perturbation (delta_w) relative to the
    //  single linearization point (W_linearization_point_) that we had at the
    //  start of the update.
    const gtsam::Pose3 A = X_W_k_inv * H_current_mean;
    const gtsam::Pose3 G_w = A * L_e_;
    gtsam::StereoCamera gtsam_stereo_camera_current(G_w.inverse(),
                                                    stereo_calibration_);

    gtsam::Vector weights = gtsam::Vector::Ones(num_measurements);

    for (size_t i = 0; i < num_measurements; ++i) {
      const TrackletId& tracklet_id = tracklets.at(i);
      const Feature::Ptr feature = frame->at(tracklet_id);
      CHECK(feature);

      const auto [stereo_keypoint_status, stereo_measurement] =
          rgbd_camera_->getStereo(feature);
      if (!stereo_keypoint_status) {
        weights(i) = 0.0;  // Skip point
        continue;
      }

      const auto& z_obs = stereo_measurement;

      const gtsam::Point3& m_L = m_linearized_.at(tracklet_id);

      gtsam::StereoPoint2 z_pred_current;
      try {
        z_pred_current = gtsam_stereo_camera_current.project(m_L);
      } catch (const gtsam::StereoCheiralityException& e) {
        LOG(WARNING) << "Warning: Point " << i << " behind camera. Skipping.";
        weights(i) = 0.0;  // Skip point
        continue;
      }

      // Calculate non-linear residual
      gtsam::Vector3 y_nonlinear = (z_obs - z_pred_current).vector();

      // //CHECK we cannot do this with gtsam's noise models (we can...)
      // // Calculate Mahalanobis-like distance (whitened error)
      double error_sq = y_nonlinear.transpose() * R_inv_ * y_nonlinear;
      double error = std::sqrt(error_sq);

      re_weighted_error(i) = error;

      // Calculate Huber weight w(e) = min(1, delta / |e|)
      weights(i) = (error <= huber_k_) ? 1.0 : huber_k_ / error;
      if (/*weights(i) < 0.99 &&*/ iter == num_irls_iterations - 1) {
        VLOG(50) << "  [Meas " << i << "] Final Weight: " << weights(i)
                 << " (Error: " << error << ")";
      }
    }

    // 3c. Construct the giant matrix A_qr for this iteration
    gtsam::Matrix A_qr =
        gtsam::Matrix::Zero(total_rows + StateDim, StateDim + 1);

    for (size_t i = 0; i < num_measurements; ++i) {
      double w_i = weights(i);
      if (w_i < 1e-6) continue;  // Skip 0-weighted points

      // R_robust = R / w_i  (Note: w is |e|^-1, so R_robust = R * |e|/delta)
      // R_robust_inv = R_inv * w_i
      // We need W_i such that W_i^T * W_i = R_robust_inv
      // W_i = sqrt(w_i) * R_inv_sqrt

      gtsam::Matrix33 R_robust_i = R_noise_ / w_i;
      gtsam::Matrix33 L_robust_i = R_robust_i.llt().matrixL();
      gtsam::Matrix33 R_robust_inv_sqrt = L_robust_i.inverse();

      size_t row_idx = i * ZDim;

      A_qr.block<ZDim, StateDim>(row_idx, 0) =
          R_robust_inv_sqrt * H_stacked.block<3, StateDim>(row_idx, 0);
      A_qr.block<ZDim, 1>(row_idx, StateDim) =
          R_robust_inv_sqrt * y_linearized.segment<3>(row_idx);
    }

    // 3d. Fill the prior part of the QR matrix
    // This is *always* the original prior
    A_qr.block<StateDim, StateDim>(total_rows, 0) = R_info_prior;
    A_qr.block<StateDim, 1>(total_rows, StateDim) = d_info_prior;

    // 3e. Perform QR decomposition
    Eigen::HouseholderQR<gtsam::Matrix> qr(A_qr);
    gtsam::Matrix R_full = qr.matrixQR().triangularView<Eigen::Upper>();

    // 3f. Extract the new R_info and d_info for the *next* iteration
    R_info_ = R_full.block<StateDim, StateDim>(0, 0);
    d_info_ = R_full.block<StateDim, 1>(0, StateDim);
  }

  // (Optional) Enforce R is upper-triangular
  R_info_ = R_info_.triangularView<Eigen::Upper>();

  HybridObjectMotionSRIFResult result;
  result.error = initial_error.norm();
  result.reweighted_error = re_weighted_error.norm();

  logger_ << timestamp_ << frame_id_ << average_residual << result.error
          << result.reweighted_error << num_new_features << num_tracked_features
          << conditionNumberR(R_info_);

  return result;
}

void HybridObjectMotionSRIF::resetState(const gtsam::Pose3& L_e,
                                        FrameId frame_id_e,
                                        Timestamp timestamp_e) {
  // 2. Initialize SRIF State (R, d) from EKF State (W, P)
  // W_linearization_point_ is set to initial_state_W
  // The initial perturbation is 0, so the initial d_info_ is 0.
  d_info_ = gtsam::Vector6::Zero();

  // Calculate initial Information Matrix Lambda = P^-1
  gtsam::Matrix66 Lambda_initial = initial_P_.inverse();

  // Calculate R_info_ from Cholesky decomposition: Lambda = R^T * R
  // We use LLT (L*L^T) and take the upper-triangular factor of L^T.
  // Or, more robustly, U^T*U from a U^T*U decomposition.
  // Eigen's LLT computes L (lower) such that A = L * L^T.
  // We need U (upper) such that A = U^T * U.
  // A.llt().matrixU() gives U where A = L*L^T (U is L^T). No.
  // A.llt().matrixL() gives L where A = L*L^T.
  // Let's use Eigen's LDLT and reconstruct.
  // A more direct way:
  R_info_ = Lambda_initial.llt().matrixU();  // L^T

  // will this give us discontinuinities betweek keyframes?
  //  reset other variables
  previous_H_ = gtsam::Pose3::Identity();
  L_e_ = L_e;
  frame_id_e_ = frame_id_e;
  frame_id_ = frame_id_e;
  timestamp_e_ = timestamp_e;
  timestamp_ = timestamp_e;
  X_K_ = gtsam::Pose3::Identity();

  m_linearized_.clear();

  // should always be identity since the deviation from L_e = I when frame = e
  // the initial H should not be parsed into the constructor by this logic as
  // well!
  H_linearization_point_ = gtsam::Pose3::Identity();
}

void HybridObjectMotionSRIF::relinearize() {
  // not really relineaize? Just setting delta and R to zero and updating H_lin
  // but h_lin gets updated in predict anyway?
  H_linearization_point_ = getBestEstimate();
  previous_H_ = gtsam::Pose3::Identity();
  d_info_ = gtsam::Vector6::Zero();

  gtsam::Matrix66 Lambda_initial = initial_P_.inverse();
  R_info_ = Lambda_initial.llt().matrixU();  // L^T
}

}  // namespace dyno
