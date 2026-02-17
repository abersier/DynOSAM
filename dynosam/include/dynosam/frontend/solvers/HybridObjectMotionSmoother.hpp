#pragma once

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/ISAM2UpdateParams.h>
#include <gtsam_unstable/nonlinear/FixedLagSmoother.h>

#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_cv/RGBDCamera.hpp"
#include "dynosam_opt/Symbols.hpp"

namespace dyno {

class HybridObjectMotionSmoother : public gtsam::FixedLagSmoother {
 public:
  DYNO_POINTER_TYPEDEFS(HybridObjectMotionSmoother)

  HybridObjectMotionSmoother(ObjectId object_id, const gtsam::Pose3& L_KF,
                             FrameId frame_id_k, Timestamp timestamp_k,
                             Camera::Ptr camera, double smootherLag = 0.0);

  void update(const gtsam::Pose3& H_w_km1_k_predict, Frame::Ptr frame,
              const TrackletIds& tracklets);

  PoseWithMotionTrajectory trajectory() const;

  //   /** Print the factor for debugging and testing (implementing Testable) */
  //   void print(const std::string& s = "IncrementalFixedLagSmoother:\n",
  //       const KeyFormatter& keyFormatter = DefaultKeyFormatter) const
  //       override;

  //   /** Check if two IncrementalFixedLagSmoother Objects are equal */
  //   bool equals(const FixedLagSmoother& rhs, double tol = 1e-9) const
  //   override;

  FrameId getKeyFrameId() const { return frames_since_lKF_.front(); }
  FrameId getFrameId() const { return frames_since_lKF_.back(); }
  Timestamp getTimestamp() const { return timestamps_since_lKF_.back(); }

  Motion3ReferenceFrame getKeyFramedMotionReference() const;
  gtsam::Pose3 getKeyFramedMotion() const;

  inline gtsam::Pose3 getBestEstimate() const { return getKeyFramedMotion(); }
  gtsam::Pose3 getF2FMotion() const;

  gtsam::Pose3 getKeyFramePose() const { return L_KF_; }
  gtsam::Pose3 getPose() const;

  gtsam::FastMap<TrackletId, gtsam::Point3> getCurrentLinearizedPoints() const;

  // This is basically reset
  //  What information from the previous state do we propogate over (ie.
  //  points?) if any
  void createNewKeyedMotion(const gtsam::Pose3& L_KF, FrameId frame_id_k,
                            Timestamp timestamp_k);

  void update(const gtsam::NonlinearFactorGraph& newFactors,
              const gtsam::Values& newTheta,
              const KeyTimestampMap& timestamps = KeyTimestampMap(),
              const gtsam::ISAM2UpdateParams& update_params =
                  gtsam::ISAM2UpdateParams());

  /** Compute an estimate from the incomplete linear delta computed during the
   * last update. This delta is incomplete because it was not updated below
   * wildfire_threshold.  If only a single variable is needed, it is faster to
   * call calculateEstimate(const KEY&).
   */
  gtsam::Values calculateEstimate() const override {
    return isam_.calculateEstimate();
  }

  /** Compute an estimate for a single variable using its incomplete linear
   * delta computed during the last update.  This is faster than calling the
   * no-argument version of calculateEstimate, which operates on all variables.
   * @param key
   * @return
   */
  template <class VALUE>
  VALUE calculateEstimate(gtsam::Key key) const {
    return isam_.calculateEstimate<VALUE>(key);
  }

  /** return the current set of iSAM2 parameters */
  const gtsam::ISAM2Params& params() const { return isam_.params(); }

  /** Access the current set of factors */
  const gtsam::NonlinearFactorGraph& getFactors() const {
    return isam_.getFactorsUnsafe();
  }

  /** Access the current linearization point */
  const gtsam::Values& getLinearizationPoint() const {
    return isam_.getLinearizationPoint();
  }

  /** Access the current set of deltas to the linearization point */
  const gtsam::VectorValues& getDelta() const { return isam_.getDelta(); }

  /// Calculate marginal covariance on given variable
  gtsam::Matrix marginalCovariance(gtsam::Key key) const {
    return isam_.marginalCovariance(key);
  }

  /// Get results of latest isam2 update
  const gtsam::ISAM2Result& getISAM2Result() const { return isamResult_; }

  /// Get the iSAM2 object which is used for the inference internally
  const gtsam::ISAM2& getISAM2() const { return isam_; }

 protected:
  const ObjectId object_id_;
  gtsam::Pose3 L_KF_;

  // Trajectory since last KF?
  // Updated when new KF made since past variables will not be updated
  // same when marginalized
  // vector of frames and timestamps related to variables since last KF?
  // first frame is KF and last is k
  FrameIds frames_since_lKF_;
  Timestamps timestamps_since_lKF_;

  std::shared_ptr<RGBDCamera> rgbd_camera_;
  gtsam::Cal3_S2Stereo::shared_ptr stereo_calibration_;

  /** Create default parameters */
  static gtsam::ISAM2Params DefaultISAM2Params() {
    gtsam::ISAM2Params params;
    params.findUnusedFactorSlots = true;
    params.keyFormatter = DynosamKeyFormatter;
    params.relinearizeThreshold = 0.01;
    // this value is very important for accuracy
    params.relinearizeSkip = 1;
    return params;
  }

  /** An iSAM2 object used to perform inference. The smoother lag is controlled
   * by what factors are removed each iteration */
  gtsam::ISAM2 isam_;

  /** Store results of latest isam2 update */
  gtsam::ISAM2Result isamResult_;

  /** Erase any keys associated with timestamps before the provided time */
  void eraseKeysBefore(double timestamp);

  /** Fill in an iSAM2 ConstrainedKeys structure such that the provided keys are
   * eliminated before all others */
  void createOrderingConstraints(
      const gtsam::KeyVector& marginalizableKeys,
      boost::optional<gtsam::FastMap<gtsam::Key, int> >& constrainedKeys) const;

 private:
  inline gtsam::FixedLagSmootherResult update(
      const gtsam::NonlinearFactorGraph&,
      const gtsam::Values&,  //
      const KeyTimestampMap&, const gtsam::FactorIndices&) override {}

 private:
  //   /** Private methods for printing debug information */
  //   static void PrintKeySet(const KeySet& keys, const std::string& label =
  //       "Keys:");
  //   static void PrintSymbolicFactor(const GaussianFactor::shared_ptr&
  //   factor); static void PrintSymbolicGraph(const GaussianFactorGraph& graph,
  //       const std::string& label = "Factor Graph:");
  //   static void PrintSymbolicTree(const gtsam::ISAM2& isam,
  //       const std::string& label = "Bayes Tree:");
  //   static void PrintSymbolicTreeHelper(
  //       const gtsam::ISAM2Clique::shared_ptr& clique, const std::string
  //       indent =
  //           "");
};

}  // namespace dyno
