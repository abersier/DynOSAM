#pragma once

#include <gtsam/nonlinear/ISAM2.h>
#include <gtsam/nonlinear/ISAM2UpdateParams.h>
#include <gtsam_unstable/nonlinear/FixedLagSmoother.h>
#include <gtsam_unstable/slam/SmartStereoProjectionPoseFactor.h>

#include <nlohmann/json.hpp>

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/frontend/solvers/HybridObjectMotionSolver-Impl.hpp"
#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam_common/Trajectories.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_cv/RGBDCamera.hpp"
#include "dynosam_opt/ISAM2.hpp"  //FOR TESTING!!!
#include "dynosam_opt/ISAM2Result.hpp"
#include "dynosam_opt/ISAM2UpdateParams.hpp"
#include "dynosam_opt/IncrementalOptimization.hpp"
#include "dynosam_opt/Symbols.hpp"

namespace dyno {

class HybridObjectMotionSmoother : public HybridObjectMotionSolverImpl,
                                   public gtsam::FixedLagSmoother {
 public:
  DYNO_POINTER_TYPEDEFS(HybridObjectMotionSmoother)

  struct Result {
    gtsam::KeyVector marginalized_keys;
    gtsam::KeyList additional_keys_reeliminate;
    double update_time_ms{0};
    double marginalize_time_ms{0};
    // gtsam::ISAM2Result isam_result;
    dyno::ISAM2Result isam_result;
  };

  static HybridObjectMotionSmoother::Ptr CreateWithInitialMotion(
      const ObjectId object_id, double smoother_lag,
      const gtsam::Pose3& L_KF_km1, Frame::Ptr frame_km1,
      const TrackletIds& tracklets);

  ~HybridObjectMotionSmoother();

  // should only be called once a valid createNewKeyedMotion has been called!
  void update(const gtsam::Pose3& H_w_km1_k_predict, Frame::Ptr frame,
              const TrackletIds& tracklets) override;

  // This is basically reset
  //  What information from the previous state do we propogate over (ie.
  //  points?) if any
  void createNewKeyedMotion(const gtsam::Pose3& L_KF, Frame::Ptr frame,
                            const TrackletIds& tracklets) override;

  // trajectory should with F2F motion!!
  PoseWithMotionTrajectory trajectory() const override;

  /**
   * @brief Construct local trajectory representing the object path since
   * (inclusive) the last keyframe (ie KF_k -> k)
   *
   * @return PoseWithMotionTrajectory
   */
  PoseWithMotionTrajectory localTrajectory() const override;

  FrameId keyFrameId() const override { return frames_since_lKF_.front(); }
  FrameId frameId() const override { return frames_since_lKF_.back(); }
  Timestamp timestamp() const override { return timestamps_since_lKF_.back(); }

  // Motion3ReferenceFrame getKeyFramedMotionReference() const override;
  gtsam::Pose3 keyFrameMotion() const;

  gtsam::Pose3 frameToFrameMotion() const override;
  gtsam::Pose3 keyFramePose() const override;

  gtsam::FastMap<TrackletId, gtsam::Point3> getObjectPoints() const override;

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
  // const gtsam::ISAM2Params& params() const { return isam_.params(); }
  const dyno::ISAM2Params& params() const { return isam_.params(); }

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
  // const gtsam::ISAM2Result& getISAM2Result() const { return isamResult_; }
  const dyno::ISAM2Result& getISAM2Result() const { return isamResult_; }

  /// Get the iSAM2 object which is used for the inference internally
  // const gtsam::ISAM2& getISAM2() const { return isam_; }
  const dyno::ISAM2& getISAM2() const { return isam_; }

  struct DebugResult {
    HybridObjectMotionSmoother::Result result;
    ISAM2Stats smoother_stats;
    ObjectId object_id{0};
    FrameId frame_id{0};
    Timestamp timestamp{0};

    // tracking data
    size_t average_feature_age{0};
    size_t num_tracks{0};

    FrameId frame_id_KF{0};
    int num_landmarks_in_smoother{0};
    int num_motions_in_smoother{0};
  };

 private:
  std::map<gtsam::Key, gtsam::Point3> getObjectPointsFromState(
      const gtsam::Values& values) const;

  inline std::map<gtsam::Key, gtsam::Point3> getObjectPointsFromSmootherState()
      const {
    return getObjectPointsFromState(smoother_state_);
  }

  std::map<gtsam::Key, gtsam::Point3> getObjectPointsFromStateSinceLastKF()
      const {
    return getObjectPointsFromState(state_since_lKF_);
  }

  std::map<gtsam::Key, gtsam::Pose3> getObjectMotionsFromState(
      const gtsam::Values& values) const;

  std::map<gtsam::Key, gtsam::Pose3> getObjectMotionsFromSmootherState() const {
    return getObjectMotionsFromState(smoother_state_);
  }

  std::map<gtsam::Key, gtsam::Pose3> getObjectMotionsFromStateSinceLastKF()
      const {
    return getObjectMotionsFromState(state_since_lKF_);
  }

  Result updateFromInitialMotion(const gtsam::Pose3& H_W_KF_k_initial,
                                 Frame::Ptr frame,
                                 const TrackletIds& tracklets);

  Result updateSmoother(
      const gtsam::NonlinearFactorGraph& newFactors,
      const gtsam::Values& newTheta,
      const KeyTimestampMap& timestamps = KeyTimestampMap(),
      const dyno::ISAM2UpdateParams& update_params = dyno::ISAM2UpdateParams());

  PoseWithMotionTrajectory localTrajectoryImpl(
      bool include_keyframe = false) const;

 protected:
  HybridObjectMotionSmoother(ObjectId object_id, Camera::Ptr camera,
                             double smootherLag = 0.0);

  // Trajectory since last KF?
  // Updated when new KF made since past variables will not be updated
  // same when marginalized
  // vector of frames and timestamps related to variables since last KF?
  // first frame is KF and last is k
  FrameIds frames_since_lKF_;
  std::vector<Timestamp> timestamps_since_lKF_;

  std::shared_ptr<RGBDCamera> rgbd_camera_;
  gtsam::Cal3_S2Stereo::shared_ptr stereo_calibration_;

  // Trajectory up to the current KF
  // Nont only will this trajectory be "frozen" (in the sense that)
  // no motions will be in the current state
  // but also all motions will be related to a different KeyMotion pose
  PoseWithMotionTrajectory trajectory_till_lKF_;
  FrameRangeData<gtsam::Pose3> keyframe_range_;

  /** Create default parameters */
  // static gtsam::ISAM2Params DefaultISAM2Params() {
  //   gtsam::ISAM2Params params;
  //   params.findUnusedFactorSlots = true;
  //   params.keyFormatter = DynosamKeyFormatter;
  //   params.relinearizeThreshold = 0.01;
  //   // this value is very important for accuracy
  //   params.relinearizeSkip = 1;
  //   params.evaluateNonlinearError = true;
  //   return params;
  // }
  static dyno::ISAM2Params DefaultISAM2Params() {
    dyno::ISAM2Params params;
    params.findUnusedFactorSlots = true;
    // OKAY this seems to be extremely important!
    // when cacheLinearizedFactors = true (default) at least on gtsam 4.2.0
    // then when collecting the factors that are affected by new variables
    // relinearizeAffectedFactors checks if the effected keys are also part of
    // the relinearization keys as provided by both the fluid relin check and as
    // part of the additional params. New affected keys as part of smart factors
    // are NOT part of the relin keys and therefore the cached LINEAR factor
    // will be used (which does not include the new variable!)
    params.cacheLinearizedFactors = false;
    params.keyFormatter = DynosamKeyFormatter;
    params.relinearizeThreshold = 0.01;
    // this value is very important for accuracy
    // params.relinearizeSkip = 1;
    params.evaluateNonlinearError = true;
    return params;
  }

  //! Updated every update and includes only values in the smoother
  gtsam::Values smoother_state_;
  //! Best estimate of all values since the last KF
  //! May include more values that what is currently in the smoother window
  gtsam::Values state_since_lKF_;

  /** An iSAM2 object used to perform inference. The smoother lag is controlled
   * by what factors are removed each iteration */
  // gtsam::ISAM2 isam_;
  dyno::ISAM2 isam_;

  /** Store results of latest isam2 update */
  // gtsam::ISAM2Result isamResult_;
  dyno::ISAM2Result isamResult_;

  std::vector<DebugResult> debug_results_;

  gtsam::FastMap<FrameId, gtsam::Pose3> camera_poses_;

  /// SmartFactor stuff
  FactorMap<gtsam::SmartStereoProjectionPoseFactor::shared_ptr> factor_map_;
  gtsam::FastMap<gtsam::SmartStereoProjectionPoseFactor::shared_ptr, TrackletId>
      factor_to_tracklet_id_;

  // Keyframe LM solve stuff
  gtsam::NonlinearFactorGraph KF_factors_;
  gtsam::Values KF_values_;

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
      const KeyTimestampMap&, const gtsam::FactorIndices&) override {
    throw DynosamException("Not implemented!");
  }

 private:
};

using json = nlohmann::json;
void to_json(json& j, const HybridObjectMotionSmoother::Result& result);
void to_json(json& j, const HybridObjectMotionSmoother::DebugResult& result);

}  // namespace dyno
