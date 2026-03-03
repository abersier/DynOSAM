#include "dynosam/frontend/solvers/HybridObjectMotionSmoother.hpp"

#include <gtsam/linear/NoiseModel.h>

#include "dynosam/factors/HybridFormulationFactors.hpp"
#include "dynosam_common/utils/TimingStats.hpp"
#include "dynosam_opt/FactorGraphTools.hpp"
#include "dynosam_opt/IncrementalOptimization.hpp"
#include "dynosam_opt/Symbols.hpp"

namespace dyno {

gtsam::Symbol PointSymbol(TrackletId tracklet_id) {
  return gtsam::Symbol(kDynamicLandmarkSymbolChar, tracklet_id);
}

HybridObjectMotionSmoother::Ptr
HybridObjectMotionSmoother::CreateWithInitialMotion(
    const ObjectId object_id, double smoother_lag, const gtsam::Pose3& L_KF_km1,
    Frame::Ptr frame_km1, const TrackletIds& tracklets, const Solver& solver) {
  auto smoother = std::shared_ptr<HybridObjectMotionSmoother>(
      new HybridObjectMotionSmoother(object_id, frame_km1->getCamera(),
                                     smoother_lag, solver));

  smoother->createNewKeyedMotion(L_KF_km1, frame_km1, tracklets);
  return smoother;
}

// TODO: really should initalise with frame and tracklet ids...
HybridObjectMotionSmoother::HybridObjectMotionSmoother(ObjectId object_id,
                                                       Camera::Ptr camera,
                                                       double smootherLag,
                                                       const Solver& solver)
    : HybridObjectMotionSolverImpl(object_id, camera),
      gtsam::FixedLagSmoother(smootherLag),
      logger_prefix_("hybrid_motion_smoother_j" + std::to_string(object_id)),
      isam_(DefaultISAM2Params()) {
  CHECK_NOTNULL(stereo_calibration_);

  if (solver == Solver::Smart) {
    LOG(INFO) << "Running smoother in smart mode";
    get_keyframe_motion_impl =
        std::bind(&HybridObjectMotionSmoother::keyFrameMotionSmart, this,
                  std::placeholders::_1, std::placeholders::_2);
    update_motion_from_initial_impl = std::bind(
        &HybridObjectMotionSmoother::updateFromInitialMotionSmart, this,
        std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
  } else if (solver == Solver::Full) {
    LOG(INFO) << "Running smoother in full state mode";
    get_keyframe_motion_impl =
        std::bind(&HybridObjectMotionSmoother::keyFrameMotionFullState, this,
                  std::placeholders::_1, std::placeholders::_2);
    update_motion_from_initial_impl = std::bind(
        &HybridObjectMotionSmoother::updateFromInitialMotionFullState, this,
        std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
  } else if (solver == Solver::MotionOnly) {
    LOG(INFO) << "Running smoother in motion only mode";
    // both represent motion using H_W_KF_k so we can use
    // keyFrameMotionFullState
    get_keyframe_motion_impl =
        std::bind(&HybridObjectMotionSmoother::keyFrameMotionFullState, this,
                  std::placeholders::_1, std::placeholders::_2);
    update_motion_from_initial_impl = std::bind(
        &HybridObjectMotionSmoother::updateFromInitialMotionOnly, this,
        std::placeholders::_1, std::placeholders::_2, std::placeholders::_3);
  }
}

HybridObjectMotionSmoother::~HybridObjectMotionSmoother() {
  if (!debug_results_.empty()) {
    // const std::string file_name = logger_prefix_ + "_debug.bson";
    // LOG(INFO) << "Writing solver debug file: " << file_name;
    // const std::string file_path = getOutputFilePath(file_name);
    // JsonConverter::WriteOutJson(debug_results_, file_path,
    //                             JsonConverter::Format::BSON);
  }
}

PoseWithMotionTrajectory HybridObjectMotionSmoother::trajectory() const {
  // only from KF -> k (assume continuous?)
  PoseWithMotionTrajectory trajectory = trajectory_upto_lKF_;
  // do not include KF in result from local trajectory
  // as it will already be in trajectory_till_lKF_
  constexpr static bool kIncludeKFInTrajectory = false;
  trajectory.insert(localTrajectoryImpl(kIncludeKFInTrajectory));

  return trajectory;
}

PoseWithMotionTrajectory HybridObjectMotionSmoother::localTrajectory() const {
  constexpr static bool kIncludeKFInTrajectory = true;
  return localTrajectoryImpl(kIncludeKFInTrajectory);
}

gtsam::Pose3 HybridObjectMotionSmoother::keyFrameMotion() const {
  const gtsam::Symbol sym(ObjectMotionSymbol(object_id_, frameId()));
  // TODO: bring back!
  //  CHECK(isam_.valueExists(sym));
  CHECK(smoother_state_.exists(sym));
  // const gtsam::Pose3 L_W_KF = keyFramePose();
  // const gtsam::Pose3 H_W_KF_k = smoother_state_.at<gtsam::Pose3>(sym);
  // const gtsam::Pose3 G_W_KF_k =
  // smoother_state_.at<gtsam::Pose3>(sym).inverse(); const gtsam::Pose3
  // H_W_KF_k = camera_poses_.at(frameId()) * G_W_KF_k * L_W_KF.inverse();
  const gtsam::Pose3 H_W_KF_k =
      get_keyframe_motion_impl(frameId(), smoother_state_);

  return H_W_KF_k;
}

Motion3ReferenceFrame HybridObjectMotionSmoother::frameToFrameMotionReference()
    const {
  const gtsam::Pose3 H_W_KF_k = keyFrameMotion();
  if (keyFrameId() == frameId()) {
    return Motion3ReferenceFrame(H_W_KF_k, Motion3ReferenceFrame::Style::F2F,
                                 ReferenceFrame::GLOBAL, keyFrameId(),
                                 frameId());
  }
  const gtsam::Pose3 L_W_KF = keyFramePose();

  FrameId frame_id_km1 = frameId() - 1u;

  const gtsam::Symbol prev_motion_symbol(
      ObjectMotionSymbol(object_id_, frame_id_km1));

  //. TODO: bring back check!
  CHECK(smoother_state_.exists(prev_motion_symbol));
  // CHECK(isam_.valueExists(prev_motion_symbol))
  //     << DynosamKeyFormatter(prev_motion_symbol);
  // const gtsam::Pose3 H_W_KF_km1 =
  //     smoother_state_.at<gtsam::Pose3>(prev_motion_symbol);
  // const gtsam::Pose3 G_W_KF_km1 =
  // smoother_state_.at<gtsam::Pose3>(prev_motion_symbol).inverse(); const
  // gtsam::Pose3 H_W_KF_km1 = camera_poses_.at(frame_id_km1) * G_W_KF_km1 *
  // L_W_KF.inverse();
  const gtsam::Pose3 H_W_KF_km1 =
      get_keyframe_motion_impl(frame_id_km1, smoother_state_);

  gtsam::Pose3 H_W_km1_k = H_W_KF_k * H_W_KF_km1.inverse();
  return Motion3ReferenceFrame(H_W_km1_k, Motion3ReferenceFrame::Style::F2F,
                               ReferenceFrame::GLOBAL, frame_id_km1, frameId());
}

gtsam::FastMap<TrackletId, gtsam::Point3>
HybridObjectMotionSmoother::getObjectPoints() const {
  // TODO: smoother state or LKF state?
  const std::map<gtsam::Key, gtsam::Point3> keyed_object_point_map =
      getObjectPointsFromSmootherState();

  gtsam::FastMap<TrackletId, gtsam::Point3> object_point_map;
  for (const auto& [key, point] : keyed_object_point_map) {
    gtsam::Symbol sym(key);
    TrackletId tracklet_id = (TrackletId)sym.index();
    object_point_map.insert2(tracklet_id, point);
  }
  return object_point_map;
}

gtsam::Pose3 HybridObjectMotionSmoother::keyFramePose() const {
  auto kf_data = keyframe_range_.find(frameId());
  CHECK(kf_data);

  const auto [_, LKF] = *kf_data;
  return LKF;
}

bool HybridObjectMotionSmoother::createNewKeyedMotion(
    const gtsam::Pose3& L_KF, Frame::Ptr frame, const TrackletIds& tracklets) {
  if (VLOG_IS_ON(10)) {
    const std::string current_frame =
        frames_since_lKF_.empty() ? "None" : std::to_string(frameId());
    const std::string current_KF =
        frames_since_lKF_.empty() ? "None" : std::to_string(keyFrameId());
    VLOG(10) << "Creating new KeyMotion "
             << info_string(frame->getFrameId(), object_id_)
             << " current k=" << current_frame << " KF=" << current_KF;
  }

  // isam_ = gtsam::ISAM2(DefaultISAM2Params());
  isam_ = dyno::ISAM2(DefaultISAM2Params());

  // update fixed trajectory using current kf state
  // must do this before temporal/keyframe data-structures are reset
  // relies on state_since_lKF_ to fill trajectory values
  // trajectory includes keyframe!
  PoseWithMotionTrajectory trajectory_till_lKF;
  if (trajectory_upto_lKF_.empty()) {
    // if the trajectory is currently emppty, get the full trajectory
    // which will include the keyframe as the first frame of the trajectory.
    trajectory_till_lKF = std::move(localTrajectory());
  } else {
    // get trajectory without keyframe (ie the first frame of the local traj)
    // since this frame will be the last frame of the current trajectory
    // (trajectory_upto_lKF_)
    constexpr static bool kIncludeKFInTrajectory = false;
    trajectory_till_lKF =
        std::move(localTrajectoryImpl(kIncludeKFInTrajectory));
  }
  trajectory_upto_lKF_.insert(trajectory_till_lKF);

  smoother_state_.clear();

  frames_since_lKF_.clear();
  timestamps_since_lKF_.clear();
  state_since_lKF_.clear();

  // clear internal timestamp mapping so as to not confuse the Fixed Lag
  timestampKeyMap_.clear();
  keyTimestampMap_.clear();

  factor_map_.clear();
  factor_to_tracklet_id_.clear();

  m_L_points_.clear();

  // NEW
  // TODO: not clearning camera poses?
  // TODO: and factor stuff!?

  // TODO: batch solver
  KF_values_.clear();
  KF_factors_.resize(0);

  keyframe_range_.startNewActiveRange(frame->getFrameId(), L_KF);

  // update and add points at initial frame corresponding with an identity
  // motion allow the update function to insert new frames and timestamps given
  // the frame
  return updateFromInitialMotion(gtsam::Pose3::Identity(), frame, tracklets)
      .solver_okay;
}

bool HybridObjectMotionSmoother::update(const gtsam::Pose3& H_W_km1_k_predict,
                                        Frame::Ptr frame,
                                        const TrackletIds& tracklets) {
  CHECK(!frames_since_lKF_.empty())
      << "HybridObjectMotionSmoother::update "
      << " cannot be called without first creating a valid MotionFrame!";

  gtsam::Pose3 H_W_KF_km1 = gtsam::Pose3::Identity();
  // if we have at least one previous entry (otherwise frameId behaviour is
  // undefined which arguanle is poor design!)
  // if (isam_.valueExists(ObjectMotionSymbol(object_id_, frameId()))) {
  if (smoother_state_.exists(ObjectMotionSymbol(object_id_, frameId()))) {
    // const gtsam::Pose3 G_W_KF_k = smoother_state_.at<gtsam::Pose3>(
    //   ObjectMotionSymbol(object_id_, frameId())).inverse();
    // const gtsam::Pose3 L_KF = keyFramePose();
    // H_W_KF_km1  = camera_poses_.at(frameId()) * G_W_KF_k * L_KF.inverse();
    H_W_KF_km1 = get_keyframe_motion_impl(frameId(), smoother_state_);

    // H_W_KF_km1 = smoother_state_.at<gtsam::Pose3>(
    //     ObjectMotionSymbol(object_id_, frameId()));
  }
  // propogate initial guess
  const gtsam::Pose3 H_W_KF_k = H_W_km1_k_predict * H_W_KF_km1;
  return updateFromInitialMotion(H_W_KF_k, frame, tracklets).solver_okay;
}

PoseWithMotionTrajectory HybridObjectMotionSmoother::localTrajectoryImpl(
    bool include_keyframe) const {
  CHECK_EQ(frames_since_lKF_.size(), timestamps_since_lKF_.size());

  if (frames_since_lKF_.empty()) {
    return PoseWithMotionTrajectory{};
  }

  auto kf_data = keyframe_range_.find(frameId());
  const auto [frame_KF_id, L_W_KF] = *kf_data;

  CHECK_EQ(frame_KF_id, keyFrameId());

  // build trajectory from best state estimate since last KF
  PoseWithMotionTrajectory local_trajectory;
  for (size_t i = 0; i < frames_since_lKF_.size(); i++) {
    const FrameId frame_id = frames_since_lKF_.at(i);
    const Timestamp timestamp = timestamps_since_lKF_.at(i);

    // sanity check that all frames are part of the same KF range
    CHECK(kf_data->contains(frame_id));

    const gtsam::Symbol H_key_k = ObjectMotionSymbol(object_id_, frame_id);
    // TODO: bit of a hack - in the case that the isam2 solver fails
    // e.g with ILS, the values will not be added to the smoother
    //  and therefore will not appear in state_since_lKF_
    //  currently just skip!!
    if (!state_since_lKF_.exists(H_key_k)) {
      continue;
    }
    CHECK(state_since_lKF_.exists(H_key_k)) << DynosamKeyFormatter(H_key_k);

    // // const gtsam::Pose3 H_W_KF_k =
    // state_since_lKF_.at<gtsam::Pose3>(H_key_k); const gtsam::Pose3 G_W_KF_k =
    // state_since_lKF_.at<gtsam::Pose3>(H_key_k).inverse(); const gtsam::Pose3
    // H_W_KF_k = camera_poses_.at(frame_id) * G_W_KF_k * L_W_KF.inverse();
    const gtsam::Pose3 H_W_KF_k =
        get_keyframe_motion_impl(frame_id, state_since_lKF_);

    Motion3ReferenceFrame f2f_motion;
    if (i == 0) {
      CHECK_EQ(frame_id, frame_KF_id);

      // skip this frame if requested
      if (!include_keyframe) {
        continue;
      }

      f2f_motion =
          Motion3ReferenceFrame(H_W_KF_k, Motion3ReferenceFrame::Style::F2F,
                                ReferenceFrame::GLOBAL, frame_id, frame_id);
    } else {
      FrameId frame_id_km1 = frame_id - 1u;
      // sanity check that the previous frame is frame id -1 (ie. we're
      // consecutive!)
      CHECK_EQ(frame_id_km1, frames_since_lKF_.at(i - 1));

      const gtsam::Symbol H_key_km1 =
          ObjectMotionSymbol(object_id_, frame_id - 1u);
      CHECK(state_since_lKF_.exists(H_key_km1));

      // const gtsam::Pose3 H_W_KF_km1 =
      //     state_since_lKF_.at<gtsam::Pose3>(H_key_km1);
      // const gtsam::Pose3 G_W_KF_km1 =
      // state_since_lKF_.at<gtsam::Pose3>(H_key_km1).inverse(); const
      // gtsam::Pose3 H_W_KF_km1 = camera_poses_.at(frame_id - 1u) * G_W_KF_km1
      // * L_W_KF.inverse();
      const gtsam::Pose3 H_W_KF_km1 =
          get_keyframe_motion_impl(frame_id - 1u, state_since_lKF_);
      const gtsam::Pose3 H_W_km1_k = H_W_KF_k * H_W_KF_km1.inverse();

      f2f_motion = Motion3ReferenceFrame(
          H_W_km1_k, Motion3ReferenceFrame::Style::F2F, ReferenceFrame::GLOBAL,
          frame_id - 1u, frame_id);
    }

    const gtsam::Pose3 L_W_k = H_W_KF_k * L_W_KF;

    local_trajectory.insert(frame_id, timestamp, {L_W_k, f2f_motion});
  }

  return local_trajectory;
}

HybridObjectMotionSmoother::Result
HybridObjectMotionSmoother::updateFromInitialMotionFullState(
    const gtsam::Pose3& H_W_KF_k_initial, Frame::Ptr frame,
    const TrackletIds& tracklets) {
  const FrameId frame_id = frame->getFrameId();
  const Timestamp timestamp = frame->getTimestamp();

  // update temporal data-structure immediately so frameId() and
  // keyFrameId() functions work
  frames_since_lKF_.push_back(frame_id);
  timestamps_since_lKF_.push_back(timestamp);

  const double frame_as_double = static_cast<double>(frame_id);

  const gtsam::Symbol H_key_k = ObjectMotionSymbol(object_id_, frame_id);
  // fixed camera pose
  const gtsam::Pose3 X_W_k = frame->getPose();
  // get current keyframe pose
  const gtsam::Pose3 L_KF = keyFramePose();

  gtsam::Values new_values;
  gtsam::NonlinearFactorGraph new_factors;

  KeyTimestampMap timestamps;
  // add motions
  timestamps[H_key_k] = frame_as_double;
  new_values.insert(H_key_k, H_W_KF_k_initial);

  gtsam::SharedNoiseModel stereo_noise_model =
      gtsam::noiseModel::Isotropic::Sigma(3u, 2.0);
  stereo_noise_model =
      factor_graph_tools::robustifyHuber(0.01, stereo_noise_model);
  CHECK(stereo_noise_model);

  // for debug stats
  size_t num_tracks_used = 0;
  size_t avg_feature_age = 0;

  for (const TrackletId& tracklet_id : tracklets) {
    const Feature::Ptr feature = frame->at(tracklet_id);
    CHECK(feature);

    const gtsam::Symbol m_key(PointSymbol(tracklet_id));

    const auto [stereo_keypoint_status, stereo_measurement] =
        rgbd_camera_->getStereo(feature);

    if (!stereo_keypoint_status) {
      continue;
    }

    bool is_new = false;
    // if variable is removed (ie due to marginalization)!
    // this is re-initalizing it!!! Is this what we want
    // to do!!?
    if (!isam_.valueExists(m_key)) {
      const gtsam::Point3 m_X_k = frame->backProjectToCamera(tracklet_id);
      Landmark m_L_init = HybridObjectMotion::projectToObject3(
          X_W_k, H_W_KF_k_initial, L_KF, m_X_k);

      new_values.insert(m_key, m_L_init);
      is_new = true;
    }

    num_tracks_used++;
    avg_feature_age += feature->age();

    timestamps[m_key] = frame_as_double;

    auto factor = boost::make_shared<StereoHybridMotionFactor2>(
        stereo_measurement, L_KF, X_W_k, stereo_noise_model,
        stereo_calibration_, H_key_k, m_key, true /*throw ceirality*/
    );

    CHECK(factor);

    new_factors += factor;
  }

  if (num_tracks_used == 0) {
    HybridObjectMotionSmoother::Result result;
    result.solver_okay = false;
    return result;
  }

  avg_feature_age /= num_tracks_used;

  if (frameId() == keyFrameId()) {
    gtsam::SharedNoiseModel identity_motion_model =
        gtsam::noiseModel::Isotropic::Sigma(6u, 0.01);

    // TODO: add prior on this first motion to make it identity!
    new_factors.addPrior<gtsam::Pose3>(H_key_k, gtsam::Pose3::Identity(),
                                       identity_motion_model);
  }

  if (frame_id > 2) {
    const gtsam::Symbol H_key_km1 =
        ObjectMotionSymbol(object_id_, frame_id - 1u);
    const gtsam::Symbol H_key_km2 =
        ObjectMotionSymbol(object_id_, frame_id - 2u);

    // TODO: params
    gtsam::SharedNoiseModel smoothing_motion_model =
        gtsam::noiseModel::Isotropic::Sigma(6u, 0.1);

    // TODO: ALL motions should use the same L_KF_
    //  if L_KF_ is only updated when we reset internal ISAM then no problem!
    if (isam_.valueExists(H_key_km1) && isam_.valueExists(H_key_km2)) {
      VLOG(10) << "Adding smoothing factor "
               << info_string(frame_id, object_id_);
      new_factors.emplace_shared<HybridSmoothingFactor>(
          H_key_km2, H_key_km1, H_key_k, L_KF, smoothing_motion_model);
    }
  }

  auto result = this->updateSmoother(new_factors, new_values, timestamps,
                                     ISAM2UpdateParams{});

  smoother_state_ = calculateEstimate();
  state_since_lKF_.insert_or_assign(smoother_state_);

  DebugResult debug_result;
  debug_result.result = result;

  // TODO: debug flag
  debug_result.smoother_stats.fill(&isam_);
  debug_result.object_id = object_id_;
  debug_result.frame_id = frame_id;
  debug_result.timestamp = timestamp;
  debug_result.frame_id_KF = keyFrameId();

  debug_result.num_tracks = num_tracks_used;
  debug_result.average_feature_age = avg_feature_age;

  debug_result.num_landmarks_in_smoother =
      getObjectPointsFromSmootherState().size();
  debug_result.num_motions_in_smoother =
      getObjectMotionsFromSmootherState().size();

  debug_results_.push_back(std::move(debug_result));

  return result;
}

HybridObjectMotionSmoother::Result
HybridObjectMotionSmoother::updateFromInitialMotionSmart(
    const gtsam::Pose3& H_W_KF_k_initial, Frame::Ptr frame,
    const TrackletIds& tracklets) {
  const FrameId frame_id = frame->getFrameId();
  const Timestamp timestamp = frame->getTimestamp();

  // update temporal data-structure immediately so frameId() and
  // keyFrameId() functions work
  frames_since_lKF_.push_back(frame_id);
  timestamps_since_lKF_.push_back(timestamp);

  const double frame_as_double = static_cast<double>(frame_id);

  const gtsam::Symbol H_key_k = ObjectMotionSymbol(object_id_, frame_id);
  // fixed camera pose
  const gtsam::Pose3 X_W_k = frame->getPose();
  // get current keyframe pose
  const gtsam::Pose3 L_KF = keyFramePose();

  camera_poses_.insert2(frame_id, X_W_k);

  gtsam::Values new_values;
  gtsam::NonlinearFactorGraph new_factors;

  KeyTimestampMap timestamps;
  // add motions
  timestamps[H_key_k] = frame_as_double;

  const gtsam::Pose3 G_W = X_W_k.inverse() * H_W_KF_k_initial * L_KF;
  new_values.insert(H_key_k, G_W.inverse());

  gtsam::SharedNoiseModel stereo_noise_model =
      gtsam::noiseModel::Isotropic::Sigma(3u, 2.0);
  // stereo_noise_model =
  //     factor_graph_tools::robustifyHuber(0.01, stereo_noise_model);
  CHECK(stereo_noise_model);

  // for debug stats
  size_t num_tracks_used = 0;
  size_t avg_feature_age = 0;

  gtsam::FastMap<gtsam::FactorIndex, gtsam::KeySet> newly_affected_keys;

  for (const TrackletId& tracklet_id : tracklets) {
    const Feature::Ptr feature = frame->at(tracklet_id);
    CHECK(feature);

    const gtsam::Symbol m_key(PointSymbol(tracklet_id));

    // bool is_new = false;
    // if variable is removed (ie due to marginalization)!
    // this is re-initalizing it!!! Is this what we want
    // to do!!?
    // if (!isam_.valueExists(m_key)) {
    //   const gtsam::Point3 m_X_k = frame->backProjectToCamera(tracklet_id);
    //   Landmark m_L_init = HybridObjectMotion::projectToObject3(
    //       X_W_k, H_W_KF_k_initial, L_KF, m_X_k);

    //   new_values.insert(m_key, m_L_init);
    //   is_new = true;
    // }
    const auto [stereo_keypoint_status, stereo_measurement] =
        rgbd_camera_->getStereo(feature);
    if (!stereo_keypoint_status) {
      continue;
    }

    if (!factor_map_.exists(tracklet_id)) {
      const gtsam::Point3 m_X_k = frame->backProjectToCamera(tracklet_id);
      Landmark m_L_init = HybridObjectMotion::projectToObject3(
          X_W_k, H_W_KF_k_initial, L_KF, m_X_k);

      gtsam::SmartStereoProjectionParams smart_factor_params;
      // for some reason JACOBIAN_SVD more stable (we get const char*
      // exception!?) smart_factor_params.linearizationMode =
      //   gtsam::LinearizationMode::JACOBIAN_SVD;
      smart_factor_params.setLinearizationMode(gtsam::HESSIAN);
      smart_factor_params.setDegeneracyMode(gtsam::ZERO_ON_DEGENERACY);
      smart_factor_params.setDynamicOutlierRejectionThreshold(8.0);
      smart_factor_params.setRetriangulationThreshold(1.0e-3);

      // totalReprojectionError
      auto factor = boost::make_shared<gtsam::SmartStereoProjectionPoseFactor>(
          stereo_noise_model, smart_factor_params);
      // hacky way to force the result to get set!
      // factor->totalReprojectionError({}, m_L_init);
      // factor->point()

      // must happen before updating new factors
      const Slot starting_slot = new_factors.size();
      factor_map_.insert2(tracklet_id, std::make_pair(factor, starting_slot));
      factor_to_tracklet_id_.insert2(factor, tracklet_id);
      // newly_affected_keys.insert2(starting_slot, {H_key_k});

      // only add when new factors!
      new_factors += factor;
    } else {
      Slot current_slot = factor_map_.at(tracklet_id).second;
      // factor_map_.at(tracklet_id).first->print("SS", DynosamKeyFormatter);
      newly_affected_keys.insert2(static_cast<gtsam::FactorIndex>(current_slot),
                                  {H_key_k});

      {
        // test!
        const auto factors_in_smoother = getFactors();
        CHECK_LT(current_slot, factors_in_smoother.size());

        auto this_factor = factor_map_.at(tracklet_id).first;
        auto factor_in_smoother = factors_in_smoother.at(current_slot);
        CHECK(this_factor->equals(*factor_in_smoother));
      }
    }

    auto factor = factor_map_.at(tracklet_id).first;
    factor->add(stereo_measurement, H_key_k, stereo_calibration_);

    num_tracks_used++;
    avg_feature_age += feature->age();

    // timestamps[m_key] = frame_as_double;

    // auto factor = boost::make_shared<StereoHybridMotionFactor2>(
    //     stereo_measurement, L_KF, X_W_k, stereo_noise_model,
    //     stereo_calibration_, H_key_k, m_key, true /*throw ceirality*/
    // );
    // CHECK(factor);
  }

  if (num_tracks_used == 0) {
    HybridObjectMotionSmoother::Result result;
    result.solver_okay = false;
    return result;
  }

  avg_feature_age /= num_tracks_used;

  if (frameId() == keyFrameId()) {
    gtsam::SharedNoiseModel identity_motion_model =
        gtsam::noiseModel::Isotropic::Sigma(6u, 0.0001);

    // TODO: add prior on this first motion to make it identity!
    // H_W is identity on the first motion so equation reduces
    const gtsam::Pose3 G_W_I = X_W_k.inverse() * L_KF;
    new_factors.addPrior<gtsam::Pose3>(H_key_k, G_W_I.inverse(),
                                       identity_motion_model);
  }

  if (frame_id > 2) {
    const gtsam::Symbol H_key_km1 =
        ObjectMotionSymbol(object_id_, frame_id - 1u);
    const gtsam::Symbol H_key_km2 =
        ObjectMotionSymbol(object_id_, frame_id - 2u);

    // TODO: params
    gtsam::SharedNoiseModel smoothing_motion_model =
        gtsam::noiseModel::Isotropic::Sigma(6u, 0.1);

    // TODO: ALL motions should use the same L_KF_
    //  if L_KF_ is only updated when we reset internal ISAM then no problem!
    // smoother is different now as we estimate for G_w!
    // if (isam_.valueExists(H_key_km1) && isam_.valueExists(H_key_km2)) {
    //   VLOG(10) << "Adding smoothing factor "
    //            << info_string(frame_id, object_id_);
    //   new_factors.emplace_shared<HybridSmoothingFactor>(
    //       H_key_km2, H_key_km1, H_key_k, L_KF, smoothing_motion_model);
    // }
  }

  dyno::ISAM2UpdateParams update_params;
  update_params.newAffectedKeys = std::move(newly_affected_keys);

  HybridObjectMotionSmoother::Result result = this->updateSmoother(
      new_factors, new_values, KeyTimestampMap{}, update_params);

  if (!result.solver_okay) {
    return result;
  }

  // update smoother slots
  const auto& isam_result = result.isam_result;
  const gtsam::FactorIndices& new_factor_indicies =
      isam_result.newFactorsIndices;

  if (isam_result.errorBefore && isam_result.errorAfter) {
    LOG(INFO) << "ISAM error - before: " << isam_result.getErrorBefore()
              << " after: " << isam_result.getErrorAfter();
  }

  const auto factors_in_smoother = getFactors();
  for (size_t i = 0; i < new_factors.size(); i++) {
    gtsam::FactorIndex new_index = new_factor_indicies.at(i);
    auto nonlinear_factor = new_factors.at(i);
    CHECK_EQ(nonlinear_factor, factors_in_smoother.at(new_index));

    auto smart_factor =
        boost::dynamic_pointer_cast<gtsam::SmartStereoProjectionPoseFactor>(
            nonlinear_factor);
    if (smart_factor) {
      CHECK(factor_to_tracklet_id_.exists(smart_factor));
      TrackletId tracklet_for_factor = factor_to_tracklet_id_.at(smart_factor);

      // update slot!
      factor_map_.at(tracklet_for_factor).second = static_cast<Slot>(new_index);

      CHECK_EQ(factor_map_.at(tracklet_for_factor).first, smart_factor);
    }
  }

  // deal with keys that were marginalized
  const gtsam::KeyVector& marginalized_keys = result.marginalized_keys;
  const gtsam::KeyVector& keys_in_smoother = getLinearizationPoint().keys();

  // should be calculateEstimate!
  // smoother_state_.insert_or_assign(new_values);
  smoother_state_ = calculateEstimate();

  // HACK! (fill smoother states with points!!!)
  for (const auto& [tracklet_id, factor_pair] : factor_map_) {
    const gtsam::Symbol m_key(PointSymbol(tracklet_id));

    const Slot slot = factor_pair.second;
    const auto smart_factor = factor_pair.first;

    gtsam::TriangulationResult triangulation_result = smart_factor->point();

    if (triangulation_result) {
      smoother_state_.insert(m_key, *triangulation_result);
    }
  }

  state_since_lKF_.insert_or_assign(smoother_state_);

  DebugResult debug_result;
  debug_result.result = result;

  // TODO: debug flag
  // debug_result.smoother_stats.fill(&isam_);
  debug_result.object_id = object_id_;
  debug_result.frame_id = frame_id;
  debug_result.timestamp = timestamp;
  debug_result.frame_id_KF = keyFrameId();

  debug_result.num_tracks = num_tracks_used;
  debug_result.average_feature_age = avg_feature_age;

  debug_result.num_landmarks_in_smoother =
      getObjectPointsFromSmootherState().size();
  debug_result.num_motions_in_smoother =
      getObjectMotionsFromSmootherState().size();

  debug_results_.push_back(std::move(debug_result));

  return result;
}

HybridObjectMotionSmoother::Result
HybridObjectMotionSmoother::updateFromInitialMotionOnly(
    const gtsam::Pose3& H_W_KF_k_initial, Frame::Ptr frame,
    const TrackletIds& tracklets) {
  const FrameId frame_id = frame->getFrameId();
  const Timestamp timestamp = frame->getTimestamp();

  // update temporal data-structure immediately so frameId() and
  // keyFrameId() functions work
  frames_since_lKF_.push_back(frame_id);
  timestamps_since_lKF_.push_back(timestamp);

  const double frame_as_double = static_cast<double>(frame_id);

  const gtsam::Symbol H_key_k = ObjectMotionSymbol(object_id_, frame_id);
  // fixed camera pose
  const gtsam::Pose3 X_W_k = frame->getPose();
  // get current keyframe pose
  const gtsam::Pose3 L_KF = keyFramePose();

  camera_poses_.insert2(frame_id, X_W_k);

  gtsam::Values new_values;
  gtsam::NonlinearFactorGraph new_factors;

  KeyTimestampMap timestamps;
  // add motions
  timestamps[H_key_k] = frame_as_double;

  new_values.insert(H_key_k, H_W_KF_k_initial);

  gtsam::SharedNoiseModel stereo_noise_model =
      gtsam::noiseModel::Isotropic::Sigma(3u, 2.0);
  stereo_noise_model =
      factor_graph_tools::robustifyHuber(0.01, stereo_noise_model);

  // for debug stats
  size_t num_tracks_used = 0;
  size_t avg_feature_age = 0;

  for (const TrackletId& tracklet_id : tracklets) {
    const Feature::Ptr feature = frame->at(tracklet_id);
    CHECK(feature);

    const auto [stereo_keypoint_status, stereo_measurement] =
        rgbd_camera_->getStereo(feature);
    if (!stereo_keypoint_status) {
      continue;
    }

    if (!m_L_points_.exists(tracklet_id)) {
      const gtsam::Point3 m_X_k = frame->backProjectToCamera(tracklet_id);
      Landmark m_L_init = HybridObjectMotion::projectToObject3(
          X_W_k, H_W_KF_k_initial, L_KF, m_X_k);
      m_L_points_.insert2(tracklet_id, m_L_init);
    }

    auto factor = boost::make_shared<StereoHybridMotionFactor3>(
        stereo_measurement, L_KF, X_W_k, m_L_points_.at(tracklet_id),
        stereo_noise_model, stereo_calibration_, H_key_k, true);

    new_factors += factor;

    num_tracks_used++;
    avg_feature_age += feature->age();
  }

  if (num_tracks_used == 0) {
    HybridObjectMotionSmoother::Result result;
    result.solver_okay = false;
    return result;
  }

  avg_feature_age /= num_tracks_used;

  if (frameId() == keyFrameId()) {
    gtsam::SharedNoiseModel identity_motion_model =
        gtsam::noiseModel::Isotropic::Sigma(6u, 0.01);

    // TODO: add prior on this first motion to make it identity!
    new_factors.addPrior<gtsam::Pose3>(H_key_k, gtsam::Pose3::Identity(),
                                       identity_motion_model);
  }

  if (frame_id > 2) {
    const gtsam::Symbol H_key_km1 =
        ObjectMotionSymbol(object_id_, frame_id - 1u);
    const gtsam::Symbol H_key_km2 =
        ObjectMotionSymbol(object_id_, frame_id - 2u);

    // TODO: params
    gtsam::SharedNoiseModel smoothing_motion_model =
        gtsam::noiseModel::Isotropic::Sigma(6u, 0.1);

    // TODO: ALL motions should use the same L_KF_
    //  if L_KF_ is only updated when we reset internal ISAM then no problem!
    if (isam_.valueExists(H_key_km1) && isam_.valueExists(H_key_km2)) {
      VLOG(10) << "Adding smoothing factor "
               << info_string(frame_id, object_id_);
      new_factors.emplace_shared<HybridSmoothingFactor>(
          H_key_km2, H_key_km1, H_key_k, L_KF, smoothing_motion_model);
    }
  }

  auto result = this->updateSmoother(new_factors, new_values, timestamps,
                                     ISAM2UpdateParams{});

  smoother_state_ = calculateEstimate();

  // fill states with points
  for (const auto& [tracklet_id, point] : m_L_points_) {
    const gtsam::Symbol m_key(PointSymbol(tracklet_id));
    smoother_state_.insert(PointSymbol(tracklet_id), point);
  }

  state_since_lKF_.insert_or_assign(smoother_state_);

  // TODO: debug
  return result;
}

gtsam::Pose3 HybridObjectMotionSmoother::keyFrameMotionFullState(
    FrameId frame_id, const gtsam::Values& values) const {
  const gtsam::Symbol H_key_k = ObjectMotionSymbol(object_id_, frame_id);
  CHECK(values.exists(H_key_k));

  return values.at<gtsam::Pose3>(H_key_k);
}

gtsam::Pose3 HybridObjectMotionSmoother::keyFrameMotionSmart(
    FrameId frame_id, const gtsam::Values& values) const {
  const gtsam::Symbol H_key_k = ObjectMotionSymbol(object_id_, frame_id);
  CHECK(values.exists(H_key_k));

  auto kf_data = keyframe_range_.find(frame_id);
  const auto [_, L_W_KF] = *kf_data;

  const gtsam::Pose3 G_W_KF_k = values.at<gtsam::Pose3>(H_key_k).inverse();
  const gtsam::Pose3 H_W_KF_k =
      camera_poses_.at(frame_id) * G_W_KF_k * L_W_KF.inverse();
  return H_W_KF_k;
}

// HybridObjectMotionSmoother::Result
// HybridObjectMotionSmoother::updateFromInitialMotion(
//     const gtsam::Pose3& H_W_KF_k_initial, Frame::Ptr frame,
//     const TrackletIds& tracklets) {
//   // using frame_id before update

//   const FrameId frame_id = frame->getFrameId();

//   // update temporal data-structure immediately so frameId() and
//   // keyFrameId() functions work
//   const bool is_keyframe = frames_since_lKF_.empty();

//   frames_since_lKF_.push_back(frame_id);
//   timestamps_since_lKF_.push_back(frame->getTimestamp());

//   if (is_keyframe) {
//     CHECK(frameId() == keyFrameId());
//     LOG(INFO) << "IS KF!";
//   }

//   const double frame_as_double = static_cast<double>(frame_id);

//   const gtsam::Symbol H_key_k = ObjectMotionSymbol(object_id_, frame_id);
//   // fixed camera pose
//   const gtsam::Pose3 X_W_k = frame->getPose();
//   // get current keyframe pose
//   const gtsam::Pose3 L_KF = keyFramePose();

//   camera_poses_.insert2(frame_id, X_W_k);

//   gtsam::Values new_values;
//   gtsam::NonlinearFactorGraph new_factors;

//   KeyTimestampMap timestamps;
//   // add motions
//   timestamps[H_key_k] = frame_as_double;

//   new_values.insert(H_key_k, H_W_KF_k_initial);

//   gtsam::SharedNoiseModel stereo_noise_model =
//       gtsam::noiseModel::Isotropic::Sigma(3u, 2.0);
//   stereo_noise_model =
//       factor_graph_tools::robustifyHuber(0.01, stereo_noise_model);
//   CHECK(stereo_noise_model);

//   // for debug stats
//   size_t num_tracks_used = 0;
//   size_t avg_feature_age = 0;

//   for (const TrackletId& tracklet_id : tracklets) {
//     const Feature::Ptr feature = frame->at(tracklet_id);
//     CHECK(feature);

//     const gtsam::Symbol m_key(PointSymbol(tracklet_id));

//     // bool is_new = false;
//     // if variable is removed (ie due to marginalization)!
//     // this is re-initalizing it!!! Is this what we want
//     // to do!!?
//     // if (!isam_.valueExists(m_key)) {
//     //   const gtsam::Point3 m_X_k = frame->backProjectToCamera(tracklet_id);
//     //   Landmark m_L_init = HybridObjectMotion::projectToObject3(
//     //       X_W_k, H_W_KF_k_initial, L_KF, m_X_k);

//     //   new_values.insert(m_key, m_L_init);
//     //   is_new = true;
//     // }
//     const auto [stereo_keypoint_status, stereo_measurement] =
//         rgbd_camera_->getStereo(feature);
//     if (!stereo_keypoint_status) {
//       continue;
//     }

//     // encapsulates all values since the last KF including the ones being
//     // optimisated if(!KF_values_.exists(m_key)) {
//     // assume we only make new featues when KF
//     if (is_keyframe) {
//       const gtsam::Point3 m_X_k = frame->backProjectToCamera(tracklet_id);
//       Landmark m_L_init = HybridObjectMotion::projectToObject3(
//           X_W_k, H_W_KF_k_initial, L_KF, m_X_k);

//       new_values.insert(m_key, m_L_init);
//       LOG(INFO) << "Initalising new point " << DynosamKeyFormatter(m_key);
//     } else {
//       CHECK(KF_values_.exists(m_key)) << DynosamKeyFormatter(m_key);
//     }
//     num_tracks_used++;
//     avg_feature_age += feature->age();

//     // timestamps[m_key] = frame_as_double;

//     auto factor = boost::make_shared<StereoHybridMotionFactor2>(
//         stereo_measurement, L_KF, X_W_k, stereo_noise_model,
//         stereo_calibration_, H_key_k, m_key, true /*throw ceirality*/
//     );
//     CHECK(factor);
//     new_factors += factor;
//   }

//   avg_feature_age /= num_tracks_used;

//   if (frameId() == keyFrameId()) {
//     CHECK(is_keyframe);
//     gtsam::SharedNoiseModel identity_motion_model =
//         gtsam::noiseModel::Isotropic::Sigma(6u, 0.01);

//     // TODO: add prior on this first motion to make it identity!
//     // H_W is identity on the first motion so equation reduces
//     // const gtsam::Pose3 G_W_I = X_W_k.inverse() * L_KF;
//     // new_factors.addPrior<gtsam::Pose3>(H_key_k, G_W_I.inverse(),
//     //                                    identity_motion_model);
//     new_factors.addPrior<gtsam::Pose3>(H_key_k, gtsam::Pose3::Identity(),
//                                        identity_motion_model);
//   }

//   if (frame_id > 2) {
//     const gtsam::Symbol H_key_km1 =
//         ObjectMotionSymbol(object_id_, frame_id - 1u);
//     const gtsam::Symbol H_key_km2 =
//         ObjectMotionSymbol(object_id_, frame_id - 2u);

//     // TODO: params
//     gtsam::SharedNoiseModel smoothing_motion_model =
//         gtsam::noiseModel::Isotropic::Sigma(6u, 0.1);

//     // TODO: ALL motions should use the same L_KF_
//     //  if L_KF_ is only updated when we reset internal ISAM then no problem!
//     // smoother is different now as we estimate for G_w!
//     // if (isam_.valueExists(H_key_km1) && isam_.valueExists(H_key_km2)) {
//     //   VLOG(10) << "Adding smoothing factor "
//     //            << info_string(frame_id, object_id_);
//     //   new_factors.emplace_shared<HybridSmoothingFactor>(
//     //       H_key_km2, H_key_km1, H_key_k, L_KF, smoothing_motion_model);
//     // }
//   }

//   if (is_keyframe) {
//     KF_values_ = std::move(new_values);
//     KF_factors_ = std::move(new_factors);
//     LOG(INFO) << "Is KF here";

//     smoother_state_.insert(KF_values_);
//   } else {
//     gtsam::Values values = KF_values_;
//     values.insert(new_values);

//     gtsam::NonlinearFactorGraph graph = KF_factors_;
//     graph += new_factors;

//     LOG(INFO) << "Performing batch opt new theta " << new_values.size()
//               << " new factors " << new_factors.size();

//     gtsam::LevenbergMarquardtParams solver_params;
//     solver_params.setMaxIterations(20);

//     gtsam::LevenbergMarquardtOptimizer solver(graph, values, solver_params);

//     try {
//       gtsam::Values refined_values = solver.optimize();

//       // update landmarks and optimised motions

//       auto refined_object_points = getObjectPointsFromState(refined_values);
//       for (const auto& [key, value] : refined_object_points) {
//         KF_values_.insert_or_assign(key, value);
//       }

//       const gtsam::Symbol kf_motion_symbol(
//           ObjectMotionSymbol(object_id_, keyFrameId()));
//       // always upda the motion directly becuase we know it will be in the
//       next
//       // state
//       KF_values_.update(kf_motion_symbol,
//                         refined_values.at<gtsam::Pose3>(kf_motion_symbol));

//       // no update to current motion becuase it will not be in the next
//       system
//       // to be solved
//       // // insert becuase new
//       // KF_values_.insert(H_key_k,
//       refined_values.at<gtsam::Pose3>(H_key_k));

//       // add all values!
//       smoother_state_.insert_or_assign(refined_values);
//     } catch (const gtsam::ValuesKeyDoesNotExist& e) {
//       LOG(FATAL) << "ValuesKeyDoesNotExist: " <<
//       DynosamKeyFormatter(e.key());
//     }
//   }

//   LOG(INFO) << "here";

//   // should be calculateEstimate!
//   // smoother_state_.insert_or_assign(new_values);
//   // smoother_state_ = calculateEstimate();
//   state_since_lKF_.insert_or_assign(smoother_state_);

//   LOG(INFO) << "here";

//   DebugResult debug_result;
//   // debug_result.result = result;

//   // TODO: debug flag
//   // debug_result.smoother_stats.fill(&isam_);
//   debug_result.object_id = object_id_;
//   debug_result.frame_id = frameId();
//   debug_result.timestamp = timestamp();
//   debug_result.frame_id_KF = keyFrameId();

//   debug_result.num_tracks = num_tracks_used;
//   debug_result.average_feature_age = avg_feature_age;

//   debug_result.num_landmarks_in_smoother =
//       getObjectPointsFromSmootherState().size();
//   debug_result.num_motions_in_smoother =
//       getObjectMotionsFromSmootherState().size();

//   debug_results_.push_back(std::move(debug_result));

//   return HybridObjectMotionSmoother::Result{};
// }

HybridObjectMotionSmoother::Result
HybridObjectMotionSmoother::updateFromInitialMotion(
    const gtsam::Pose3& H_W_KF_k_initial, Frame::Ptr frame,
    const TrackletIds& tracklets) {
  // using frame_id before update
  return update_motion_from_initial_impl(H_W_KF_k_initial, frame, tracklets);
}

HybridObjectMotionSmoother::Result HybridObjectMotionSmoother::updateSmoother(
    const gtsam::NonlinearFactorGraph& newFactors,
    const gtsam::Values& newTheta, const KeyTimestampMap& timestamps,
    const dyno::ISAM2UpdateParams& update_params) {
  gtsam::FastVector<size_t> removedFactors;
  boost::optional<gtsam::FastMap<gtsam::Key, int> > constrainedKeys = {};

  Result result;
  // Update the Timestamps associated with the factor keys
  updateKeyTimestampMap(timestamps);

  // Get current timestamp
  double current_timestamp = getCurrentTimestamp();
  LOG(INFO) << "Current timestamp: " << current_timestamp;

  // Find the set of variables to be marginalized out
  LOG(INFO) << "Findig keys before " << current_timestamp - smootherLag_;
  gtsam::KeyVector marginalizableKeys =
      findKeysBefore(current_timestamp - smootherLag_);
  result.marginalized_keys = marginalizableKeys;

  // Force iSAM2 to put the marginalizable variables at the beginning
  createOrderingConstraints(marginalizableKeys, constrainedKeys);

  std::cout << "Gets to marginalize due to filter: ";
  for (const auto& key : marginalizableKeys) {
    std::cout << DynosamKeyFormatter(key) << " ";
  }
  std::cout << std::endl;

  std::unordered_set<gtsam::Key> additionalKeys =
      BayesTreeMarginalizationHelper<
          dyno::ISAM2>::gatherAdditionalKeysToReEliminate(isam_,
                                                          marginalizableKeys);

  gtsam::KeyList additionalMarkedKeys(additionalKeys.begin(),
                                      additionalKeys.end());
  result.additional_keys_reeliminate = additionalMarkedKeys;

  dyno::ISAM2UpdateParams mutable_update_params = update_params;
  if (!mutable_update_params.extraReelimKeys) {
    mutable_update_params.extraReelimKeys = gtsam::KeyList{};
  }
  mutable_update_params.extraReelimKeys->insert(
      mutable_update_params.extraReelimKeys->begin(),
      additionalMarkedKeys.begin(), additionalMarkedKeys.end());

  // mutable_update_params.constrainedKeys = constrainedKeys;
  if (constrainedKeys) {
    mutable_update_params.constrainedKeys.emplace(*constrainedKeys);
  }

  utils::ChronoTimingStats update_timer(logger_prefix_ + ".isam_update", 10);

  // getDefaultILSErrorHandlingHooks(handle_failed_object)
  using SmootherInterface = IncrementalInterface<decltype(isam_)>;
  SmootherInterface smoother(&isam_);
  result.solver_okay = smoother.optimize(
      &isamResult_,
      [&](const SmootherInterface::Smoother&,
          SmootherInterface::UpdateArguments& update_arguments) {
        update_arguments.new_values = newTheta;
        update_arguments.new_factors = newFactors;
        update_arguments.update_params = mutable_update_params;
      },
      getDefaultILSErrorHandlingHooks());

  // isamResult_ = isam_.update(newFactors, newTheta, mutable_update_params);
  result.update_time_ms = update_timer.stop();
  result.isam_result = isamResult_;

  // Marginalize out any needed variables
  if (marginalizableKeys.size() > 0) {
    gtsam::FastList<gtsam::Key> leafKeys(marginalizableKeys.begin(),
                                         marginalizableKeys.end());
    utils::ChronoTimingStats marginalize_timer(
        logger_prefix_ + ".marginalize_leaves", 10);
    isam_.marginalizeLeaves(leafKeys);
    result.marginalize_time_ms = marginalize_timer.stop();
  }
  // Remove marginalized keys from the KeyTimestampMap
  eraseKeyTimestampMap(marginalizableKeys);

  return result;
}

std::map<gtsam::Key, gtsam::Point3>
HybridObjectMotionSmoother::getObjectPointsFromState(
    const gtsam::Values& values) const {
  return values.extract<gtsam::Point3>(
      Symbol::ChrTest(kDynamicLandmarkSymbolChar));
}

std::map<gtsam::Key, gtsam::Pose3>
HybridObjectMotionSmoother::getObjectMotionsFromState(
    const gtsam::Values& values) const {
  return values.extract<gtsam::Pose3>(Symbol::ChrTest(kObjectMotionSymbolChar));
}

void HybridObjectMotionSmoother::eraseKeysBefore(double timestamp) {
  TimestampKeyMap::iterator end = timestampKeyMap_.lower_bound(timestamp);
  TimestampKeyMap::iterator iter = timestampKeyMap_.begin();
  while (iter != end) {
    keyTimestampMap_.erase(iter->second);
    timestampKeyMap_.erase(iter++);
  }
}

/* ************************************************************************* */
void HybridObjectMotionSmoother::createOrderingConstraints(
    const gtsam::KeyVector& marginalizableKeys,
    boost::optional<gtsam::FastMap<gtsam::Key, int> >& constrainedKeys) const {
  if (marginalizableKeys.size() > 0) {
    constrainedKeys = gtsam::FastMap<gtsam::Key, int>();
    // Generate ordering constraints so that the marginalizable variables will
    // be eliminated first Set all variables to Group1
    for (const TimestampKeyMap::value_type& timestamp_key : timestampKeyMap_) {
      constrainedKeys->operator[](timestamp_key.second) = 1;
    }
    // Set marginalizable variables to Group0
    for (gtsam::Key key : marginalizableKeys) {
      constrainedKeys->operator[](key) = 0;
    }
  }
}

///// SmartSolver

// void to_json(json& j, const HybridObjectMotionSmoother::Result& result) {
//   j["marginalized_keys"] = result.marginalized_keys;
//   j["additional_keys_reeliminate"] = result.additional_keys_reeliminate;
//   j["update_time_ms"] = result.update_time_ms;
//   j["marginalize_time_ms"] = result.marginalize_time_ms;
//   j["isam_result"] = result.isam_result;
// }

// void to_json(json& j, const HybridObjectMotionSmoother::DebugResult& result)
// {
//   j["smoother_result"] = result.result;
//   j["smoother_stats"] = result.smoother_stats;
//   j["object_id"] = result.object_id;
//   j["frame_id"] = result.frame_id;
//   j["timestamp"] = result.timestamp;
//   j["average_feature_age"] = result.average_feature_age;
//   j["num_tracks"] = result.num_tracks;
//   j["frame_id_KF"] = result.frame_id_KF;
//   j["num_landmarks_in_smoother"] = result.num_landmarks_in_smoother;
//   j["num_motions_in_smoother"] = result.num_motions_in_smoother;
// }

}  // namespace dyno
