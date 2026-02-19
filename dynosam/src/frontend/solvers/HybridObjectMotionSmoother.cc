#include "dynosam/frontend/solvers/HybridObjectMotionSmoother.hpp"

#include <gtsam/linear/NoiseModel.h>

#include "dynosam/factors/HybridFormulationFactors.hpp"
#include "dynosam_common/utils/TimingStats.hpp"
#include "dynosam_opt/FactorGraphTools.hpp"
#include "dynosam_opt/Symbols.hpp"

namespace dyno {

gtsam::Symbol PointSymbol(TrackletId tracklet_id) {
  return gtsam::Symbol(kDynamicLandmarkSymbolChar, tracklet_id);
}

HybridObjectMotionSmoother::Ptr
HybridObjectMotionSmoother::CreateWithInitialMotion(
    const ObjectId object_id, double smoother_lag, const gtsam::Pose3& L_KF_km1,
    Frame::Ptr frame_km1, const TrackletIds& tracklets) {
  auto smoother = std::shared_ptr<HybridObjectMotionSmoother>(
      new HybridObjectMotionSmoother(object_id, frame_km1->getCamera(),
                                     smoother_lag));
  smoother->createNewKeyedMotion(L_KF_km1, frame_km1, tracklets);
  return smoother;
}

// TODO: really should initalise with frame and tracklet ids...
HybridObjectMotionSmoother::HybridObjectMotionSmoother(ObjectId object_id,
                                                       Camera::Ptr camera,
                                                       double smootherLag)
    : gtsam::FixedLagSmoother(smootherLag),
      object_id_(object_id),
      logger_prefix_("hybrid_motion_smoother_j" + std::to_string(object_id)),
      isam_(DefaultISAM2Params()) {
  rgbd_camera_ = CHECK_NOTNULL(camera)->safeGetRGBDCamera();
  CHECK(rgbd_camera_);
  stereo_calibration_ = rgbd_camera_->getFakeStereoCalib();
}

HybridObjectMotionSmoother::~HybridObjectMotionSmoother() {
  if (!debug_results_.empty()) {
    const std::string file_name = logger_prefix_ + "_debug.bson";
    LOG(INFO) << "Writing solver debug file: " << file_name;
    const std::string file_path = getOutputFilePath(file_name);
    JsonConverter::WriteOutJson(debug_results_, file_path,
                                JsonConverter::Format::BSON);
  }
}

PoseWithMotionTrajectory HybridObjectMotionSmoother::trajectory() const {
  // only from KF -> k (assume continuous?)
  PoseWithMotionTrajectory trajectory = trajectory_till_lKF_;
  trajectory.insert(localTrajectory());

  return trajectory;
}

PoseWithMotionTrajectory HybridObjectMotionSmoother::localTrajectory() const {
  CHECK_EQ(frames_since_lKF_.size(), timestamps_since_lKF_.size());

  if (frames_since_lKF_.empty()) {
    return PoseWithMotionTrajectory{};
  }

  auto kf_data = keyframe_range_.find(getFrameId());
  const auto [frame_KF_id, L_W_KF] = *kf_data;

  CHECK_EQ(frame_KF_id, getKeyFrameId());

  // build trajectory from best state estimate since last KF
  PoseWithMotionTrajectory local_trajectory;
  for (size_t i = 0; i < frames_since_lKF_.size(); i++) {
    const FrameId frame_id = frames_since_lKF_.at(i);
    const Timestamp timestamp = timestamps_since_lKF_.at(i);

    // sanity check that all frames are part of the same KF range
    CHECK(kf_data->contains(frame_id));

    const gtsam::Symbol H_key_k = ObjectMotionSymbol(object_id_, frame_id);
    CHECK(state_since_lKF_.exists(H_key_k));

    const gtsam::Pose3 H_W_KF_k = state_since_lKF_.at<gtsam::Pose3>(H_key_k);

    Motion3ReferenceFrame f2f_motion;
    if (i == 0) {
      CHECK_EQ(frame_id, frame_KF_id);
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

      const gtsam::Pose3 H_W_KF_km1 =
          state_since_lKF_.at<gtsam::Pose3>(H_key_km1);
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

Motion3ReferenceFrame HybridObjectMotionSmoother::getKeyFramedMotionReference()
    const {
  return Motion3ReferenceFrame(
      getKeyFramedMotion(), Motion3ReferenceFrame::Style::KF,
      ReferenceFrame::GLOBAL, getKeyFrameId(), getFrameId());
}

gtsam::Pose3 HybridObjectMotionSmoother::getKeyFramedMotion() const {
  const gtsam::Symbol sym(ObjectMotionSymbol(object_id_, getFrameId()));
  CHECK(isam_.valueExists(sym));
  const gtsam::Pose3 H_W_KF_k = calculateEstimate<gtsam::Pose3>(sym);
  return H_W_KF_k;
}

gtsam::Pose3 HybridObjectMotionSmoother::getF2FMotion() const {
  const gtsam::Pose3 H_W_KF_k = getKeyFramedMotion();
  if (getKeyFrameId() == getFrameId()) {
    return H_W_KF_k;
  }

  FrameId frame_id_km1 = getFrameId() - 1u;

  const gtsam::Symbol prev_motion_symbol(
      ObjectMotionSymbol(object_id_, frame_id_km1));

  CHECK(isam_.valueExists(prev_motion_symbol))
      << DynosamKeyFormatter(prev_motion_symbol);
  const gtsam::Pose3 H_W_KF_km1 =
      calculateEstimate<gtsam::Pose3>(prev_motion_symbol);

  return H_W_KF_k * H_W_KF_km1.inverse();
}

gtsam::FastMap<TrackletId, gtsam::Point3>
HybridObjectMotionSmoother::getCurrentLinearizedPoints() const {
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

gtsam::Pose3 HybridObjectMotionSmoother::getKeyFramePose() const {
  auto kf_data = keyframe_range_.find(getFrameId());
  CHECK(kf_data);

  const auto [_, LKF] = *kf_data;
  return LKF;
}

gtsam::Pose3 HybridObjectMotionSmoother::getPose() const {
  return getKeyFramedMotion() * getKeyFramePose();
}

HybridObjectMotionSmoother::Result
HybridObjectMotionSmoother::createNewKeyedMotion(const gtsam::Pose3& L_KF,
                                                 Frame::Ptr frame,
                                                 const TrackletIds& tracklets) {
  if (VLOG_IS_ON(10)) {
    const std::string current_frame =
        frames_since_lKF_.empty() ? "None" : std::to_string(getFrameId());
    const std::string current_KF =
        frames_since_lKF_.empty() ? "None" : std::to_string(getKeyFrameId());
    VLOG(10) << "Creating new KeyMotion "
             << info_string(frame->getFrameId(), object_id_)
             << " current k=" << current_frame << " KF=" << current_KF;
  }

  isam_ = gtsam::ISAM2(DefaultISAM2Params());

  // update fixed trajectory using current kf state
  // must do this before temporal/keyframe data-structures are reset
  // relies on state_since_lKF_ to fill trajectory values
  trajectory_till_lKF_ = localTrajectory();

  frames_since_lKF_.clear();
  timestamps_since_lKF_.clear();
  state_since_lKF_.clear();

  // clear internal timestamp mapping so as to not confuse the Fixed Lag
  timestampKeyMap_.clear();
  keyTimestampMap_.clear();

  keyframe_range_.startNewActiveRange(frame->getFrameId(), L_KF);

  // update and add points at initial frame corresponding with an identity
  // motion allow the update function to insert new frames and timestamps given
  // the frame
  return updateFromInitialMotion(gtsam::Pose3::Identity(), frame, tracklets);
}

HybridObjectMotionSmoother::Result HybridObjectMotionSmoother::update(
    const gtsam::Pose3& H_W_km1_k_predict, Frame::Ptr frame,
    const TrackletIds& tracklets) {
  CHECK(!frames_since_lKF_.empty())
      << "HybridObjectMotionSmoother::update "
      << " cannot be called without first creating a valid MotionFrame!";

  gtsam::Pose3 H_W_KF_km1 = gtsam::Pose3::Identity();
  // if we have at least one previous entry (otherwise getFrameId behaviour is
  // undefined which arguanle is poor design!)
  if (isam_.valueExists(ObjectMotionSymbol(object_id_, getFrameId()))) {
    H_W_KF_km1 = calculateEstimate<gtsam::Pose3>(
        ObjectMotionSymbol(object_id_, getFrameId()));
  }
  // propogate initial guess
  const gtsam::Pose3 H_W_KF_k = H_W_km1_k_predict * H_W_KF_km1;
  return updateFromInitialMotion(H_W_KF_k, frame, tracklets);
}

HybridObjectMotionSmoother::Result
HybridObjectMotionSmoother::updateFromInitialMotion(
    const gtsam::Pose3& H_W_KF_k_initial, Frame::Ptr frame,
    const TrackletIds& tracklets) {
  // using frame_id before update

  const FrameId frame_id = frame->getFrameId();
  const Timestamp timestamp = frame->getTimestamp();

  // update temporal data-structure immediately so getFrameId() and
  // getKeyFrameId() functions work
  frames_since_lKF_.push_back(frame_id);
  timestamps_since_lKF_.push_back(timestamp);

  const double frame_as_double = static_cast<double>(frame_id);

  const gtsam::Symbol H_key_k = ObjectMotionSymbol(object_id_, frame_id);
  // fixed camera pose
  const gtsam::Pose3 X_W_k = frame->getPose();
  // get current keyframe pose
  const gtsam::Pose3 L_KF = getKeyFramePose();

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

  for (const TrackletId& tracklet_id : tracklets) {
    const Feature::Ptr feature = frame->at(tracklet_id);
    CHECK(feature);

    const gtsam::Symbol m_key(PointSymbol(tracklet_id));

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

    const auto [stereo_keypoint_status, stereo_measurement] =
        rgbd_camera_->getStereo(feature);
    if (!stereo_keypoint_status) {
      if (is_new) {
        new_values.erase(m_key);
      }
      continue;
    }

    timestamps[m_key] = frame_as_double;

    auto factor = boost::make_shared<StereoHybridMotionFactor2>(
        stereo_measurement, L_KF, X_W_k, stereo_noise_model,
        stereo_calibration_, H_key_k, m_key, true /*throw ceirality*/
    );
    CHECK(factor);

    new_factors += factor;
  }

  if (getFrameId() == getKeyFrameId()) {
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
  debug_result.frame_id = getFrameId();
  debug_result.timestamp = getTimestamp();
  debug_result.frame_id_KF = getKeyFrameId();

  debug_result.num_landmarks_in_smoother =
      getObjectPointsFromSmootherState().size();
  debug_result.num_motions_in_smoother =
      getObjectMotionsFromSmootherState().size();

  debug_results_.push_back(std::move(debug_result));

  return result;
}

HybridObjectMotionSmoother::Result HybridObjectMotionSmoother::updateSmoother(
    const gtsam::NonlinearFactorGraph& newFactors,
    const gtsam::Values& newTheta, const KeyTimestampMap& timestamps,
    const gtsam::ISAM2UpdateParams& update_params) {
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
          gtsam::ISAM2>::gatherAdditionalKeysToReEliminate(isam_,
                                                           marginalizableKeys);

  gtsam::KeyList additionalMarkedKeys(additionalKeys.begin(),
                                      additionalKeys.end());
  result.additional_keys_reeliminate = additionalMarkedKeys;

  gtsam::ISAM2UpdateParams mutable_update_params = update_params;
  if (!mutable_update_params.extraReelimKeys) {
    mutable_update_params.extraReelimKeys = gtsam::KeyList{};
  }
  mutable_update_params.extraReelimKeys->insert(
      mutable_update_params.extraReelimKeys->begin(),
      additionalMarkedKeys.begin(), additionalMarkedKeys.end());

  mutable_update_params.constrainedKeys = constrainedKeys;

  utils::ChronoTimingStats update_timer(logger_prefix_ + ".isam_update", 10);
  isamResult_ = isam_.update(newFactors, newTheta, mutable_update_params);
  result.update_time_ms = update_timer.stop();
  result.isam_result = isamResult_;

  // Marginalize out any needed variables
  if (marginalizableKeys.size() > 0) {
    gtsam::FastList<gtsam::Key> leafKeys(marginalizableKeys.begin(),
                                         marginalizableKeys.end());
    utils::ChronoTimingStats marginalize_timer(
        logger_prefix_ + ".marginalize_leaves", 10);
    isam_.marginalizeLeaves(leafKeys);
    result.marginalize_time_ms = marginalize_timer.deltaMilliseconds();
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

void to_json(json& j, const HybridObjectMotionSmoother::Result& result) {
  j["marginalized_keys"] = result.marginalized_keys;
  j["additional_keys_reeliminate"] = result.additional_keys_reeliminate;
  j["update_time_ms"] = result.update_time_ms;
  j["marginalize_time_ms"] = result.marginalize_time_ms;
  j["isam_result"] = result.isam_result;
}

void to_json(json& j, const HybridObjectMotionSmoother::DebugResult& result) {
  j["smoother_result"] = result.result;
  j["smoother_stats"] = result.smoother_stats;
  j["object_id"] = result.object_id;
  j["frame_id"] = result.frame_id;
  j["timestamp"] = result.timestamp;
  j["frame_id_KF"] = result.frame_id_KF;
  j["num_landmarks_in_smoother"] = result.num_landmarks_in_smoother;
  j["num_motions_in_smoother"] = result.num_motions_in_smoother;
}

}  // namespace dyno
