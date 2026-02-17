#include "dynosam/frontend/solvers/HybridObjectMotionSmoother.hpp"

#include <gtsam/linear/NoiseModel.h>

#include "dynosam/factors/HybridFormulationFactors.hpp"
#include "dynosam_opt/FactorGraphTools.hpp"
#include "dynosam_opt/Symbols.hpp"

namespace dyno {

gtsam::Symbol PointSymbol(TrackletId tracklet_id) {
  return gtsam::Symbol(kDynamicLandmarkSymbolChar, tracklet_id);
}

// TODO: really should initalise with frame and tracklet ids...
HybridObjectMotionSmoother::HybridObjectMotionSmoother(
    ObjectId object_id, const gtsam::Pose3& L_KF, FrameId frame_id_k,
    Timestamp timestamp_k, Camera::Ptr camera, double smootherLag)
    : gtsam::FixedLagSmoother(smootherLag),
      object_id_(object_id),
      L_KF_(L_KF),
      frames_since_lKF_({frame_id_k}),
      timestamps_since_lKF_({timestamp_k}),
      isam_(DefaultISAM2Params()) {
  rgbd_camera_ = CHECK_NOTNULL(camera)->safeGetRGBDCamera();
  CHECK(rgbd_camera_);
  stereo_calibration_ = rgbd_camera_->getFakeStereoCalib();
}

PoseWithMotionTrajectory HybridObjectMotionSmoother::trajectory() const {
  // only from KF -> k (assume continuous?)
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
  const gtsam::Values opt_est = isam_.calculateEstimate();

  std::map<gtsam::Key, gtsam::Point3> keyed_object_point_map =
      opt_est.extract<gtsam::Point3>(
          Symbol::ChrTest(kDynamicLandmarkSymbolChar));

  gtsam::FastMap<TrackletId, gtsam::Point3> object_point_map;
  for (const auto& [key, point] : keyed_object_point_map) {
    gtsam::Symbol sym(key);
    TrackletId tracklet_id = (TrackletId)sym.index();
    object_point_map.insert2(tracklet_id, point);
  }
  return object_point_map;
}

gtsam::Pose3 HybridObjectMotionSmoother::getPose() const {
  return getKeyFramedMotion() * getKeyFramePose();
}

void HybridObjectMotionSmoother::createNewKeyedMotion(const gtsam::Pose3& L_KF,
                                                      FrameId frame_id_k,
                                                      Timestamp timestamp_k) {
  isam_ = gtsam::ISAM2(DefaultISAM2Params());
  L_KF_ = L_KF;

  frames_since_lKF_ = FrameIds({frame_id_k});
  timestamps_since_lKF_ = Timestamps({timestamp_k});
}

void HybridObjectMotionSmoother::update(const gtsam::Pose3& H_W_km1_k_predict,
                                        Frame::Ptr frame,
                                        const TrackletIds& tracklets) {
  // using frame_id before update
  gtsam::Pose3 H_W_KF_km1 = gtsam::Pose3::Identity();
  if (isam_.valueExists(ObjectMotionSymbol(object_id_, getFrameId()))) {
    H_W_KF_km1 = calculateEstimate<gtsam::Pose3>(
        ObjectMotionSymbol(object_id_, getFrameId()));
  }

  const FrameId frame_id = frame->getFrameId();
  const Timestamp timestamp = frame->getTimestamp();

  frame_id_ = frame->getFrameId();
  timestamp_ = frame->getTimestamp();

  const double frame_as_double = static_cast<double>(frame_id_);

  const gtsam::Symbol H_key_k = ObjectMotionSymbol(object_id_, frame_id_);

  // propogate initial guess
  const gtsam::Pose3 H_W_KF_k = H_W_km1_k_predict * H_W_KF_km1;
  // fixed camera pose
  const gtsam::Pose3 X_W_k = frame->getPose();

  gtsam::Values new_values;
  gtsam::NonlinearFactorGraph new_factors;

  KeyTimestampMap timestamps;
  // add motions
  timestamps[H_key_k] = frame_as_double;
  new_values.insert(H_key_k, H_W_KF_k);

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
    // if variable is removed this is re-initalizing it!!! Is this what we want
    // to do!!?
    if (!isam_.valueExists(m_key)) {
      const gtsam::Point3 m_X_k = frame->backProjectToCamera(tracklet_id);
      Landmark m_L_init =
          HybridObjectMotion::projectToObject3(X_W_k, H_W_KF_k, L_KF_, m_X_k);

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
        stereo_measurement, L_KF_, X_W_k, stereo_noise_model,
        stereo_calibration_, H_key_k, m_key, true /*throw ceirality*/
    );
    CHECK(factor);

    new_factors += factor;
  }

  if (frame_id_ == frame_id_KF_) {
    gtsam::SharedNoiseModel identity_motion_model =
        gtsam::noiseModel::Isotropic::Sigma(6u, 0.01);

    // TODO: add prior on this first motion to make it identity!
    new_factors.addPrior<gtsam::Pose3>(H_key_k, gtsam::Pose3::Identity(),
                                       identity_motion_model);
  }

  if (frame_id_ > 2) {
    const gtsam::Symbol H_key_km1 =
        ObjectMotionSymbol(object_id_, frame_id_ - 1u);
    const gtsam::Symbol H_key_km2 =
        ObjectMotionSymbol(object_id_, frame_id_ - 2u);

    // TODO: params
    gtsam::SharedNoiseModel smoothing_motion_model =
        gtsam::noiseModel::Isotropic::Sigma(6u, 0.1);

    // TODO: ALL motions should use the same L_KF_
    //  if L_KF_ is only updated when we reset internal ISAM then no problem!
    if (isam_.valueExists(H_key_km1) && isam_.valueExists(H_key_km2)) {
      new_factors.emplace_shared<HybridSmoothingFactor>(
          H_key_km2, H_key_km1, H_key_k, L_KF_, smoothing_motion_model);
    }
  }

  this->update(new_factors, new_values, timestamps, ISAM2UpdateParams{});
}

void HybridObjectMotionSmoother::update(
    const gtsam::NonlinearFactorGraph& newFactors,
    const gtsam::Values& newTheta, const KeyTimestampMap& timestamps,
    const gtsam::ISAM2UpdateParams& update_params) {
  gtsam::FastVector<size_t> removedFactors;
  boost::optional<gtsam::FastMap<gtsam::Key, int> > constrainedKeys = {};

  // Update the Timestamps associated with the factor keys
  updateKeyTimestampMap(timestamps);

  // Get current timestamp
  double current_timestamp = getCurrentTimestamp();
  LOG(INFO) << "Current timestamp: " << current_timestamp;

  // Find the set of variables to be marginalized out
  LOG(INFO) << "Findig keys before " << current_timestamp - smootherLag_;
  gtsam::KeyVector marginalizableKeys =
      findKeysBefore(current_timestamp - smootherLag_);

  // Force iSAM2 to put the marginalizable variables at the beginning
  createOrderingConstraints(marginalizableKeys, constrainedKeys);

  // std::cout << "Gets to marginalize due to filter: ";
  // for (const auto& key : marginalizableKeys) {
  //     std::cout << DynosamKeyFormatter(key) << " ";
  // }
  // std::cout << std::endl;

  std::unordered_set<gtsam::Key> additionalKeys =
      BayesTreeMarginalizationHelper<
          gtsam::ISAM2>::gatherAdditionalKeysToReEliminate(isam_,
                                                           marginalizableKeys);

  gtsam::KeyList additionalMarkedKeys(additionalKeys.begin(),
                                      additionalKeys.end());

  gtsam::ISAM2UpdateParams mutable_update_params = update_params;
  if (!mutable_update_params.extraReelimKeys) {
    mutable_update_params.extraReelimKeys = gtsam::KeyList{};
  }
  mutable_update_params.extraReelimKeys->insert(
      mutable_update_params.extraReelimKeys->begin(),
      additionalMarkedKeys.begin(), additionalMarkedKeys.end());

  mutable_update_params.constrainedKeys = constrainedKeys;

  isamResult_ = isam_.update(newFactors, newTheta, mutable_update_params);

  // Marginalize out any needed variables
  if (marginalizableKeys.size() > 0) {
    gtsam::FastList<gtsam::Key> leafKeys(marginalizableKeys.begin(),
                                         marginalizableKeys.end());
    isam_.marginalizeLeaves(leafKeys);
  }
  // Remove marginalized keys from the KeyTimestampMap
  eraseKeyTimestampMap(marginalizableKeys);
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

}  // namespace dyno
