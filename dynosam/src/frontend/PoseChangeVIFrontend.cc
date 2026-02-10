#include "dynosam/frontend/PoseChangeVIFrontend.hpp"

namespace dyno {

PoseChangeVIFrontend::PoseChangeVIFrontend(
    const DynoParams& params, Camera::Ptr camera,
    HybridFormulationKeyFrame::Ptr formulation,
    ImageDisplayQueue* display_queue,
    const SharedGroundTruth& shared_ground_truth)
    : VIFrontend("pc-frontend", params, camera, display_queue,
                 shared_ground_truth),
      formulation_(CHECK_NOTNULL(formulation)),
      map_(CHECK_NOTNULL(formulation->map())) {
  object_motion_solver_ = std::make_unique<ObjectMotionSolverFilter>(
      ObjectMotionSolverFilter::Params{}, camera->getParams());
}

PoseChangeVIFrontend::SpinReturn PoseChangeVIFrontend::boostrapSpin(
    FrontendInputPacketBase::ConstPtr input) {
  Frame::Ptr frame_k = featureTrack(input);
  const auto frame_id_k = input->getFrameId();
  const auto timestamp_k = input->getTimestamp();

  dyno_state_.camera_trajectory.insert(frame_id_k, timestamp_k,
                                       gtsam::Pose3::Identity());

  RealtimeOutput::Ptr realtime_output = std::make_shared<RealtimeOutput>();
  realtime_output->state.frame_id = frame_id_k;
  realtime_output->state.timestamp = timestamp_k;
  realtime_output->state.camera_trajectory = dyno_state_.camera_trajectory;

  logRealTimeOutput(realtime_output);

  if (pose_change_backend_sink_) {
    pose_change_backend_sink_(std::make_shared<PoseChangeInput>());
  }

  // add initial state/propogate
  // add measurements to map
  // add motion PC info to formulation (ie. what was pre-update)
  // build factors

  return {State::Nominal, realtime_output};
}

PoseChangeVIFrontend::SpinReturn PoseChangeVIFrontend::nominalSpin(
    FrontendInputPacketBase::ConstPtr input) {
  ImageContainer::Ptr image_container = input->image_container_;
  const auto frame_id_k = input->getFrameId();
  const auto timestamp_k = input->getTimestamp();

  ImuFrontend::PimPtr pim = nullptr;
  std::optional<gtsam::NavState> imu_propogated_nav_state_k =
      tryPropogateImu(input, nav_state_lkf_, pim);

  //! Rotation from k-1 to k in k-1
  std::optional<gtsam::Rot3> R_km1_k;
  if (imu_propogated_nav_state_k) {
    CHECK(pim);
    R_km1_k = nav_state_km1_.attitude().inverse() *
              imu_propogated_nav_state_k->attitude();
  }

  Frame::Ptr frame_k = featureTrack(input, R_km1_k);
  Frame::Ptr frame_km1 = tracker_->getPreviousFrame();
  CHECK(frame_km1);

  VLOG(5) << to_string(tracker_->getTrackerInfo());

  FeaturePtrs stereo_matches_1;
  bool stereo_matching_result =
      tryStereoMatchStaticFeatures(frame_k, image_container, stereo_matches_1);

  // when providing the propogated imu state only provide if it was
  // actually filled by a prediction from the IMU - otherwise it will ne
  // nullopt. This tells the function to use a constant motion model from the
  // previous frame ie. T_km1_k_ if tracking fails
  const bool ego_motion_solve =
      solveAndRefineEgoMotion(frame_k, frame_km1, nav_state_km1_, T_km1_k_,
                              imu_propogated_nav_state_k, R_km1_k);

  if (stereo_matching_result) {
    // Need to match aagain after optical flow used to update the keypoints
    // This seems to make a pretty big difference!!
    FeaturePtrs stereo_matches_2;
    tryStereoMatchStaticFeatures(frame_k, image_container, stereo_matches_2);
  }

  // we currently use the frame pose as the nav state - this value can come from
  // either the VO OR the IMU, depending on the result from the
  // solveCameraMotion this is only relevant since we dont solve incremental so
  // the backend is not immediately updating the frontend at which point we can
  // just use the best estimate in the case of the VO, the nav_state velocity
  const gtsam::NavState nav_state_k(frame_k->getPose(),
                                    (imu_propogated_nav_state_k)
                                        ? imu_propogated_nav_state_k->velocity()
                                        : gtsam::Vector3(0, 0, 0));

  T_km1_k_ = nav_state_km1_.pose().inverse() * nav_state_k.pose();
  T_lkf_k_ = nav_state_lkf_.pose().inverse() * nav_state_k.pose();
  nav_state_km1_ = nav_state_k;

  dyno_state_.camera_trajectory.insert(frame_id_k, timestamp_k,
                                       nav_state_k.pose());

  ObjectPoseChangeInfoMap pose_change_infos;
  solveObjectMotions(dyno_state_.object_trajectories, pose_change_infos,
                     frame_k, frame_km1);

  RealtimeOutput::Ptr realtime_output = std::make_shared<RealtimeOutput>();
  realtime_output->state.frame_id = frame_id_k;
  realtime_output->state.timestamp = timestamp_k;
  realtime_output->state.camera_trajectory = dyno_state_.camera_trajectory;
  realtime_output->state.object_trajectories = dyno_state_.object_trajectories;

  CameraMeasurementStatusVector static_measurements;
  fillMeasurementsFromFeatureIterator(
      &static_measurements, frame_k->usableStaticFeaturesBegin(), frame_id_k,
      timestamp_k, static_pixel_sigmas_, static_point_sigma_,
      &realtime_output->state.local_static_map);

  CameraMeasurementStatusVector dynamic_measurements;
  fillMeasurementsFromFeatureIterator(
      &dynamic_measurements, frame_k->usableDynamicFeaturesBegin(), frame_id_k,
      timestamp_k, dynamic_pixel_sigmas_, dynamic_point_sigma_,
      &realtime_output->state.dynamic_map);

  // add initial state/propogate
  // add measurements to map
  // add motion PC info to formulation (ie. what was pre-update)
  // build factors

  fillDebugImagery(realtime_output->debug_imagery, frame_k, frame_km1);

  pushImageToDisplayQueue("Tracks",
                          realtime_output->debug_imagery.tracking_image);

  logRealTimeOutput(realtime_output);

  if (pose_change_backend_sink_) {
    pose_change_backend_sink_(std::make_shared<PoseChangeInput>());
  }

  return {State::Nominal, realtime_output};
}

void PoseChangeVIFrontend::solveObjectMotions(
    MultiObjectTrajectories& trajectories, ObjectPoseChangeInfoMap& infos,
    Frame::Ptr frame_k, Frame::Ptr frame_km1) {
  trajectories = object_motion_solver_->solve(frame_k, frame_km1);

  auto filters = object_motion_solver_->getFilters();

  auto frame_id = frame_k->getFrameId();

  const auto object_estimates_k = trajectories.entriesAtFrame(frame_id);
  for (const auto& [object_id, entry] : object_estimates_k) {
    CHECK(filters.exists(object_id));
    auto filter = filters.at(object_id);
    auto object_kf_status = object_motion_solver_->getKeyFrameStatus(object_id);
    auto object_motion_track_status =
        object_motion_solver_->getTrackingStatus(object_id);

    ObjectPoseChangeInfo info;
    info.frame_id = frame_id;
    info.H_W_KF_k = filter->getKeyFramedMotionReference();
    info.L_W_KF = filter->getKeyFramePose();
    info.L_W_k = filter->getPose();

    info.regular_keyframe = false;
    info.anchor_keyframe = false;

    info.motion_track_status = object_motion_track_status;

    if (info.motion_track_status == ObjectTrackingStatus::New ||
        info.motion_track_status == ObjectTrackingStatus::WellTracked) {
      if (object_kf_status == ObjectKeyFrameStatus::AnchorKeyFrame) {
        info.regular_keyframe = true;
        info.anchor_keyframe = true;
      } else if (object_kf_status == ObjectKeyFrameStatus::RegularKeyFrame) {
        info.regular_keyframe = true;
      }
    }

    const auto fixed_points = filter->getCurrentLinearizedPoints();
    for (const auto& [tracklet_id, m_L] : fixed_points) {
      info.initial_object_points.push_back(LandmarkStatus::Dynamic(
          // currently no covariance!
          Point3Measurement(m_L), LandmarkStatus::MeaninglessFrame, NaN,
          tracklet_id, object_id, ReferenceFrame::OBJECT));
    }

    LOG(INFO) << "Making hybrid info for j=" << object_id << " with "
              << "motion KF: " << info.H_W_KF_k.from()
              << " to: " << info.H_W_KF_k.to()
              << " track status: " << to_string(object_motion_track_status)
              << " with regular kf " << std::boolalpha << info.regular_keyframe
              << " anchor kf " << info.anchor_keyframe;

    infos.insert2(object_id, info);
  }
}

}  // namespace dyno
