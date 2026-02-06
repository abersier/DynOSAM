#include "dynosam/frontend/VIFrontend.hpp"

#include "dynosam/frontend/vision/MotionSolver.hpp"
#include "dynosam_cv/RGBDCamera.hpp"

namespace dyno {

Frontend::Frontend(const std::string& name, const DynoParams& params,
                   ImageDisplayQueue* display_queue)
    : Base(name), dyno_params_(params), display_queue_(display_queue) {
  // TODO: logger!!
}

bool Frontend::pushImageToDisplayQueue(const std::string& wname,
                                       const cv::Mat& image) {
  if (!display_queue_) {
    return false;
  }

  display_queue_->push(ImageToDisplay(wname, image));
  return true;
}

void Frontend::validateInput(
    const FrontendInputPacketBase::ConstPtr& input) const {
  const auto image_container = input->image_container_;

  if (!image_container) {
    throw DynosamException("Image container is null!");
  }

  const bool has_rgb = image_container->hasRgb();
  const bool has_depth_image = image_container->hasDepth();
  const bool has_stereo = image_container->hasRightRgb();

  if (!has_rgb) {
    throw InvalidImageContainerException1(*image_container, "Missing RGB");
  }

  if (!has_depth_image && !has_stereo) {
    throw InvalidImageContainerException1(*image_container,
                                          "Missing Depth or Stereo");
  }
}

VIFrontend::VIFrontend(const std::string& name, const DynoParams& params,
                       Camera::Ptr camera, ImageDisplayQueue* display_queue)
    : Frontend(name, params, display_queue),
      camera_(CHECK_NOTNULL(camera)),
      ego_motion_solver_(params.frontend_params_.ego_motion_solver_params,
                         camera->getParams()),
      imu_frontend_(params.frontend_params_.imu_params) {
  const auto& frontend_params = dyno_params_.frontend_params_;
  tracker_ =
      std::make_unique<FeatureTracker>(frontend_params, camera_, display_queue);
}

Frame::Ptr VIFrontend::featureTrack(
    const FrontendInputPacketBase::ConstPtr input,
    std::optional<gtsam::Rot3> R_km1_k) {
  ImageContainer::Ptr image_container = input->image_container_;
  Frame::Ptr frame = tracker_->track(input->getFrameId(), input->getTimestamp(),
                                     *image_container, R_km1_k);
  CHECK(frame->updateDepths());
  return frame;
}

std::optional<gtsam::NavState> VIFrontend::tryPropogateImu(
    const FrontendInputPacketBase::ConstPtr input,
    const gtsam::NavState& nav_state_lIMU, ImuFrontend::PimPtr& pim_out) {
  if (!input->imu_measurements.has_value()) {
    return {};
  }

  auto imu_measurements = input->imu_measurements.value();

  pim_out = imu_frontend_.preintegrateImuMeasurements(imu_measurements);
  return pim_out->predict(nav_state_lIMU, gtsam::imuBias::ConstantBias{});
}

bool VIFrontend::tryStereoMatch(Frame::Ptr frame,
                                ImageContainer::Ptr image_container,
                                FeaturePtrs& stereo_features_out,
                                FeatureContainer& left_features_in_out) {
  if (!image_container->hasRightRgb()) {
    return false;
  }

  std::shared_ptr<RGBDCamera> rgbd_camera = camera_->safeGetRGBDCamera();
  CHECK(rgbd_camera) << "Stereo imagery provided at k= " << frame->getFrameId()
                     << " but rgbd camera is null!";

  const cv::Mat& left_rgb = image_container->rgb();
  const cv::Mat& right_rgb = image_container->rightRgb();

  return tracker_->stereoTrack(stereo_features_out, left_features_in_out,
                               left_rgb, right_rgb, rgbd_camera->baseline());
}

bool VIFrontend::tryStereoMatchStaticFeatures(
    Frame::Ptr frame, ImageContainer::Ptr image_container,
    FeaturePtrs& stereo_features_out) {
  return tryStereoMatch(frame, image_container, stereo_features_out,
                        frame->static_features_);
}

bool VIFrontend::solveAndRefineEgoMotion(
    Frame::Ptr frame_k, const Frame::Ptr& frame_km1,
    const gtsam::NavState& nav_state_km1, const gtsam::Pose3& T_km1_k,
    std::optional<gtsam::NavState> propogated_nav_state_k,
    std::optional<gtsam::Rot3> R_km1_k) {
  utils::ChronoTimingStats timer(this->moduleName() + ".camera_motion");
  const auto& frontend_params = dyno_params_.frontend_params_;

  if (!frontend_params.use_ego_motion_pnp) {
    LOG(WARNING) << "Frontend param use_ego_motion_pnp set to false but only "
                    "PnP implemented";
  }

  // solve PnP
  Pose3SolverResult pnp_result =
      ego_motion_solver_.geometricOutlierRejection3d2d(frame_km1, frame_k,
                                                       R_km1_k);

  // sanity check
  const TrackletIds tracklets = frame_k->static_features_.collectTracklets();
  // tracklets shoudl be more (or same as) correspondances as there will be new
  // points untracked
  CHECK_GE(tracklets.size(),
           pnp_result.inliers.size() + pnp_result.outliers.size());
  frame_k->static_features_.markOutliers(pnp_result.outliers);

  if (pnp_result.status != TrackingStatus::VALID ||
      pnp_result.inliers.size() < 30) {
    // try propogate pose with available models
    if (propogated_nav_state_k) {
      frame_k->T_world_camera_ = propogated_nav_state_k->pose();
      VLOG(10) << "Number usable features invalid or too few at k= "
               << frame_k->getFrameId()
               << " - using IMU propogated pose to set camera pose!";
    } else {
      frame_k->T_world_camera_ = nav_state_km1.pose() * T_km1_k;
      VLOG(10) << "Number usable features invalid or too few at k= "
               << frame_k->getFrameId()
               << " - using constant velocity model to propogated camera pose!";
    }

    // TODO: should almost definitely do this in future, but right now we use
    // measurements to construct a framenode in the backend so if there are no
    // measurements we get a frame_node null.... for now... make hack and set
    // all ages of inliers to 1!!! since we need n measurements in the backend
    // this will ensure that they dont get added to the
    //  optimisation problem but will get added to the map...
    // for (const auto& inlier : result.inliers) {
    // frame_k->static_features_.getByTrackletId(inlier)->age(1u);
    // }
    return false;
  } else {
    // update camera pose
    frame_k->T_world_camera_ = pnp_result.best_result;

    if (frontend_params.refine_camera_pose_with_joint_of) {
      VLOG(10) << "Refining camera pose with joint optical-flow";

      utils::ChronoTimingStats timer(this->moduleName() +
                                     ".camera_motion.refine");

      const auto& joint_optical_flow_opt_params =
          frontend_params.object_motion_solver_params.joint_of_params;
      OpticalFlowAndPoseOptimizer flow_optimizer(joint_optical_flow_opt_params);

      const auto joint_optical_flow_result =
          flow_optimizer.optimizeAndUpdate<CalibrationType>(
              frame_km1, frame_k, pnp_result.inliers, pnp_result.best_result);

      frame_k->T_world_camera_ =
          joint_optical_flow_result.best_result.refined_pose;

      VLOG(15) << "Refined camera pose with optical flow - error before: "
               << joint_optical_flow_result.error_before.value_or(NaN)
               << " error_after: "
               << joint_optical_flow_result.error_after.value_or(NaN);
    }
    return true;
  }
}

void VIFrontend::fillDebugImagery(DebugImagery& debug_imagery,
                                  const Frame::Ptr& frame_k,
                                  const Frame::Ptr& frame_km1) const {
  debug_imagery.tracking_image = tracker_->computeImageTracks(
      *frame_km1, *frame_k,
      dyno_params_.frontend_params_.image_tracks_vis_params);

  const ImageContainer& processed_ic = frame_k->image_container_;

  if (processed_ic.hasRgb()) {
    debug_imagery.rgb_viz = ImageType::RGBMono::toRGB(processed_ic.rgb());
  }

  if (processed_ic.hasDepth()) {
    debug_imagery.depth_viz = ImageType::Depth::toRGB(processed_ic.depth());
  }

  if (processed_ic.hasOpticalFlow()) {
    debug_imagery.flow_viz =
        ImageType::OpticalFlow::toRGB(processed_ic.opticalFlow());
  }

  if (processed_ic.hasObjectMask()) {
    debug_imagery.mask_viz =
        ImageType::MotionMask::toRGB(processed_ic.objectMotionMask());
  }

  // const auto& camera_params = camera_->getParams();
  // const auto& K = camera_params.getCameraMatrix();
  // const auto& D = camera_params.getDistortionCoeffs();

  // const gtsam::Pose3& X_k = frame_k->getPose();

  // // poses are expected to be in the world frame
  // gtsam::FastMap<ObjectId, gtsam::Pose3> poses_k_map =
  //     object_poses.collectByFrame(frame_k->getFrameId());
  // std::vector<gtsam::Pose3> poses_k_vec;
  // std::transform(poses_k_map.begin(), poses_k_map.end(),
  //                 std::back_inserter(poses_k_vec),
  //                 [&X_k](const std::pair<ObjectId, gtsam::Pose3>& pair) {
  //                 // put object pose into the camera frame so it can be
  //                 // projected into the image
  //                 return X_k.inverse() * pair.second;
  //                 });

  // TODO: bring back when visualisation is unified with incremental solver!!
  //  utils::drawObjectPoseAxes(tracking_image, K, D, poses_k_vec);
  // return tracking_image;
}

RegularVIFrontend::RegularVIFrontend(const DynoParams& params,
                                     Camera::Ptr camera,
                                     ImageDisplayQueue* display_queue)
    : VIFrontend("regular-frontend", params, camera, display_queue) {
  auto object_motion_solver_params =
      dyno_params_.frontend_params_.object_motion_solver_params;
  // add ground truth hook
  // object_motion_solver_params.ground_truth_packets_request = [&]() {
  //   return this->shared_module_info.getGroundTruthPackets();
  // };
  object_motion_solver_params.refine_motion_with_3d = false;
  object_motion_solver_ = std::make_unique<ConsecutiveFrameObjectMotionSolver>(
      object_motion_solver_params, camera_->getParams());
}

RegularVIFrontend::SpinReturn RegularVIFrontend::boostrapSpin(
    FrontendInputPacketBase::ConstPtr input) {
  Frame::Ptr frame_k = featureTrack(input);
  const auto frame_id_k = input->getFrameId();
  const auto timestamp_k = input->getTimestamp();

  VisionImuPacket::Ptr vision_imu_packet = std::make_shared<VisionImuPacket>();
  vision_imu_packet->frameId(frame_id_k);
  vision_imu_packet->timestamp(timestamp_k);
  vision_imu_packet->groundTruthPacket(input->optional_gt_);
  // TODO: fill measurements
  // TODO: send to backend

  dyno_state_.camera_trajectory.insert(frame_id_k, timestamp_k,
                                       gtsam::Pose3::Identity());

  RealtimeOutput::Ptr realtime_output = std::make_shared<RealtimeOutput>();
  realtime_output->state.frame_id = frame_id_k;
  realtime_output->state.timestamp = timestamp_k;
  realtime_output->state.camera_trajectory = dyno_state_.camera_trajectory;

  return {State::Nominal, realtime_output};
}

RegularVIFrontend::SpinReturn RegularVIFrontend::nominalSpin(
    FrontendInputPacketBase::ConstPtr input) {
  ImageContainer::Ptr image_container = input->image_container_;
  const auto frame_id_k = input->getFrameId();
  const auto timestamp_k = input->getTimestamp();

  ImuFrontend::PimPtr pim = nullptr;
  std::optional<gtsam::NavState> imu_propogated_nav_state_k =
      tryPropogateImu(input, nav_state_km1_, pim);

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
  nav_state_km1_ = nav_state_k;

  dyno_state_.camera_trajectory.insert(frame_id_k, timestamp_k,
                                       nav_state_k.pose());

  dyno_state_.object_trajectories =
      object_motion_solver_->solve(frame_k, frame_km1);

  // construct output packet for backend
  VisionImuPacket::Ptr vision_imu_packet = std::make_shared<VisionImuPacket>();
  vision_imu_packet->frameId(frame_id_k);
  vision_imu_packet->timestamp(timestamp_k);
  vision_imu_packet->pim(pim);
  vision_imu_packet->groundTruthPacket(input->optional_gt_);

  fillOutputPacketWithTracks(vision_imu_packet, *frame_k, nav_state_k.pose(),
                             T_km1_k_, dyno_state_.object_trajectories);

  // we solve every frame so reset the preintegraion
  // this (of course) assumes we get IMU data between each frame
  if (imu_propogated_nav_state_k) {
    imu_frontend_.resetIntegration();
  }

  RealtimeOutput::Ptr realtime_output = std::make_shared<RealtimeOutput>();
  realtime_output->state.frame_id = frame_id_k;
  realtime_output->state.timestamp = timestamp_k;
  realtime_output->state.camera_trajectory = dyno_state_.camera_trajectory;
  realtime_output->state.object_trajectories = dyno_state_.object_trajectories;
  realtime_output->state.local_static_map =
      vision_imu_packet->staticLandmarkMeasurements();
  realtime_output->state.dynamic_map =
      vision_imu_packet->dynamicLandmarkMeasurements();

  fillDebugImagery(realtime_output->debug_imagery, frame_k, frame_km1);

  pushImageToDisplayQueue("Tracks",
                          realtime_output->debug_imagery.tracking_image);

  // if (FLAGS_set_dense_labelled_cloud) {
  //     VLOG(30) << "Setting dense labelled cloud";
  //     utils::ChronoTimingStats labelled_clout_timer(
  //         this->moduleName() + ".dense_labelled_cloud");
  //     const cv::Mat& board_detection_mask =
  //     tracker_->getBoarderDetectionMask();
  //     realtime_output->dense_labelled_cloud =
  //         frame_k->projectToDenseCloud(&board_detection_mask);

  //     //TODO: remove dense labelled cloud from VIOutput!!
  // }

  // TODO: log!

  return {State::Nominal, realtime_output};
}

void RegularVIFrontend::fillOutputPacketWithTracks(
    VisionImuPacket::Ptr vision_imu_packet, const Frame& frame,
    const gtsam::Pose3 X_W_k, const gtsam::Pose3& T_k_1_k,
    const MultiObjectTrajectories& object_trajectories) const {
  CHECK(vision_imu_packet);
  const auto frame_id = frame.getFrameId();
  // construct image tracks
  const double& static_pixel_sigma =
      dyno_params_.backend_params_.static_pixel_noise_sigma;
  const double& static_point_sigma =
      dyno_params_.backend_params_.static_point_noise_sigma;

  const double& dynamic_pixel_sigma =
      dyno_params_.backend_params_.dynamic_pixel_noise_sigma;
  const double& dynamic_point_sigma =
      dyno_params_.backend_params_.dynamic_point_noise_sigma;

  gtsam::Vector2 static_pixel_sigmas;
  static_pixel_sigmas << static_pixel_sigma, static_pixel_sigma;

  gtsam::Vector2 dynamic_pixel_sigmas;
  dynamic_pixel_sigmas << dynamic_pixel_sigma, dynamic_pixel_sigma;

  auto& camera = *this->camera_;
  auto fill_camera_measurements =
      [&camera](FeatureFilterIterator it,
                CameraMeasurementStatusVector* measurements, FrameId frame_id,
                const gtsam::Vector2& pixel_sigmas, double depth_sigma) {
        std::shared_ptr<RGBDCamera> rgbd_camera = camera.safeGetRGBDCamera();

        for (const Feature::Ptr& f : it) {
          const TrackletId tracklet_id = f->trackletId();
          const Keypoint& kp = f->keypoint();
          const ObjectId object_id = f->objectId();
          CHECK_EQ(f->objectId(), object_id);
          CHECK(Feature::IsUsable(f));

          MeasurementWithCovariance<Keypoint> kp_measurement =
              MeasurementWithCovariance<Keypoint>::FromSigmas(kp, pixel_sigmas);
          CameraMeasurement camera_measurement(kp_measurement);

          // This can come from either stereo or rgbd
          if (f->hasDepth()) {
            // MeasurementWithCovariance<Landmark> landmark_measurement(
            //     // assume sigma_u and sigma_v are identical
            //     vision_tools::backProjectAndCovariance(
            //         *f, camera, pixel_sigmas(0), depth_sigma));
            // camera_measurement.landmark(landmark_measurement);
            Landmark landmark;
            camera.backProject(kp, f->depth(), &landmark);

            gtsam::Vector3 sigmas;
            sigmas << depth_sigma, depth_sigma, depth_sigma;

            MeasurementWithCovariance<Landmark> landmark_measurement =
                MeasurementWithCovariance<Landmark>::FromSigmas(landmark,
                                                                sigmas);
            camera_measurement.landmark(landmark_measurement);
          }

          if (f->hasRightKeypoint()) {
            CHECK(f->hasDepth())
                << "Right keypoint set for feature but no depth!";
            MeasurementWithCovariance<Keypoint> right_kp_measurement =
                MeasurementWithCovariance<Keypoint>::FromSigmas(
                    f->rightKeypoint(), pixel_sigmas);
            camera_measurement.rightKeypoint(right_kp_measurement);
          }
          // no right keypoint and has rgbd camera and has depth, project
          // keypoint into right camera
          else if (rgbd_camera && f->hasDepth()) {
            bool right_projection_result = rgbd_camera->projectRight(f);
            if (!right_projection_result) {
              // TODO: for now mark as outlier and ignore point
              f->markOutlier();
              continue;
            }

            CHECK(f->hasRightKeypoint());
            MeasurementWithCovariance<Keypoint> right_kp_measurement =
                MeasurementWithCovariance<Keypoint>::FromSigmas(
                    f->rightKeypoint(), pixel_sigmas);
            camera_measurement.rightKeypoint(right_kp_measurement);
          }

          if (f->keypointType() == KeyPointType::STATIC) {
            CHECK_EQ(object_id, background_label);
          } else {
            CHECK_NE(object_id, background_label);
          }

          measurements->push_back(
              CameraMeasurementStatus(camera_measurement, frame_id, tracklet_id,
                                      object_id, ReferenceFrame::LOCAL));
        }
      };
  VisionImuPacket::CameraTracks camera_tracks;
  auto* static_measurements = &camera_tracks.measurements;
  fill_camera_measurements(frame.usableStaticFeaturesBegin(),
                           static_measurements, frame_id, static_pixel_sigmas,
                           static_point_sigma);
  camera_tracks.X_W_k = X_W_k;
  camera_tracks.T_k_1_k = T_k_1_k;
  vision_imu_packet->cameraTracks(camera_tracks);

  // First collect all dynamic measurements then split them by object
  // This is a bit silly
  CameraMeasurementStatusVector dynamic_measurements;
  fill_camera_measurements(frame.usableDynamicFeaturesBegin(),
                           &dynamic_measurements, frame_id,
                           dynamic_pixel_sigmas, dynamic_point_sigma);

  VisionImuPacket::ObjectTrackMap object_tracks;

  const auto object_estimates_k = object_trajectories.entriesAtFrame(frame_id);
  for (const auto& [object_id, object_estimate] : object_estimates_k) {
    const auto& L_W_k = object_estimate.data.pose;
    const auto& H_W_km1_k = object_estimate.data.motion;

    CHECK_EQ(H_W_km1_k.from(), frame_id - 1u);
    CHECK_EQ(H_W_km1_k.to(), frame_id);

    VisionImuPacket::ObjectTracks object_track;
    object_track.H_W_k_1_k = H_W_km1_k;
    object_track.L_W_k = L_W_k;
    object_tracks.insert2(object_id, object_track);
  }

  for (const auto& dm : dynamic_measurements) {
    const auto& object_id = dm.objectId();
    // throw out features detected on objects where the tracking failed
    if (object_tracks.exists(object_id)) {
      VisionImuPacket::ObjectTracks& object_track = object_tracks.at(object_id);
      object_track.measurements.push_back(dm);
    }
  }
  vision_imu_packet->objectTracks(object_tracks);
}

PoseChangeVIFrontend::PoseChangeVIFrontend(const DynoParams& params,
                                           Camera::Ptr camera,
                                           ImageDisplayQueue* display_queue)
    : VIFrontend("pc-frontend", params, camera, display_queue) {}

PoseChangeVIFrontend::SpinReturn PoseChangeVIFrontend::boostrapSpin(
    FrontendInputPacketBase::ConstPtr input) {}

PoseChangeVIFrontend::SpinReturn PoseChangeVIFrontend::nominalSpin(
    FrontendInputPacketBase::ConstPtr input) {}

}  // namespace dyno
