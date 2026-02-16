#include "dynosam/frontend/VIFrontend.hpp"

#include "dynosam_cv/RGBDCamera.hpp"

namespace dyno {

DEFINE_bool(use_frontend_logger, false,
            "If true, the frontend logger will be used");

RGBDFrontendLogger::RGBDFrontendLogger()
    : EstimationModuleLogger("frontend"),
      tracking_length_hist_file_name_(
          getOutputFilePath("tracklet_length_hist.json")) {}

void RGBDFrontendLogger::logTrackingLengthHistogram(const Frame::Ptr frame) {
  gtsam::FastMap<ObjectId, Histogram> histograms =
      vision_tools::makeTrackletLengthHistorgram(frame);
  // collect histograms per object and then nest them per frame
  // must cast keys (object id, frame id) to string to get the json library to
  // properly construct nested maps
  json per_object_hist;
  for (const auto& [object_id, hist] : histograms) {
    per_object_hist[std::to_string(object_id)] = hist;
  }
  tracklet_length_json_[std::to_string(frame->getFrameId())] = per_object_hist;
}

RGBDFrontendLogger::~RGBDFrontendLogger() {
  JsonConverter::WriteOutJson(tracklet_length_json_,
                              tracking_length_hist_file_name_);
}

Frontend::Frontend(const std::string& name, const DynoParams& params,
                   ImageDisplayQueue* display_queue,
                   const SharedGroundTruth& shared_ground_truth)
    : Base(name),
      dyno_params_(params),
      display_queue_(display_queue),
      shared_ground_truth_(shared_ground_truth) {
  if (FLAGS_use_frontend_logger) {
    LOG(INFO) << "Using front-end logger!";
    logger_ = std::make_unique<RGBDFrontendLogger>();
  }
}

bool Frontend::pushImageToDisplayQueue(const std::string& wname,
                                       const cv::Mat& image) {
  if (!display_queue_) {
    return false;
  }

  display_queue_->push(ImageToDisplay(wname, image));
  return true;
}

void Frontend::logRealTimeOutput(const RealtimeOutput::Ptr& output) {
  if (logger_) {
    auto ground_truths = shared_ground_truth_.access();
    const auto& dyno_state = output->state;
    const auto frame_id = dyno_state.frame_id;

    // TODO: right now we only log per frame!! Does not account for the updated
    // trajectory!
    logger_->logCameraPose(frame_id, dyno_state.camera_trajectory,
                           ground_truths);

    logger_->logObjectTrajectory(frame_id, dyno_state.object_trajectories,
                                 ground_truths);

    // TODO: not logging map points?

    // TODO: historgram?
  }
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
    throw InvalidImageContainerException(*image_container, "Missing RGB");
  }

  if (!has_depth_image && !has_stereo) {
    throw InvalidImageContainerException(*image_container,
                                         "Missing Depth or Stereo");
  }
}

VIFrontend::VIFrontend(const std::string& name, const DynoParams& params,
                       Camera::Ptr camera, ImageDisplayQueue* display_queue,
                       const SharedGroundTruth& shared_ground_truth)
    : Frontend(name, params, display_queue, shared_ground_truth),
      camera_(CHECK_NOTNULL(camera)),
      pnp_ransac_(params.frontend_params_.ego_motion_pnp_ransac_params,
                  camera->getParams()),
      optical_flow_pose_solver_(OpticalFlowAndPoseSolverParams{}),
      imu_frontend_(params.frontend_params_.imu_params) {
  const auto& frontend_params = dyno_params_.frontend_params_;
  tracker_ =
      std::make_unique<FeatureTracker>(frontend_params, camera_, display_queue);

  rgbd_camera_ = camera_->safeGetRGBDCamera();
  CHECK_NOTNULL(rgbd_camera_);

  // measurement sigmas
  static_point_sigma_ = dyno_params_.backend_params_.static_point_noise_sigma;
  dynamic_point_sigma_ = dyno_params_.backend_params_.dynamic_point_noise_sigma;

  const double& static_pixel_sigma =
      dyno_params_.backend_params_.static_pixel_noise_sigma;
  static_pixel_sigmas_ << static_pixel_sigma, static_pixel_sigma;

  const double& dynamic_pixel_sigma =
      dyno_params_.backend_params_.dynamic_pixel_noise_sigma;
  dynamic_pixel_sigmas_ << dynamic_pixel_sigma, dynamic_pixel_sigma;
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

  AbsolutePoseCorrespondences correspondences;
  frame_k->getCorrespondences(correspondences, *frame_km1, KeyPointType::STATIC,
                              frame_k->landmarkWorldKeypointCorrespondance());

  // solve PnP
  Pose3SolverResult pnp_result =
      pnp_ransac_.solve3d2d(correspondences, R_km1_k);

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

      const auto refinement_result =
          optical_flow_pose_solver_.optimizeAndUpdate(
              frame_km1, frame_k, pnp_result.inliers, pnp_result.best_result);

      frame_k->T_world_camera_ = refinement_result.best_result.refined_pose;

      VLOG(15) << "Refined camera pose with optical flow - error before: "
               << refinement_result.error_before.value_or(NaN)
               << " error_after: "
               << refinement_result.error_after.value_or(NaN);
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

void VIFrontend::fillMeasurementsFromFeatureIterator(
    CameraMeasurementStatusVector* measurements, FeatureFilterIterator it,
    FrameId frame_id, Timestamp timestamp, const gtsam::Vector2& pixel_sigmas,
    double depth_sigma, StatusLandmarkVector* landmarks) const {
  CHECK(measurements);

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
      rgbd_camera_->backProject(kp, f->depth(), &landmark);

      gtsam::Vector3 sigmas;
      sigmas << depth_sigma, depth_sigma, depth_sigma;

      MeasurementWithCovariance<Landmark> landmark_measurement =
          MeasurementWithCovariance<Landmark>::FromSigmas(landmark, sigmas);
      camera_measurement.landmark(landmark_measurement);

      if (landmarks) {
        landmarks->push_back(LandmarkStatus(landmark_measurement, frame_id,
                                            timestamp, tracklet_id, object_id,
                                            ReferenceFrame::LOCAL));
      }
    }

    if (f->hasRightKeypoint()) {
      CHECK(f->hasDepth()) << "Right keypoint set for feature but no depth!";
      MeasurementWithCovariance<Keypoint> right_kp_measurement =
          MeasurementWithCovariance<Keypoint>::FromSigmas(f->rightKeypoint(),
                                                          pixel_sigmas);
      camera_measurement.rightKeypoint(right_kp_measurement);
    }
    // no right keypoint and has rgbd camera and has depth, project
    // keypoint into right camera
    else if (rgbd_camera_ && f->hasDepth()) {
      bool right_projection_result = rgbd_camera_->projectRight(f);
      if (!right_projection_result) {
        // TODO: for now mark as outlier and ignore point
        f->markOutlier();
        continue;
      }

      CHECK(f->hasRightKeypoint());
      MeasurementWithCovariance<Keypoint> right_kp_measurement =
          MeasurementWithCovariance<Keypoint>::FromSigmas(f->rightKeypoint(),
                                                          pixel_sigmas);
      camera_measurement.rightKeypoint(right_kp_measurement);
    }

    if (f->keypointType() == KeyPointType::STATIC) {
      CHECK_EQ(object_id, background_label);
    } else {
      CHECK_NE(object_id, background_label);
    }

    measurements->push_back(
        CameraMeasurementStatus(camera_measurement, frame_id, timestamp,
                                tracklet_id, object_id, ReferenceFrame::LOCAL));
  }
}

}  // namespace dyno
