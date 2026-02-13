#include "dynosam/frontend/solvers/HybridObjectMotionSolver.hpp"

namespace dyno {

HybridObjectMotionSolver::HybridObjectMotionSolver(
    const HybridObjectMotionSolverParams& params,
    const CameraParams& camera_params,
    const SharedGroundTruth& shared_ground_truth)
    : params_(params),
      pnp_ransac_solver_(params.pnp_ransac_params, camera_params),
      optical_flow_pose_solver_(params.optical_flow_solver_params),
      shared_ground_truth_(shared_ground_truth) {}

MultiObjectTrajectories HybridObjectMotionSolver::solve(Frame::Ptr frame_k,
                                                        Frame::Ptr frame_km1,
                                                        bool parallel_solve) {
  // Handle lost objects: objects in filters_ but not in current frame's
  // object_observations_
  // the objects in of object_observations_ should correspond with
  // the objects that had a successful motion estimation in the previous frame
  // and not just the set of objects that were observed.
  // The base ObjectMotionSolver::solve function should erase the observations
  // with with failed solves!
  std::set<ObjectId> current_objects;
  for (const auto& [obj_id, _] : frame_k->object_observations_) {
    current_objects.insert(obj_id);
  }
  for (const auto& [obj_id, _] : filters_) {
    if (current_objects.find(obj_id) == current_objects.end()) {
      object_statuses_[obj_id] = ObjectTrackingStatus::Lost;
      deleteObject(obj_id);
      LOG(INFO) << "Object " << obj_id << " marked as Lost at frame "
                << frame_k->getFrameId();
    }
    object_keyframe_statuses_[obj_id] = ObjectKeyFrameStatus::NonKeyFrame;
  }

  pose_change_info_.clear();

  // Call base solve
  return ObjectMotionSolver::solve(frame_k, frame_km1, parallel_solve);
}

bool HybridObjectMotionSolver::solveImpl(
    Frame::Ptr frame_k, Frame::Ptr frame_km1, ObjectId object_id,
    Motion3ReferenceFrame& motion_estimate) {
  // Initialize or update tracking status
  bool is_new = !filters_.exists(object_id);
  bool is_resampled = std::find(frame_k->retracked_objects_.begin(),
                                frame_k->retracked_objects_.end(),
                                object_id) != frame_k->retracked_objects_.end();

  // How does this not break if there is no previous tracking status?
  const ObjectTrackingStatus previous_tracking_state =
      object_statuses_[object_id];

  LOG(INFO) << "Previous tracking state " << to_string(previous_tracking_state)
            << " " << info_string(frame_k->getFrameId(), object_id);

  // get the corresponding feature pairs
  AbsolutePoseCorrespondences dynamic_correspondences;
  bool corr_result = frame_k->getDynamicCorrespondences(
      dynamic_correspondences, *frame_km1, object_id,
      frame_k->landmarkWorldKeypointCorrespondance());

  const size_t& n_matches = dynamic_correspondences.size();

  TrackletIds all_tracklets;
  std::transform(dynamic_correspondences.begin(), dynamic_correspondences.end(),
                 std::back_inserter(all_tracklets),
                 [](const AbsolutePoseCorrespondence& corres) {
                   return corres.tracklet_id_;
                 });
  CHECK_EQ(all_tracklets.size(), n_matches);

  Pose3SolverResult geometric_result =
      pnp_ransac_solver_.solve3d2d(dynamic_correspondences);

  TrackletIds inlier_tracklets = geometric_result.inliers;
  const TrackletIds& outlier_tracklets = geometric_result.outliers;

  const size_t num_inliers =
      inlier_tracklets.size();  // after outlier rejection

  if (inlier_tracklets.size() < 4 ||
      geometric_result.status != TrackingStatus::VALID) {
    LOG(WARNING) << "Could not make initial frame for object " << object_id
                 << " as not enough inlier tracks!";
    object_statuses_[object_id] = ObjectTrackingStatus::PoorlyTracked;

    return false;
  } else {
    object_statuses_[object_id] = ObjectTrackingStatus::WellTracked;
  }

  bool object_retracked = false;
  if (previous_tracking_state == ObjectTrackingStatus::PoorlyTracked ||
      previous_tracking_state == ObjectTrackingStatus::Lost) {
    LOG(INFO) << "Previous tracking status "
              << to_string(previous_tracking_state) << " setting to retracked";
    object_retracked = true;
  }

  const ObjectTrackingStatus current_object_state = object_statuses_[object_id];

  const gtsam::Pose3 X_W_k = frame_k->getPose();
  const gtsam::Pose3 G_W = geometric_result.best_result;

  gtsam::Pose3 G_W_inv = G_W.inverse();

  if (true) {
    auto refinement_result = optical_flow_pose_solver_.optimizeAndUpdate(
        frame_km1, frame_k, inlier_tracklets, G_W);
    // still need to take the inverse as we get the inverse of G out
    // update G_W_inv
    G_W_inv = refinement_result.best_result.refined_pose.inverse();
    // inliers should be a subset of the original refined inlier tracks
    inlier_tracklets = refinement_result.inliers;

    // VLOG(10) << "Refined object " << object_id
    //           << "pose with optical flow - error before: "
    //           << flow_opt_result.error_before.value_or(NaN)
    //           << " error_after: " <<
    //           flow_opt_result.error_after.value_or(NaN);
  }
  const gtsam::Pose3 H_W_km1_k_pnp = X_W_k * G_W_inv;

  if (!is_new) {
    auto filter = filters_.at(object_id);
    CHECK_NOTNULL(filter);
    // covers poorly tracked and lost (must be well-tracked to get here)
    if (object_retracked) {
      LOG(INFO)
          << "Object was poorly tracked or LOST previously. Creating new KF "
             "from centroid "
          << info_string(frame_km1->getFrameId(), object_id);
      // TODO: here would be to check somehow that we dont have features
      // between the last well tracked state
      gtsam::Pose3 new_KF_pose =
          constructPoseFromCentroid(frame_km1, inlier_tracklets);
      // // alert to new KF that has connection with previous KF
      // object_keyframe_statuses_[object_id] =
      //     ObjectKeyFrameStatus::AnchorKeyFrame;

      // by erasing num_kfs_per_object_ we ensure that the very next KF will be
      // an anchor KF which is necessary since the KF pose has moved!
      num_kfs_per_object_.erase(object_id);

      // if(!num_kfs_per_object_.exists(object_id)) {
      //   num_kfs_per_object_[object_id] = 1;
      // }
      // else {
      //   num_kfs_per_object_.at(object_id)++;
      // }

      filter->resetState(new_KF_pose, frame_km1->getFrameId());
      filter->predictAndUpdate(gtsam::Pose3::Identity(), frame_km1,
                               inlier_tracklets, 2);
    }
  } else {
    LOG(INFO) << "Object " << object_id << " initialized as New at frame "
              << frame_k->getFrameId();
    auto filter = createAndInsertFilter(object_id, frame_km1, inlier_tracklets);
    filter->predictAndUpdate(gtsam::Pose3::Identity(), frame_km1,
                             inlier_tracklets, 2);

    num_kfs_per_object_.erase(object_id);
  }

  // if(!is_new) {
  //   // at this point the current tracking status can either be NEW or
  //   // WellTracked and in this condition block is MUST be WellTRacked
  //   CHECK_EQ(current_object_state, ObjectTrackingStatus::WellTracked);

  //   auto filter = filters_.at(object_id);
  //   CHECK_NOTNULL(filter);
  //   // needs resetting ie. resampling or made keyframe
  //   if(is_resampled || object_retracked) {
  //     gtsam::Pose3 new_KF_pose;
  //     // object_retracked is equivalaent to saying the previous state was
  //     LOST
  //     // or POORLY_TRACKED
  //     if (object_retracked) {
  //       LOG(INFO)
  //           << "Object was poorly tracked or LOST previously. Creating new KF
  //           "
  //             "from centroid "
  //           << info_string(frame_km1->getFrameId(), object_id);
  //       // TODO: here would be to check somehow that we dont have features
  //       // between the last well tracked state
  //       new_KF_pose = constructPoseFromCentroid(frame_km1, inlier_tracklets);
  //       // // alert to new KF that has connection with previous KF
  //       // object_keyframe_statuses_[object_id] =
  //       //     ObjectKeyFrameStatus::AnchorKeyFrame;

  //     }
  //     else if (previous_tracking_state == ObjectTrackingStatus::WellTracked)
  //     {
  //       // if well tracked than the previous update should be from the
  //       immediate
  //       // previous frame!
  //       CHECK_EQ(filter->getFrameId(), frame_km1->getFrameId())
  //           << "j=" << object_id << " k=" << filter->getFrameId();
  //       // this is the pose as the last frame (ie k-1) which will serve as
  //       the
  //       // new keyframe pose
  //       new_KF_pose = filter->getPose();
  //     } else {
  //       LOG(FATAL) << "Should not get here! Previous track state "
  //                 << to_string(previous_tracking_state)
  //                 << " Curent track state "
  //                 << to_string(object_statuses_[object_id])
  //                 << " j=" << object_id;
  //     }

  //     filter->resetState(new_KF_pose, frame_km1->getFrameId());
  //     filter->predictAndUpdate(gtsam::Pose3::Identity(), frame_km1,
  //                            inlier_tracklets, 2);
  //   }
  // }
  // else {
  //   LOG(INFO) << "Object " << object_id << " initialized as New at frame "
  //             << frame_k->getFrameId();
  //   auto filter = createAndInsertFilter(object_id, frame_km1,
  //   inlier_tracklets); filter->predictAndUpdate(gtsam::Pose3::Identity(),
  //   frame_km1,
  //                            inlier_tracklets, 2);
  // }

  auto filter = filters_.at(object_id);
  CHECK_NOTNULL(filter);

  filter->predictAndUpdate(H_W_km1_k_pnp, frame_k, inlier_tracklets, 2);

  const gtsam::Pose3 H_w_km1_k = filter->getF2FMotion();

  motion_estimate = Motion3ReferenceFrame(
      H_w_km1_k, Motion3ReferenceFrame::Style::F2F, ReferenceFrame::GLOBAL,
      frame_km1->getFrameId(), frame_k->getFrameId());

  frame_k->dynamic_features_.markOutliers(outlier_tracklets);

  // if reset THIS frame then reset after motion estimation so the filter
  // estimates a motion up to k and then reset for the next frame? resampled but
  // not anything else? What if we resample at the same frame we were
  // ret-tracked? if an object is re-tracked it sort of implies it was
  // re-sampled?

  // all KF logic here!
  if (is_resampled && !object_retracked) {
    if (previous_tracking_state == ObjectTrackingStatus::WellTracked) {
      //  before reset!
      ObjectPoseChangeInfo info;
      info.frame_id = filter->getFrameId();
      info.H_W_KF_k = filter->getKeyFramedMotionReference();
      info.L_W_KF = filter->getKeyFramePose();
      info.L_W_k = filter->getPose();

      info.regular_keyframe = false;
      info.anchor_keyframe = false;

      info.motion_track_status = current_object_state;

      // this is the first keyframe for this object
      // and therefore the from frame should become the (new) anchor frame
      if (!num_kfs_per_object_.exists(object_id)) {
        LOG(INFO) << "object j=" << object_id << "is first KF: making anchor";

        num_kfs_per_object_[object_id] = 1;
        object_keyframe_statuses_[object_id] =
            ObjectKeyFrameStatus::AnchorKeyFrame;
      } else {
        // if not the first keyframe for this object then no need to alert
        // backend that this should be a Anchor KF
        num_kfs_per_object_.at(object_id)++;
        object_keyframe_statuses_[object_id] =
            ObjectKeyFrameStatus::RegularKeyFrame;

        LOG(INFO) << "object j=" << object_id
                  << "is not first KF: making regular";
      }

      const auto object_kf_status = object_keyframe_statuses_[object_id];

      if (object_kf_status == ObjectKeyFrameStatus::AnchorKeyFrame) {
        info.regular_keyframe = true;
        info.anchor_keyframe = true;
      } else if (object_kf_status == ObjectKeyFrameStatus::RegularKeyFrame) {
        info.regular_keyframe = true;
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
                << " to: " << info.H_W_KF_k.to() << " with regular kf "
                << std::boolalpha << info.regular_keyframe << " anchor kf "
                << info.anchor_keyframe;

      // Make threaf safe!!
      pose_change_info_.insert2(object_id, info);

      LOG(INFO) << "object j=" << object_id
                << "resampled. Resetting state to k=" << frame_k->getFrameId();
      // if well tracked than the previous update should be from the immediate
      // previous frame!
      CHECK_EQ(filter->getFrameId(), frame_k->getFrameId())
          << "j=" << object_id << " k=" << filter->getFrameId();
      // NOTE: here we are using the CURRENT frame for predict and update
      gtsam::Pose3 new_KF_pose = filter->getPose();
      filter->resetState(new_KF_pose, frame_k->getFrameId());
      filter->predictAndUpdate(gtsam::Pose3::Identity(), frame_k,
                               inlier_tracklets, 2);

      // logic RN for KFing is tied to resampling

    } else {
      LOG(FATAL) << "Should not get here! Previous track state "
                 << to_string(previous_tracking_state) << " Curent track state "
                 << to_string(object_statuses_[object_id])
                 << " j=" << object_id;
    }
  }

  return true;
}

void HybridObjectMotionSolver::deleteObject(ObjectId object_id) {
  filters_.erase(object_id);
  num_kfs_per_object_.erase(object_id);
}

// bool HybridObjectMotionSolver::solveImpl(
//     Frame::Ptr frame_k, Frame::Ptr frame_km1, ObjectId object_id,
//     Motion3ReferenceFrame& motion_estimate) {
//   // Initialize or update tracking status
//   bool is_new = !filters_.exists(object_id);
//   bool is_resampled = std::find(frame_k->retracked_objects_.begin(),
//                                 frame_k->retracked_objects_.end(),
//                                 object_id) !=
//                                 frame_k->retracked_objects_.end();

//   // How does this not break if there is no previous tracking status?
//   const ObjectTrackingStatus previous_tracking_state =
//       object_statuses_[object_id];

//   LOG(INFO) << "Previous tracking state " <<
//   to_string(previous_tracking_state)
//             << " " << info_string(frame_k->getFrameId(), object_id);

//   // get the corresponding feature pairs
//   AbsolutePoseCorrespondences dynamic_correspondences;
//   bool corr_result = frame_k->getDynamicCorrespondences(
//       dynamic_correspondences, *frame_km1, object_id,
//       frame_k->landmarkWorldKeypointCorrespondance());

//   const size_t& n_matches = dynamic_correspondences.size();

//   TrackletIds all_tracklets;
//   std::transform(dynamic_correspondences.begin(),
//   dynamic_correspondences.end(),
//                  std::back_inserter(all_tracklets),
//                  [](const AbsolutePoseCorrespondence& corres) {
//                    return corres.tracklet_id_;
//                  });
//   CHECK_EQ(all_tracklets.size(), n_matches);

//   Pose3SolverResult geometric_result =
//       pnp_ransac_solver_.solve3d2d(dynamic_correspondences);

//   const TrackletIds& inlier_tracklets = geometric_result.inliers;

//   const size_t num_inliers =
//       inlier_tracklets.size();  // after outlier rejection

//   if (inlier_tracklets.size() < 4 ||
//       geometric_result.status != TrackingStatus::VALID) {
//     LOG(WARNING) << "Could not make initial frame for object " << object_id
//                  << " as not enough inlier tracks!";
//     object_statuses_[object_id] = ObjectTrackingStatus::PoorlyTracked;

//     return false;
//   }

//   const gtsam::Pose3 G_w_inv_pnp = geometric_result.best_result.inverse();
//   const gtsam::Pose3 H_w_km1_k_pnp = frame_k->getPose() * G_w_inv_pnp;

//   // object_retracked is equivalaent to saying the previous state was Lost or
//   // PoorlyTracked
//   bool object_retracked = false;
//   if (previous_tracking_state == ObjectTrackingStatus::PoorlyTracked ||
//       previous_tracking_state == ObjectTrackingStatus::Lost) {
//     LOG(INFO) << "Previous tracking status "
//               << to_string(previous_tracking_state) << " setting to
//               retracked";
//     object_retracked = true;
//   }

//   auto should_object_KF = [&](ObjectId object_id) -> bool {
//     bool temporal_kf = false;
//     if (!is_new) {
//       auto filter = filters_.at(object_id);
//       FrameId last_kf_id = filter->getKeyFrameId();

//       // TODO: right now force object KF to happen at some frequency for
//       testing if (frame_k->getFrameId() > 5 &&
//           frame_k->getFrameId() - last_kf_id > 5u) {
//         VLOG(5) << "Long time since last KF for j=" << object_id;
//         temporal_kf = true;
//       }
//     }
//     // TODO: making temporal kf proves problems - show interaction between
//     // "should reset" and not actually needing a reset is maybe problematic?

//     // return is_resampled || temporal_kf;
//     // return is_resampled;
//     // must KF if retracked!!
//     return is_resampled || object_retracked;
//   };

//   // includes the isa_resampled logic
//   const bool should_kf = should_object_KF(object_id);

//   if (is_new) {
//     object_statuses_[object_id] = ObjectTrackingStatus::New;
//     LOG(INFO) << "Object " << object_id << " initialized as New at frame "
//               << frame_k->getFrameId();
//     // dont update keyframe status to anchor yet - only do so if object
//     // successfully created
//   }
//   // else if (is_resampled) {
//   else if (should_kf) {
//     // what about number of points? ReTracked should indicate o
//     if (object_retracked) {
//       object_statuses_[object_id] = ObjectTrackingStatus::WellTracked;
//       object_keyframe_statuses_[object_id] =
//           ObjectKeyFrameStatus::AnchorKeyFrame;
//       LOG(INFO) << "Object " << object_id << "ReTracked at "
//                 << frame_k->getFrameId() << ", set to AnchorKeyFrame";
//       // force reset to create motion from k-1 to k?
//       filter_needs_reset_[object_id] = true;
//     } else {
//       LOG(INFO) << "Object " << object_id
//                 << "resampled & WellTracked  at frame " <<
//                 frame_k->getFrameId()
//                 << ", set to RegularKeyFrame";
//       object_keyframe_statuses_[object_id] =
//           ObjectKeyFrameStatus::RegularKeyFrame;
//     }
//     // if (num_inliers > 4) {
//     //   object_statuses_[object_id] = ObjectTrackingStatus::WellTracked;
//     //   LOG(INFO) << "Object " << object_id
//     //             << "resampled & WellTracked  at frame " <<
//     //             frame_k->getFrameId()
//     //             << ", set to RegularKeyFrame";
//     //   object_keyframe_statuses_[object_id] =
//     //       ObjectKeyFrameStatus::RegularKeyFrame;
//     // } else {
//     //   // TODO: this should actually be that all features are new!!!!!
//     //   //  keep object as poorly tracked if was poorly tracked before
//     //   object_statuses_[object_id] = ObjectTrackingStatus::WellTracked;
//     //   object_keyframe_statuses_[object_id] =
//     //       ObjectKeyFrameStatus::AnchorKeyFrame;
//     //   LOG(INFO) << "Object " << object_id
//     //             << "resampled & WellTracked  at frame " <<
//     //             frame_k->getFrameId()
//     //             << ", set to AnchorKeyFrame";
//     // }
//   } else {
//     object_statuses_[object_id] = ObjectTrackingStatus::WellTracked;
//   }

//   const ObjectTrackingStatus current_object_state =
//   object_statuses_[object_id];

//   bool new_or_reset_object = false;
//   bool filter_needs_reset = false;
//   // bool new_object = false;
//   // bool object_reset = false;
//   if (!is_new) {
//     // should only get here if well tracked!
//     // what is actual logic here -> needs reset separate to should KF in this
//     // condition but somehow couplied in should_kf decision logic?
//     if (filterNeedsReset(object_id)) {
//       LOG(INFO) << object_id << " needs retting from last frame! current k="
//                 << frame_k->getFrameId();
//       filter_needs_reset_[object_id] = false;

//       auto filter = filters_.at(object_id);

//       gtsam::Pose3 new_KF_pose;
//       // ie. could not solve motion from the frame before
//       // if (previous_tracking_state == ObjectTrackingStatus::PoorlyTracked)
//       {

//       // at this point the current tracking status can either be NEW or
//       // WellTracked and in this condition block is MUST be WellTRacked
//       CHECK_EQ(current_object_state, ObjectTrackingStatus::WellTracked);

//       // object_retracked is equivalaent to saying the previous state was
//       LOST
//       // or POORLY_TRACKED
//       if (object_retracked) {
//         LOG(INFO)
//             << "Object was poorly tracked or LOST previously. Creating new KF
//             "
//                "from centroid "
//             << info_string(frame_km1->getFrameId(), object_id);
//         // must create new initial frame
//         // TODO: here would be to check somehow that we dont have features
//         // between the last well tracked state
//         new_KF_pose = constructPoseFromCentroid(frame_km1, inlier_tracklets);
//         // alert to new KF that has connection with previous KF
//         object_keyframe_statuses_[object_id] =
//             ObjectKeyFrameStatus::AnchorKeyFrame;

//         // TODO: if only one frame dropped maybe use constant motion model?
//       }
//       // in this case 'New' also means well tracked since it can only be set
//       if
//       // PnP was good!
//       else if (previous_tracking_state == ObjectTrackingStatus::WellTracked)
//       {
//         // if well tracked than the previous update should be from the
//         immediate
//         // previous frame!
//         CHECK_EQ(filter->getFrameId(), frame_km1->getFrameId())
//             << "j=" << object_id << " k=" << filter->getFrameId();
//         // this is the pose as the last frame (ie k-1) which will serve as
//         the
//         // new keyframe pose
//         new_KF_pose = filter->getPose();
//       } else {
//         LOG(FATAL) << "Should not get here! Previous track state "
//                    << to_string(previous_tracking_state)
//                    << " Curent track state "
//                    << to_string(object_statuses_[object_id])
//                    << " j=" << object_id;
//       }

//       // NOTE: from motion at k-1
//       filter->resetState(new_KF_pose, frame_km1->getFrameId());
//       new_or_reset_object = true;
//       // stable_frame_counts_[object_id] = 0;
//       LOG(INFO) << "Object " << object_id
//                 << " reset using frame k= " << frame_km1->getFrameId();
//     }

//     // if (is_resampled) {
//     if (should_kf) {
//       LOG(INFO) << object_id
//                 << " retracked - resetting filter k=" <<
//                 frame_k->getFrameId();

//       // actually indicates that we will have a KF motion (as long as
//       tracking
//       // was good???) in the next frame
//       filter_needs_reset_[object_id] = true;
//       LOG(INFO) << "Object " << object_id << " resampled, marked for reset";
//     }
//   }
//   // becuuse we erase it if object new in previous!
//   // not sure this is the best way to handle reappearing objects!
//   if (is_new) {
//     new_or_reset_object = true;
//     createAndInsertFilter(object_id, frame_km1, inlier_tracklets);
//   }

//   auto filter = filters_.at(object_id);
//   // update and predict should be one step so that if we dont have enough
//   points
//   // NOTE: this logic seemed pretty important to ensure the estimate was
//   good!!!
//   // we dont predict?
//   if (new_or_reset_object) {
//     filter->predictAndUpdate(gtsam::Pose3::Identity(), frame_km1,
//                              inlier_tracklets, 2);
//   }

//   filter->predictAndUpdate(H_w_km1_k_pnp, frame_k, inlier_tracklets, 2);

//   bool return_result = false;
//   if (geometric_result.status == TrackingStatus::VALID) {
//     const gtsam::Pose3 H_w_km1_k = filter->getF2FMotion();

//     Motion3SolverResult motion_result;
//     motion_result.status = geometric_result.status;
//     motion_result.inliers = geometric_result.inliers;
//     motion_result.outliers = geometric_result.outliers;

//     motion_result.best_result = Motion3ReferenceFrame(
//         H_w_km1_k, Motion3ReferenceFrame::Style::F2F, ReferenceFrame::GLOBAL,
//         frame_km1->getFrameId(), frame_k->getFrameId());

//     // TODO: make thread safe!
//     frame_k->dynamic_features_.markOutliers(motion_result.outliers);
//     // motion_estimates.insert({object_id, motion_result.best_result});
//     motion_estimate = motion_result.best_result;

//     return_result = true;
//   } else {
//     // TODO: not sure lost is the right logic here!
//     object_statuses_[object_id] = ObjectTrackingStatus::Lost;
//     // stable_frame_counts_[object_id] = 0;
//     LOG(INFO) << "Object " << object_id << " set to Lost at frame "
//               << frame_k->getFrameId();
//     // so tha
//     filters_.erase(object_id);
//     return_result = false;
//   }

//   LOG(INFO) << "Object " << object_id
//             << " final status: " << to_string(object_statuses_[object_id])
//             << " with keyframe status: "
//             << to_string(object_keyframe_statuses_[object_id]);
//   return return_result;
// }

bool HybridObjectMotionSolver::filterNeedsReset(ObjectId object_id) {
  if (filter_needs_reset_.exists(object_id)) {
    return filter_needs_reset_.at(object_id);
  }
  return false;
}

gtsam::Pose3 HybridObjectMotionSolver::constructPoseFromCentroid(
    const Frame::Ptr frame, const TrackletIds& tracklets) const {
  // important to initliase with zero values (otherwise nan's!)
  gtsam::Point3 object_position(0, 0, 0);
  size_t count = 0;
  for (TrackletId tracklet : tracklets) {
    const Feature::Ptr feature = frame->at(tracklet);
    CHECK_NOTNULL(feature);

    gtsam::Point3 lmk = frame->backProjectToCamera(feature->trackletId());
    object_position += lmk;

    count++;
  }

  object_position /= count;
  object_position = frame->getPose() * object_position;
  return gtsam::Pose3(gtsam::Rot3::Identity(), object_position);
}

std::shared_ptr<HybridObjectMotionSRIF>
HybridObjectMotionSolver::createAndInsertFilter(ObjectId object_id,
                                                Frame::Ptr frame,
                                                const TrackletIds& tracklets) {
  gtsam::Matrix33 R = gtsam::Matrix33::Identity() * 1.0;
  // Initial State Covariance P (6x6)
  gtsam::Matrix66 P = gtsam::Matrix66::Identity() * 0.3;
  // Process Model noise (6x6)
  gtsam::Matrix66 Q = gtsam::Matrix66::Identity() * 0.2;

  gtsam::Pose3 keyframe_pose = constructPoseFromCentroid(frame, tracklets);

  constexpr static double kHuberKFilter = 0.05;
  auto filter = std::make_shared<HybridObjectMotionSRIF>(
      gtsam::Pose3::Identity(), keyframe_pose, frame->getFrameId(), P, Q, R,
      frame->getCamera(), kHuberKFilter);
  filters_.insert2(object_id, filter);
  object_keyframe_statuses_[object_id] = ObjectKeyFrameStatus::AnchorKeyFrame;

  LOG(INFO) << "Created new filter for object " << object_id << " at frame "
            << frame->getFrameId();

  return filter;
}

void HybridObjectMotionSolver::updateTrajectories(
    MultiObjectTrajectories& object_trajectories,
    const MotionEstimateMap& motion_estimates, Frame::Ptr frame_k,
    Frame::Ptr frame_km1) {
  const FrameId frame_id_k = frame_k->getFrameId();
  const Timestamp timestamp_k = frame_k->getTimestamp();

  for (const auto& [object_id, motion_reference_frame] : motion_estimates) {
    CHECK(filters_.exists(object_id));

    CHECK_EQ(motion_reference_frame.from(), frame_id_k - 1u);
    CHECK_EQ(motion_reference_frame.to(), frame_id_k);

    auto filter = filters_.at(object_id);
    gtsam::Pose3 L_k_j = filter->getPose();
    object_trajectories_.insert(object_id, frame_id_k, timestamp_k,
                                PoseWithMotion{L_k_j, motion_reference_frame});
  }

  object_trajectories = object_trajectories_;
}

}  // namespace dyno
