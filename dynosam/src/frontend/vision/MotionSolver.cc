/*
 *   Copyright (c) 2023 ACFR-RPG, University of Sydney, Jesse Morris
 (jesse.morris@sydney.edu.au)
 *   All rights reserved.

 *   Permission is hereby granted, free of charge, to any person obtaining a
 copy
 *   of this software and associated documentation files (the "Software"), to
 deal
 *   in the Software without restriction, including without limitation the
 rights
 *   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 *   copies of the Software, and to permit persons to whom the Software is
 *   furnished to do so, subject to the following conditions:

 *   The above copyright notice and this permission notice shall be included in
 all
 *   copies or substantial portions of the Software.

 *   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 *   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 *   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 *   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 *   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 FROM,
 *   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE
 *   SOFTWARE.
 */

#include "dynosam/frontend/vision/MotionSolver.hpp"

#include <config_utilities/config_utilities.h>
#include <glog/logging.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>  //for now? //TODO: clean
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/Values.h>
#include <tbb/tbb.h>

#include <eigen3/Eigen/Dense>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/core/eigen.hpp>
#include <opengv/types.hpp>

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/factors/HybridFormulationFactors.hpp"
#include "dynosam/factors/LandmarkMotionTernaryFactor.hpp"
#include "dynosam/factors/Pose3FlowProjectionFactor.h"
#include "dynosam/frontend/Frontend-Definitions.hpp"
#include "dynosam/frontend/vision/VisionTools.hpp"
#include "dynosam_common/DynamicObjects.hpp"
#include "dynosam_common/Flags.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/Accumulator.hpp"
#include "dynosam_common/utils/GtsamUtils.hpp"
#include "dynosam_common/utils/Numerical.hpp"
#include "dynosam_common/utils/TimingStats.hpp"
#include "dynosam_cv/RGBDCamera.hpp"
#include "dynosam_opt/FactorGraphTools.hpp"  //TODO: clean

// GTSAM Includes
// FOR TESTING!
#include <gtsam/base/Matrix.h>
#include <gtsam/base/Vector.h>
#include <gtsam/base/numericalDerivative.h>
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/PinholeCamera.h>
#include <gtsam/geometry/Pose3.h>
// #include <gtsam/base/FullLinearSolver.h>

#include "dynosam/factors/HybridFormulationFactors.hpp"
#include "dynosam/frontend/solvers/MotionOnlySolver.hpp"
#include "dynosam/frontend/solvers/OpticalFlowAndPoseSolver.hpp"

namespace dyno {

void declare_config(OpticalFlowAndPoseOptimizer::Params& config) {
  using namespace config;

  name("OpticalFlowAndPoseOptimizerParams");
  field(config.flow_sigma, "flow_sigma");
  field(config.flow_prior_sigma, "flow_prior_sigma");
  field(config.k_huber, "k_huber");
  field(config.outlier_reject, "outlier_reject");
  field(config.flow_is_future, "flow_is_future");
}

void declare_config(MotionOnlyRefinementOptimizer::Params& config) {
  using namespace config;

  name("MotionOnlyRefinementOptimizerParams");
  field(config.landmark_motion_sigma, "landmark_motion_sigma");
  field(config.projection_sigma, "projection_sigma");
  field(config.k_huber, "k_huber");
  field(config.outlier_reject, "outlier_reject");
}

void declare_config(EgoMotionSolver::Params& config) {
  using namespace config;

  name("EgoMotionSolver::Params");
  field(config.ransac_randomize, "ransac_randomize");
  field(config.ransac_use_2point_mono, "ransac_use_2point_mono");
  field(config.optimize_2d2d_pose_from_inliers,
        "optimize_2d2d_pose_from_inliers");
  field(config.ransac_threshold_pnp, "ransac_threshold_pnp");
  field(config.optimize_3d2d_pose_from_inliers,
        "optimize_3d2d_pose_from_inliers");
  field(config.ransac_threshold_stereo, "ransac_threshold_stereo");
  field(config.optimize_3d3d_pose_from_inliers,
        "optimize_3d3d_pose_from_inliers");
  field(config.ransac_iterations, "ransac_iterations");
  field(config.ransac_probability, "ransac_probability");
}
void declare_config(ConsecutiveFrameObjectMotionSolver::Params& config) {
  using namespace config;
  name("ConsecutiveFrameObjectMotionSolver::Params");

  base<EgoMotionSolver::Params>(config);
  field(config.refine_motion_with_joint_of, "refine_motion_with_joint_of");
  field(config.refine_motion_with_3d, "refine_motion_with_3d");
  field(config.joint_of_params, "joint_optical_flow");
  field(config.object_motion_refinement_params, "object_motion_3d_refinement");
}

EgoMotionSolver::EgoMotionSolver(const Params& params,
                                 const CameraParams& camera_params)
    : params_(params), camera_params_(camera_params) {}

Pose3SolverResult EgoMotionSolver::geometricOutlierRejection2d2d(
    Frame::Ptr frame_k_1, Frame::Ptr frame_k,
    std::optional<gtsam::Rot3> R_curr_ref) {
  // get correspondences
  // RelativePoseCorrespondences correspondences;
  // // this does not create proper bearing vectors (at leas tnot for 3d-2d pnp
  // // solve) bearing vectors are also not undistorted atm!!
  // {
  //   utils::ChronoTimingStats
  //   track_dynamic_timer("mono_frame_correspondences");
  //   frame_k->getCorrespondences(correspondences, *frame_k_1,
  //                               KeyPointType::STATIC,
  //                               frame_k->imageKeypointCorrespondance());
  // }

  // Pose3SolverResult result;

  // const size_t& n_matches = correspondences.size();

  // if (n_matches < 5u) {
  //   result.status = TrackingStatus::FEW_MATCHES;
  //   return result;
  // }

  // gtsam::Matrix K = camera_params_.getCameraMatrixEigen();
  // K = K.inverse();

  // TrackletIds tracklets;
  // // NOTE: currently without distortion! the correspondences should be made
  // into
  // // bearing vector elsewhere!
  // BearingVectors ref_bearing_vectors, cur_bearing_vectors;
  // for (size_t i = 0u; i < n_matches; i++) {
  //   const auto& corres = correspondences.at(i);
  //   const Keypoint& ref_kp = corres.ref_;
  //   const Keypoint& cur_kp = corres.cur_;

  //   gtsam::Vector3 ref_versor = (K * gtsam::Vector3(ref_kp(0),
  //   ref_kp(1), 1.0)); gtsam::Vector3 cur_versor = (K *
  //   gtsam::Vector3(cur_kp(0), cur_kp(1), 1.0));

  //   ref_versor = ref_versor.normalized();
  //   cur_versor = cur_versor.normalized();

  //   ref_bearing_vectors.push_back(ref_versor);
  //   cur_bearing_vectors.push_back(cur_versor);

  //   tracklets.push_back(corres.tracklet_id_);
  // }

  // RelativePoseAdaptor adapter(ref_bearing_vectors, cur_bearing_vectors);

  // const bool use_2point_mono = params_.ransac_use_2point_mono && R_curr_ref;
  // if (use_2point_mono) {
  //   adapter.setR12((*R_curr_ref).matrix());
  // }

  // gtsam::Pose3 best_result;
  // std::vector<int> ransac_inliers;
  // bool success = false;
  // if (use_2point_mono) {
  //   success = runRansac<RelativePoseProblemGivenRot>(
  //       std::make_shared<RelativePoseProblemGivenRot>(adapter,
  //                                                     params_.ransac_randomize),
  //       params_.ransac_threshold_mono, params_.ransac_iterations,
  //       params_.ransac_probability, params_.optimize_2d2d_pose_from_inliers,
  //       best_result, ransac_inliers);
  // } else {
  //   success = runRansac<RelativePoseProblem>(
  //       std::make_shared<RelativePoseProblem>(
  //           adapter, RelativePoseProblem::NISTER, params_.ransac_randomize),
  //       params_.ransac_threshold_mono, params_.ransac_iterations,
  //       params_.ransac_probability, params_.optimize_2d2d_pose_from_inliers,
  //       best_result, ransac_inliers);
  // }

  // if (!success) {
  //   result.status = TrackingStatus::INVALID;
  // } else {
  //   constructTrackletInliers(result.inliers, result.outliers,
  //   correspondences,
  //                            ransac_inliers, tracklets);
  //   // NOTE: 2-point always returns the identity rotation, hence we have to
  //   // substitute it:
  //   if (use_2point_mono) {
  //     CHECK(R_curr_ref->equals(best_result.rotation()));
  //   }
  //   result.status = TrackingStatus::VALID;
  //   result.best_result = best_result;
  // }

  // return result;
}

Pose3SolverResult EgoMotionSolver::geometricOutlierRejection3d2d(
    Frame::Ptr frame_k_1, Frame::Ptr frame_k,
    std::optional<gtsam::Rot3> R_curr_ref) {
  // AbsolutePoseCorrespondences correspondences;
  // // this does not create proper bearing vectors (at leas tnot for 3d-2d pnp
  // // solve) bearing vectors are also not undistorted atm!!
  // // TODO: change to use landmarkWorldProjectedBearingCorrespondance and then
  // // change motion solver to take already projected bearing vectors
  // {
  //   utils::ChronoTimingStats
  //   timer("motion_solver.solve_3d2d.correspondances");
  //   frame_k->getCorrespondences(correspondences, *frame_k_1,
  //                               KeyPointType::STATIC,
  //                               frame_k->landmarkWorldKeypointCorrespondance());
  // }

  // return geometricOutlierRejection3d2d(correspondences, R_curr_ref);
}

Pose3SolverResult EgoMotionSolver::geometricOutlierRejection3d2d(
    const AbsolutePoseCorrespondences& correspondences,
    std::optional<gtsam::Rot3> R_curr_ref) {
  // utils::ChronoTimingStats timer("motion_solver.solve_3d2d");
  // Pose3SolverResult result;
  // const size_t& n_matches = correspondences.size();

  // if (n_matches < 5u) {
  //   result.status = TrackingStatus::FEW_MATCHES;
  //   VLOG(5) << "3D2D tracking failed as there are to few matches" <<
  //   n_matches; return result;
  // }

  // gtsam::Matrix K = camera_params_.getCameraMatrixEigen();
  // K = K.inverse();

  // TrackletIds tracklets, inliers, outliers;
  // // NOTE: currently without distortion! the correspondences should be made
  // into
  // // bearing vector elsewhere!
  // BearingVectors bearing_vectors;
  // Landmarks points;
  // for (size_t i = 0u; i < n_matches; i++) {
  //   const AbsolutePoseCorrespondence& corres = correspondences.at(i);
  //   const Keypoint& kp = corres.cur_;
  //   // make Bearing vector
  //   gtsam::Vector3 versor = (K * gtsam::Vector3(kp(0), kp(1), 1.0));
  //   versor = versor.normalized();
  //   bearing_vectors.push_back(versor);

  //   points.push_back(corres.ref_);
  //   tracklets.push_back(corres.tracklet_id_);
  // }

  // VLOG(20) << "Collected " << tracklets.size() << " initial correspondances";

  // const double reprojection_error = params_.ransac_threshold_pnp;
  // const double avg_focal_length =
  //     0.5 * static_cast<double>(camera_params_.fx() + camera_params_.fy());
  // const double threshold =
  //     1.0 - std::cos(std::atan(std::sqrt(2.0) * reprojection_error /
  //                              avg_focal_length));

  // AbsolutePoseAdaptor adapter(bearing_vectors, points);

  // if (R_curr_ref) {
  //   adapter.setR(R_curr_ref->matrix());
  // }

  // gtsam::Pose3 best_result;
  // std::vector<int> ransac_inliers;

  // bool success;
  // {
  //   utils::ChronoTimingStats timer("motion_solver.solve_3d2d.ransac");
  //   success = runRansac<AbsolutePoseProblem>(
  //       std::make_shared<AbsolutePoseProblem>(adapter,
  //                                             AbsolutePoseProblem::KNEIP),
  //       threshold, params_.ransac_iterations, params_.ransac_probability,
  //       params_.optimize_3d2d_pose_from_inliers, best_result,
  //       ransac_inliers);
  // }

  // constructTrackletInliers(result.inliers, result.outliers, correspondences,
  //                          ransac_inliers, tracklets);

  // if (success) {
  //   if (result.inliers.size() < 5u) {
  //     result.status = TrackingStatus::FEW_MATCHES;
  //   } else {
  //     result.status = TrackingStatus::VALID;
  //     result.best_result = best_result;
  //   }

  // } else {
  //   result.status = TrackingStatus::INVALID;
  // }

  // return result;
}

void OpticalFlowAndPoseOptimizer::updateFrameOutliersWithResult(
    const Result& result, Frame::Ptr frame_k_1, Frame::Ptr frame_k) const {
  utils::ChronoTimingStats timer("of_motion_solver.update_frame");
  const auto& image_container_k = frame_k->image_container_;
  const cv::Mat& motion_mask = image_container_k.objectMotionMask();

  auto camera = frame_k->camera_;
  const auto& refined_inliers = result.inliers;
  const auto& refined_flows = result.best_result.refined_flows;

  // outliers from the result. We will update this vector with new outliers
  auto refined_outliers = result.outliers;

  for (size_t i = 0; i < refined_inliers.size(); i++) {
    TrackletId tracklet_id = refined_inliers.at(i);
    gtsam::Point2 refined_flow = refined_flows.at(i);

    Feature::Ptr feature_k_1 = frame_k_1->at(tracklet_id);
    Feature::Ptr feature_k = frame_k->at(tracklet_id);

    CHECK_EQ(feature_k->objectId(), result.best_result.object_id);

    const Keypoint kp_k_1 = feature_k_1->keypoint();
    Keypoint refined_keypoint = kp_k_1 + refined_flow;

    // check boundaries?
    if (!camera->isKeypointContained(refined_keypoint)) {
      refined_outliers.push_back(tracklet_id);
      continue;
    }

    ObjectId predicted_label =
        functional_keypoint::at<ObjectId>(refined_keypoint, motion_mask);
    if (predicted_label != result.best_result.object_id) {
      refined_outliers.push_back(tracklet_id);
      // TODO: other fields of the feature does not get updated? Inconsistencies
      // as measured flow, predicted kp etc are no longer correct!!?
      continue;
    }

    // update current keypoint
    feature_k->keypoint(refined_keypoint);
    // update refined frlow
    feature_k_1->measuredFlow(refined_flow);
    // update refined predicted keypoint
    feature_k_1->predictedKeypoint(refined_keypoint);

    // Logic is a bit convoluted and dependant on other things
    // if we have optical flow (from k to k+1 due to historical reaseons)
    // then use this to set the predicted keypoint
    // if we are tracking using the provided flow then this MUST get set
    // otherwise tracking will fail as tracking with provided optical flow uses
    // the predicted keypoint to get the next keypoint!
    if (image_container_k.hasOpticalFlow()) {
      const cv::Mat& flow_image = image_container_k.opticalFlow();
      const int x = functional_keypoint::u(refined_keypoint);
      const int y = functional_keypoint::v(refined_keypoint);
      double flow_xe = static_cast<double>(flow_image.at<cv::Vec2f>(y, x)[0]);
      double flow_ye = static_cast<double>(flow_image.at<cv::Vec2f>(y, x)[1]);

      OpticalFlow new_measured_flow(flow_xe, flow_ye);
      feature_k->measuredFlow(new_measured_flow);
      // TODO: check predicted flow is within image
      Keypoint predicted_kp = Feature::CalculatePredictedKeypoint(
          refined_keypoint, new_measured_flow);
      feature_k->predictedKeypoint(predicted_kp);
    } else {
      // we dont have a predicted flow but we assume we are using KLT tracking
      //  in this case we dont have a predicted keypoint.
      //  just use the refined flow as the measured flow
      feature_k->measuredFlow(refined_flow);
      Keypoint predicted_kp =
          Feature::CalculatePredictedKeypoint(refined_keypoint, refined_flow);
      feature_k->predictedKeypoint(predicted_kp);
    }
  }

  // update tracks
  for (const auto& outlier_tracklet : refined_outliers) {
    Feature::Ptr feature_k_1 = frame_k_1->at(outlier_tracklet);
    Feature::Ptr feature_k = frame_k->at(outlier_tracklet);

    CHECK(feature_k_1->usable());
    CHECK(feature_k->usable());

    feature_k->markOutlier();
    feature_k_1->markOutlier();
  }

  // refresh depth information for each frame
  CHECK(frame_k->updateDepths());
}

Pose3SolverResult EgoMotionSolver::geometricOutlierRejection3d3d(
    Frame::Ptr frame_k_1, Frame::Ptr frame_k,
    std::optional<gtsam::Rot3> R_curr_ref) {
  // PointCloudCorrespondences correspondences;
  // {
  //   utils::ChronoTimingStats("pc_correspondences");
  //   frame_k->getCorrespondences(
  //       correspondences, *frame_k_1, KeyPointType::STATIC,
  //       frame_k->landmarkWorldPointCloudCorrespondance());
  // }

  // return geometricOutlierRejection3d3d(correspondences, R_curr_ref);
}

Pose3SolverResult EgoMotionSolver::geometricOutlierRejection3d3d(
    const PointCloudCorrespondences& correspondences,
    std::optional<gtsam::Rot3> R_curr_ref) {
  const size_t& n_matches = correspondences.size();

  // Pose3SolverResult result;
  // if (n_matches < 5) {
  //   result.status = TrackingStatus::FEW_MATCHES;
  //   return result;
  // }

  // TrackletIds tracklets;
  // BearingVectors ref_bearing_vectors, cur_bearing_vectors;

  // for (size_t i = 0u; i < n_matches; i++) {
  //   const auto& corres = correspondences.at(i);
  //   const Landmark& ref_lmk = corres.ref_;
  //   const Landmark& cur_lmk = corres.cur_;
  //   ref_bearing_vectors.push_back(ref_lmk);
  //   cur_bearing_vectors.push_back(cur_lmk);

  //   tracklets.push_back(corres.tracklet_id_);
  // }

  // //! Setup adapter.
  // Adapter3d3d adapter(ref_bearing_vectors, cur_bearing_vectors);

  // if (R_curr_ref) {
  //   adapter.setR12((*R_curr_ref).matrix());
  // }

  // gtsam::Pose3 best_result;
  // std::vector<int> ransac_inliers;

  // bool success = runRansac<Problem3d3d>(
  //     std::make_shared<Problem3d3d>(adapter, params_.ransac_randomize),
  //     params_.ransac_threshold_stereo, params_.ransac_iterations,
  //     params_.ransac_probability, params_.optimize_3d3d_pose_from_inliers,
  //     best_result, ransac_inliers);

  // if (success) {
  //   constructTrackletInliers(result.inliers, result.outliers,
  //   correspondences,
  //                            ransac_inliers, tracklets);

  //   result.status = TrackingStatus::VALID;
  //   result.best_result = best_result;
  // } else {
  //   result.status = TrackingStatus::INVALID;
  // }

  // return result;
}

MultiObjectTrajectories ObjectMotionSolver::solve(Frame::Ptr frame_k,
                                                  Frame::Ptr frame_k_1) {
  ObjectIds failed_object_tracks;
  MotionEstimateMap motion_estimates;

  // if only 1 object, no point parallelising
  if (motion_estimates.size() <= 1) {
    for (const auto& [object_id, observations] :
         frame_k->object_observations_) {
      if (!solveImpl(frame_k, frame_k_1, object_id, motion_estimates)) {
        VLOG(5) << "Could not solve motion for object " << object_id
                << " from frame " << frame_k_1->getFrameId() << " -> "
                << frame_k->getFrameId();
        failed_object_tracks.push_back(object_id);
      }
    }
  } else {
    std::mutex mutex;
    // paralleilise the process of each function call.
    tbb::parallel_for_each(
        frame_k->object_observations_.begin(),
        frame_k->object_observations_.end(),
        [&](const std::pair<ObjectId, SingleDetectionResult>& pair) {
          const auto object_id = pair.first;
          if (!solveImpl(frame_k, frame_k_1, object_id, motion_estimates)) {
            VLOG(5) << "Could not solve motion for object " << object_id
                    << " from frame " << frame_k_1->getFrameId() << " -> "
                    << frame_k->getFrameId();

            std::lock_guard<std::mutex> lk(mutex);
            failed_object_tracks.push_back(object_id);
          }
        });
  }

  // remove objects from the object observations list
  // does not remove the features etc but stops the object being propogated to
  // the backend as we loop over the object observations in the constructOutput
  // function
  for (auto object_id : failed_object_tracks) {
    frame_k->object_observations_.erase(object_id);
  }

  // ObjectPoseMap poses;
  // ObjectMotionMap motions;
  MultiObjectTrajectories object_trajectories;
  updateTrajectories(object_trajectories, motion_estimates, frame_k, frame_k_1);

  // updateMotions(motions, motion_estimates, frame_k, frame_k_1);
  // updatePoses(poses, motion_estimates, frame_k, frame_k_1);
  // return {motions, poses};
  return object_trajectories;
}

ConsecutiveFrameObjectMotionSolver::ConsecutiveFrameObjectMotionSolver(
    const ConsecutiveFrameObjectMotionSolver::Params& params,
    const CameraParams& camera_params)
    : EgoMotionSolver(static_cast<const EgoMotionSolver::Params&>(params),
                      camera_params),
      object_motion_params(params) {}

void ConsecutiveFrameObjectMotionSolver::updateTrajectories(
    MultiObjectTrajectories& object_trajectories,
    const MotionEstimateMap& motion_estimates, Frame::Ptr frame_k,
    Frame::Ptr frame_k_1) {
  gtsam::Point3Vector object_centroids_k_1, object_centroids_k;

  for (const auto& [object_id, motion_estimate] : motion_estimates) {
    auto object_points = FeatureFilterIterator(
        const_cast<FeatureContainer&>(frame_k_1->dynamic_features_),
        [object_id, &frame_k](const Feature::Ptr& f) -> bool {
          return Feature::IsUsable(f) && f->objectId() == object_id &&
                 frame_k->exists(f->trackletId()) &&
                 frame_k->isFeatureUsable(f->trackletId());
        });

    gtsam::Point3 centroid_k_1(0, 0, 0);
    gtsam::Point3 centroid_k(0, 0, 0);
    size_t count = 0;
    for (const auto& feature : object_points) {
      gtsam::Point3 lmk_k_1 =
          frame_k_1->backProjectToCamera(feature->trackletId());
      centroid_k_1 += lmk_k_1;

      gtsam::Point3 lmk_k = frame_k->backProjectToCamera(feature->trackletId());
      centroid_k += lmk_k;

      count++;
    }

    centroid_k_1 /= count;
    centroid_k /= count;

    centroid_k_1 = frame_k_1->getPose() * centroid_k_1;
    centroid_k = frame_k->getPose() * centroid_k;

    object_centroids_k_1.push_back(centroid_k_1);
    object_centroids_k.push_back(centroid_k);
  }

  if (FLAGS_init_object_pose_from_gt) {
    CHECK(object_motion_params.ground_truth_packets_request)
        << "FLAGS_init_object_pose_from_gt is true but no ground truth packets "
           "hook is set!";

    const auto ground_truth_packets =
        object_motion_params.ground_truth_packets_request();
    LOG_IF(WARNING, !ground_truth_packets.has_value())
        << "FLAGS_init_object_pose_from_gt but no ground truth provided! "
           "Object poses will be initalised using centroid!";

    // dyno::propogateObjectPoses(object_poses_, motion_estimates,
    //                            object_centroids_k_1, object_centroids_k,
    //                            frame_k->getFrameId(), ground_truth_packets);
    dyno::propogateObjectTrajectory(
        object_trajectories_, motion_estimates, object_centroids_k_1,
        object_centroids_k, frame_k->getFrameId(), frame_k->getTimestamp(),
        frame_k_1->getTimestamp(), ground_truth_packets);
  } else {
    // dyno::propogateObjectPoses(object_poses_, motion_estimates,
    //                            object_centroids_k_1, object_centroids_k,
    //                            frame_k->getFrameId());

    dyno::propogateObjectTrajectory(
        object_trajectories_, motion_estimates, object_centroids_k_1,
        object_centroids_k, frame_k->getFrameId(), frame_k->getTimestamp(),
        frame_k_1->getTimestamp());
  }

  // TODO: these propogations should also update the motions
  object_trajectories = object_trajectories_;
  // object_poses = object_poses_;
}

// void ConsecutiveFrameObjectMotionSolver::updateMotions(
//     ObjectMotionMap& object_motions, const MotionEstimateMap&
//     motion_estimates, Frame::Ptr frame_k, Frame::Ptr) {
//   const FrameId frame_id_k = frame_k->getFrameId();
//   for (const auto& [object_id, motion_reference_frame] : motion_estimates) {
//     object_motions_.insert22(object_id, frame_id_k, motion_reference_frame);
//   }
//   object_motions = object_motions_;
// }

bool ConsecutiveFrameObjectMotionSolver::solveImpl(
    Frame::Ptr frame_k, Frame::Ptr frame_k_1, ObjectId object_id,
    MotionEstimateMap& motion_estimates) {
  Motion3SolverResult result = geometricOutlierRejection3d2d(
      frame_k_1, frame_k, frame_k->getPose(), object_id);

  frame_k->dynamic_features_.markOutliers(result.outliers);

  VLOG(15) << " object motion estimate " << object_id << " at frame "
           << frame_k->frame_id_
           << (result.status == TrackingStatus::VALID ? " success "
                                                      : " failure ")
           << ":\n"
           << "- Tracking Status: " << to_string(result.status) << '\n'
           << "- Total Correspondences: "
           << result.inliers.size() + result.outliers.size() << '\n'
           << "\t- # inliers: " << result.inliers.size() << '\n'
           << "\t- # outliers: " << result.outliers.size() << '\n';

  // if valid, remove outliers and add to motion estimation
  if (result.status == TrackingStatus::VALID) {
    motion_estimates.insert({object_id, result.best_result});
    return true;
  } else {
    return false;
  }
}

Motion3SolverResult
ConsecutiveFrameObjectMotionSolver::geometricOutlierRejection3d2d(
    Frame::Ptr frame_k_1, Frame::Ptr frame_k, const gtsam::Pose3& T_world_k,
    ObjectId object_id) {
  utils::ChronoTimingStats timer("motion_solver.object_solve3d2d");
  AbsolutePoseCorrespondences dynamic_correspondences;
  // get the corresponding feature pairs
  bool corr_result = frame_k->getDynamicCorrespondences(
      dynamic_correspondences, *frame_k_1, object_id,
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
      EgoMotionSolver::geometricOutlierRejection3d2d(dynamic_correspondences);
  Pose3SolverResult pose_result = geometric_result;

  Motion3SolverResult motion_result;
  motion_result.status = pose_result.status;

  if (pose_result.status == TrackingStatus::VALID) {
    TrackletIds refined_inlier_tracklets = pose_result.inliers;

    {
      CHECK_EQ(pose_result.inliers.size() + pose_result.outliers.size(),
               n_matches);

      // debug only (just checking that the inlier/outliers we get from the
      // geometric rejection match the original one)
      TrackletIds extracted_all_tracklets = refined_inlier_tracklets;
      extracted_all_tracklets.insert(extracted_all_tracklets.end(),
                                     pose_result.outliers.begin(),
                                     pose_result.outliers.end());
      CHECK_EQ(all_tracklets.size(), extracted_all_tracklets.size());
    }

    gtsam::Pose3 G_w = pose_result.best_result.inverse();
    if (object_motion_params.refine_motion_with_joint_of) {
      OpticalFlowAndPoseOptimizer flow_optimizer(
          object_motion_params.joint_of_params);
      // Use the original result as the input to the refine joint optical flow
      // function the result.best_result variable is actually equivalent to
      // ^wG^{-1} and we want to solve something in the form e(T, flow) =
      // [u,v]_{k-1} + {k-1}_flow_k - pi(T^{-1}^wm_{k-1}) so T must take the
      // point from k-1 in the world frame to the local frame at k-1 ^wG^{-1} =
      //^wX_k \: {k-1}^wH_k (which takes does this) but the error term uses the
      // inverse of T hence we must parse in the inverse of G
      auto flow_opt_result = flow_optimizer.optimizeAndUpdate<CalibrationType>(
          frame_k_1, frame_k, refined_inlier_tracklets,
          pose_result.best_result);
      // still need to take the inverse as we get the inverse of G out
      G_w = flow_opt_result.best_result.refined_pose.inverse();
      // inliers should be a subset of the original refined inlier tracks
      refined_inlier_tracklets = flow_opt_result.inliers;

      VLOG(10) << "Refined object " << object_id
               << "pose with optical flow - error before: "
               << flow_opt_result.error_before.value_or(NaN)
               << " error_after: " << flow_opt_result.error_after.value_or(NaN);
    }
    // still need to take the inverse as we get the inverse of G out
    gtsam::Pose3 H_w = T_world_k * G_w;

    if (object_motion_params.refine_motion_with_3d) {
      VLOG(10) << "Refining object motion pose with 3D refinement";
      MotionOnlyRefinementOptimizer motion_refinement_graph(
          object_motion_params.object_motion_refinement_params);
      auto motion_refinement_result =
          motion_refinement_graph.optimizeAndUpdate<CalibrationType>(
              frame_k_1, frame_k, refined_inlier_tracklets, object_id, H_w);

      // should be further subset
      refined_inlier_tracklets = motion_refinement_result.inliers;
      H_w = motion_refinement_result.best_result;
    }

    motion_result.status = pose_result.status;
    motion_result.best_result = Motion3ReferenceFrame(
        H_w, Motion3ReferenceFrame::Style::F2F, ReferenceFrame::GLOBAL,
        frame_k_1->getFrameId(), frame_k->getFrameId());
    motion_result.inliers = refined_inlier_tracklets;
    determineOutlierIds(motion_result.inliers, all_tracklets,
                        motion_result.outliers);

    // sanity check that we have accounted for all initial matches
    CHECK_EQ(motion_result.inliers.size() + motion_result.outliers.size(),
             n_matches);
  }

  // needed when running things like TartanAir S where we get vew few point on
  // the object...
  // if (motion_result.inliers.size() < 30) {
  //   motion_result.status = TrackingStatus::FEW_MATCHES;
  // }

  return motion_result;
}

///////////////////////////////////////////////////////////////////////////

ObjectMotionSolverFilter::ObjectMotionSolverFilter(
    const ObjectMotionSolverFilter::Params& params,
    const CameraParams& camera_params)
    : EgoMotionSolver(static_cast<const EgoMotionSolver::Params&>(params),
                      camera_params),
      filter_params_(params) {}

MultiObjectTrajectories ObjectMotionSolverFilter::solve(Frame::Ptr frame_k,
                                                        Frame::Ptr frame_k_1) {
  // Handle lost objects: objects in filters_ but not in current frame's
  // object_observations_
  std::set<ObjectId> current_objects;
  for (const auto& [obj_id, _] : frame_k->object_observations_) {
    current_objects.insert(obj_id);
  }
  for (const auto& [obj_id, _] : filters_) {
    if (current_objects.find(obj_id) == current_objects.end()) {
      object_statuses_[obj_id] = ObjectTrackingStatus::Lost;
      filters_.erase(obj_id);
      LOG(INFO) << "Object " << obj_id << " marked as Lost at frame "
                << frame_k->getFrameId();
    }
    // TODO: no re-tracked logic!!

    object_keyframe_statuses_[obj_id] = ObjectKeyFrameStatus::NonKeyFrame;
  }

  // Call base solve
  return ObjectMotionSolver::solve(frame_k, frame_k_1);
}

bool ObjectMotionSolverFilter::solveImpl(Frame::Ptr frame_k,
                                         Frame::Ptr frame_k_1,
                                         ObjectId object_id,
                                         MotionEstimateMap& motion_estimates) {
  // Initialize or update tracking status
  bool is_new = !filters_.exists(object_id);
  bool is_resampled = std::find(frame_k->retracked_objects_.begin(),
                                frame_k->retracked_objects_.end(),
                                object_id) != frame_k->retracked_objects_.end();

  const ObjectTrackingStatus previous_tracking_state =
      object_statuses_[object_id];

  LOG(INFO) << "Previous tracking state " << to_string(previous_tracking_state)
            << " " << info_string(frame_k->getFrameId(), object_id);

  // get the corresponding feature pairs
  AbsolutePoseCorrespondences dynamic_correspondences;
  bool corr_result = frame_k->getDynamicCorrespondences(
      dynamic_correspondences, *frame_k_1, object_id,
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
      EgoMotionSolver::geometricOutlierRejection3d2d(dynamic_correspondences);

  const TrackletIds& inlier_tracklets = geometric_result.inliers;

  const size_t num_inliers =
      inlier_tracklets.size();  // after outlier rejection

  if (inlier_tracklets.size() < 4 ||
      geometric_result.status != TrackingStatus::VALID) {
    LOG(WARNING) << "Could not make initial frame for object " << object_id
                 << " as not enough inlier tracks!";
    object_statuses_[object_id] = ObjectTrackingStatus::PoorlyTracked;

    return false;
  }

  const gtsam::Pose3 G_w_inv_pnp = geometric_result.best_result.inverse();
  const gtsam::Pose3 H_w_km1_k_pnp = frame_k->getPose() * G_w_inv_pnp;

  auto should_object_KF = [&](ObjectId object_id) -> bool {
    bool temporal_kf = false;
    if (!is_new) {
      auto filter = filters_.at(object_id);
      FrameId last_kf_id = filter->getKeyFrameId();

      // TODO: right now force object KF to happen at some frequency for testing
      if (frame_k->getFrameId() > 5 &&
          frame_k->getFrameId() - last_kf_id > 5u) {
        VLOG(5) << "Long time since last KF for j=" << object_id;
        temporal_kf = true;
      }
    }
    // TODO: making temporal kf proves problems - show interaction between
    // "should reset" and not actually needing a reset is maybe problematic?

    // return is_resampled || temporal_kf;
    return is_resampled;
  };

  // includes the isa_resampled logic
  const bool should_kf = should_object_KF(object_id);

  if (is_new) {
    object_statuses_[object_id] = ObjectTrackingStatus::New;
    LOG(INFO) << "Object " << object_id << " initialized as New at frame "
              << frame_k->getFrameId();
    // dont update keyframe status to anchor yet - only do so if object
    // successfully created
  }
  // else if (is_resampled) {
  else if (should_kf) {
    // TODO: do we always want to RKF if resampled?
    // idea NO - depending on how many correspondences we had on the resampled
    // frame! ie if the tracking was actually still good (dont need to KF at
    // all!)
    // TODO: currently not right logic as inliers must always be > 4 to even get
    // here!!
    if (num_inliers > 4) {
      object_statuses_[object_id] = ObjectTrackingStatus::WellTracked;
      LOG(INFO) << "Object " << object_id
                << "resampled & WellTracked  at frame " << frame_k->getFrameId()
                << ", set to RegularKeyFrame";
      object_keyframe_statuses_[object_id] =
          ObjectKeyFrameStatus::RegularKeyFrame;
    } else {
      // TODO: this should actually be that all features are new!!!!!
      //  keep object as poorly tracked if was poorly tracked before
      object_statuses_[object_id] = ObjectTrackingStatus::WellTracked;
      object_keyframe_statuses_[object_id] =
          ObjectKeyFrameStatus::AnchorKeyFrame;
      LOG(INFO) << "Object " << object_id
                << "resampled & WellTracked  at frame " << frame_k->getFrameId()
                << ", set to AnchorKeyFrame";
    }
  } else {
    object_statuses_[object_id] = ObjectTrackingStatus::WellTracked;
  }

  bool new_or_reset_object = false;
  bool filter_needs_reset = false;
  // bool new_object = false;
  // bool object_reset = false;
  if (!is_new) {
    // should only get here if well tracked!
    // what is actual logic here -> needs reset separate to should KF in this
    // condition but somehow couplied in should_kf decision logic?
    if (filterNeedsReset(object_id)) {
      LOG(INFO) << object_id << " needs retting from last frame! current k="
                << frame_k->getFrameId();
      filter_needs_reset_[object_id] = false;

      auto filter = filters_.at(object_id);

      gtsam::Pose3 new_KF_pose;
      // ie. could not solve motion from the frame before
      if (previous_tracking_state == ObjectTrackingStatus::PoorlyTracked) {
        LOG(INFO) << "Object was poorly tracked previously. Creating new KF "
                     "from centroid "
                  << info_string(frame_k_1->getFrameId(), object_id);
        // must create new initial frame
        // TODO: here would be to check somehow that we dont have features
        // between the last well tracked state
        new_KF_pose = constructPoseFromCentroid(frame_k_1, inlier_tracklets);
        // alert to new KF that has connection with previous KF
        object_keyframe_statuses_[object_id] =
            ObjectKeyFrameStatus::AnchorKeyFrame;

        // TODO: if only one frame dropped maybe use constant motion model?
      }
      // in this case 'New' also means well tracked since it can only be set if
      // PnP was good!
      else if (previous_tracking_state == ObjectTrackingStatus::WellTracked ||
               previous_tracking_state == ObjectTrackingStatus::New) {
        // if well tracked than the previous update should be from the immediate
        // previous frame!
        CHECK_EQ(filter->getFrameId(), frame_k_1->getFrameId())
            << "j=" << object_id << " k=" << filter->getFrameId();
        // this is the pose as the last frame (ie k-1) which will serve as the
        // new keyframe pose
        new_KF_pose = filter->getPose();
      } else {
        LOG(FATAL) << "Should not get here! Previous track state "
                   << to_string(previous_tracking_state)
                   << " Curent track state "
                   << to_string(object_statuses_[object_id])
                   << " j=" << object_id;
      }

      // NOTE: from motion at k-1
      filter->resetState(new_KF_pose, frame_k_1->getFrameId());
      new_or_reset_object = true;
      // stable_frame_counts_[object_id] = 0;
      LOG(INFO) << "Object " << object_id << " reset, stable count reset to 0";
    }

    // if (is_resampled) {
    if (should_kf) {
      LOG(INFO) << object_id
                << " retracked - resetting filter k=" << frame_k->getFrameId();

      // actually indicates that we will have a KF motion (as long as tracking
      // was good???) in the next frame
      filter_needs_reset_[object_id] = true;
      LOG(INFO) << "Object " << object_id << " resampled, marked for reset";
    }
  }
  // becuuse we erase it if object new in previous!
  // not sure this is the best way to handle reappearing objects!
  if (is_new) {
    new_or_reset_object = true;
    createAndInsertFilter(object_id, frame_k_1, inlier_tracklets);
  }

  auto filter = filters_.at(object_id);
  // update and predict should be one step so that if we dont have enough points
  // NOTE: this logic seemed pretty important to ensure the estimate was good!!!
  // we dont predict?
  if (new_or_reset_object) {
    filter->predictAndUpdate(gtsam::Pose3::Identity(), frame_k_1,
                             inlier_tracklets, 2);
  }

  filter->predictAndUpdate(H_w_km1_k_pnp, frame_k, inlier_tracklets, 2);

  bool return_result = false;
  if (geometric_result.status == TrackingStatus::VALID) {
    const gtsam::Pose3 H_w_km1_k = filter->getF2FMotion();

    Motion3SolverResult motion_result;
    motion_result.status = geometric_result.status;
    motion_result.inliers = geometric_result.inliers;
    motion_result.outliers = geometric_result.outliers;

    motion_result.best_result = Motion3ReferenceFrame(
        H_w_km1_k, Motion3ReferenceFrame::Style::F2F, ReferenceFrame::GLOBAL,
        frame_k_1->getFrameId(), frame_k->getFrameId());

    // TODO: make thread safe!
    frame_k->dynamic_features_.markOutliers(motion_result.outliers);
    motion_estimates.insert({object_id, motion_result.best_result});

    return_result = true;
  } else {
    // TODO: not sure lost is the right logic here!
    object_statuses_[object_id] = ObjectTrackingStatus::Lost;
    // stable_frame_counts_[object_id] = 0;
    LOG(INFO) << "Object " << object_id << " set to Lost at frame "
              << frame_k->getFrameId();
    // so tha
    filters_.erase(object_id);
    return_result = false;
  }

  LOG(INFO) << "Object " << object_id
            << " final status: " << to_string(object_statuses_[object_id])
            << " with keyframe status: "
            << to_string(object_keyframe_statuses_[object_id]);
  return return_result;
}

bool ObjectMotionSolverFilter::filterNeedsReset(ObjectId object_id) {
  if (filter_needs_reset_.exists(object_id)) {
    return filter_needs_reset_.at(object_id);
  }
  return false;
}

gtsam::Pose3 ObjectMotionSolverFilter::constructPoseFromCentroid(
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
ObjectMotionSolverFilter::createAndInsertFilter(ObjectId object_id,
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

void ObjectMotionSolverFilter::updateTrajectories(
    MultiObjectTrajectories& object_trajectories,
    const MotionEstimateMap& motion_estimates, Frame::Ptr frame_k,
    Frame::Ptr frame_k_1) {
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
