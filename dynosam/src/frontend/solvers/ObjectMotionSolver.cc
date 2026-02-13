#include "dynosam/frontend/solvers/ObjectMotionSolver.hpp"

#include <tbb/tbb.h>

#include "dynosam_common/utils/TimingStats.hpp"

namespace dyno {

MultiObjectTrajectories ObjectMotionSolver::solve(Frame::Ptr frame_k,
                                                  Frame::Ptr frame_k_1,
                                                  bool parallel_solve) {
  ObjectIds failed_object_tracks;
  MotionEstimateMap motion_estimates;

  if (parallel_solve) {
    parallelSolve(frame_k, frame_k_1, motion_estimates, failed_object_tracks);
  } else {
    sequentialSolve(frame_k, frame_k_1, motion_estimates, failed_object_tracks);
  }

  // remove objects from the object observations list
  // does not remove the features etc but stops the object being propogated to
  // the backend as we loop over the object observations in the constructOutput
  // function
  for (auto object_id : failed_object_tracks) {
    frame_k->object_observations_.erase(object_id);
  }

  MultiObjectTrajectories object_trajectories;
  updateTrajectories(object_trajectories, motion_estimates, frame_k, frame_k_1);

  return object_trajectories;
}

void ObjectMotionSolver::parallelSolve(Frame::Ptr frame_k, Frame::Ptr frame_km1,
                                       MotionEstimateMap& motion_estimates,
                                       ObjectIds& failed_object_tracks) {
  const auto& object_observations = frame_k->object_observations_;
  const auto num_objects = object_observations.size();

  // Initalise as false
  std::vector<bool> object_success(num_objects, false);
  // Preallocate vector where the object result will go in
  // result is only valid if corresponding index in object_success is true
  std::vector<MotionEstimateMap::mapped_type> object_result(num_objects);

  ObjectIds objects_ids;
  objects_ids.reserve(num_objects);
  for (const auto& [object_id, _] : object_observations) {
    objects_ids.push_back(object_id);
  }

  tbb::parallel_for(size_t(0), num_objects, [&](size_t i) {
    const auto object_id = objects_ids[i];
    // result to fill
    auto& estimate = object_result.at(i);
    object_success[i] = solveImpl(frame_k, frame_km1, object_id, estimate);
  });

  // post process and fill motion estimates and failed object tracks
  for (size_t i = 0; i < num_objects; i++) {
    const auto& object_id = objects_ids.at(i);
    if (object_success.at(i)) {
      motion_estimates.insert2(object_id, object_result.at(i));
    } else {
      failed_object_tracks.push_back(object_id);
    }
  }
}

void ObjectMotionSolver::sequentialSolve(Frame::Ptr frame_k,
                                         Frame::Ptr frame_km1,
                                         MotionEstimateMap& motion_estimates,
                                         ObjectIds& failed_object_tracks) {
  const auto& object_observations = frame_k->object_observations_;
  for (const auto& [object_id, _] : object_observations) {
    Motion3ReferenceFrame motion_estimate;
    if (solveImpl(frame_k, frame_km1, object_id, motion_estimate)) {
      motion_estimates.insert2(object_id, motion_estimate);
    } else {
      failed_object_tracks.push_back(object_id);
    }
  }
}

}  // namespace dyno
