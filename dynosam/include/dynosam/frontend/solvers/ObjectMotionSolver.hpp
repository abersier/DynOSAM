#pragma once

#include "dynosam/frontend/solvers/PnPRansac.hpp"
#include "dynosam/frontend/vision/Frame.hpp"
#include "dynosam_common/Types.hpp"

namespace dyno {

class ObjectMotionSolver {
 public:
  ObjectMotionSolver() = default;
  virtual ~ObjectMotionSolver() = default;

  /**
   * @brief Solve for the per frame motion of each object observed in k-1 to k.
   *
   * Internally calls either ObjectMotionSolver#parallelSolve or
   * ObjectMotionSolver#sequentialSolve depending on the parallel_solve flag
   * which then used the virtual solveImpl to solve for the motion of each
   * object.
   *
   * @param frame_k
   * @param frame_k_1
   * @param parallel_solve
   * @return MultiObjectTrajectories
   */
  virtual void solve(Frame::Ptr frame_k, Frame::Ptr frame_k_1,
                     MultiObjectTrajectories& trajectories_out,
                     MotionEstimateMap& motion_estimate_out,
                     bool parallel_solve = true);

  void solve(Frame::Ptr frame_k, Frame::Ptr frame_k_1,
             MultiObjectTrajectories& trajectories_out,
             bool parallel_solve = true);

 protected:
  void parallelSolve(Frame::Ptr frame_k, Frame::Ptr frame_km1,
                     MotionEstimateMap& motion_estimates,
                     ObjectIds& failed_object_tracks);
  void sequentialSolve(Frame::Ptr frame_k, Frame::Ptr frame_km1,
                       MotionEstimateMap& motion_estimates,
                       ObjectIds& failed_object_tracks);

  virtual bool solveImpl(Frame::Ptr frame_k, Frame::Ptr frame_k_1,
                         ObjectId object_id,
                         Motion3ReferenceFrame& motion_estimate) = 0;

  virtual void updateTrajectories(MultiObjectTrajectories& object_trajectories,
                                  const MotionEstimateMap& motion_estimates,
                                  Frame::Ptr frame_k, Frame::Ptr frame_k_1) = 0;
};

}  // namespace dyno
