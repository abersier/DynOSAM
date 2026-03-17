#pragma once

#include "dynosam/backend/BackendModule.hpp"
#include "dynosam/visualizer/VisualizerPipelines.hpp"  //for BackendModuleDisplay
#include "dynosam_opt/IncrementalOptimization.hpp"     // for ErrorHandlingHooks

namespace dyno {
/**
 * @brief Wrapper containing the BackendModule, any associated additional
 * visualizer and custom error hooks associated with this backend and (likely)
 * its formulation. Created by the BackendModuleFactory which is the only thing
 * that knows all the details of the particular backend and the formulation
 * used.
 *
 */
struct BackendWrapper {
  Backend::Ptr backend;
  BackendModuleDisplay::Ptr backend_viz;
};

/**
 * @brief Params needed to make a module
 *
 */
struct ModuleParams {
  BackendParams backend_params;
  Sensors sensors;
  SharedGroundTruth shared_ground_truth{};
};

class BackendModuleFactory {
 public:
  DYNO_POINTER_TYPEDEFS(BackendModuleFactory)

  virtual BackendWrapper createModule(const ModuleParams& params) = 0;
};

}  // namespace dyno
