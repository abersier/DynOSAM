#include "dynosam/backend/Formulation.hpp"

namespace dyno {

void logFromAccessor(Accessor::Ptr accessor, BackendLogger& logger, const std::optional<GroundTruthPacketMap>& ground_truth) {
  CHECK(accessor);
  const PoseTrajectory& camera_trajectory = accessor->getCameraTrajectory();
  logger.logCameraPose(camera_trajectory, ground_truth);

  const MultiObjectTrajectories& object_trajectories =
      accessor->getMultiObjectTrajectories();
  logger.logObjectTrajectory(object_trajectories, ground_truth);

  auto static_map = accessor->getFullStaticMap();
  auto dynamic_map = accessor->getFullTemporalDynamicMap();

  logger.logMapPoints(static_map);
  logger.logMapPoints(dynamic_map);
}

// FormulationBase::FormulationBase(const FormulationParams& params,
//                                  const NoiseModels& noise_models,
//                                  const Sensors& sensors,
//                                  const FormulationHooks& hooks)
//     : params_(params),
//       noise_models_(noise_models),
//       sensors_(sensors),
//       hooks_(hooks) {}

// void FormulationBase::setTheta(const gtsam::Values& linearization) {
//   theta_ = linearization;
// }

// void FormulationBase::updateTheta(const gtsam::Values& linearization) {
//   theta_.insert_or_assign(linearization);
// }

// BackendLogger::UniquePtr FormulationBase::makeFullyQualifiedLogger() const {
//   return std::make_unique<BackendLogger>(getFullyQualifiedName());
// }

// Accessor::Ptr FormulationBase::accessorFromTheta() const {
//   if (!accessor_theta_) {
//     SharedFormulationData shared_data(&theta_, &hooks_);
//     accessor_theta_ = createAccessor(shared_data);
//   }
//   return accessor_theta_;
// }

// std::string FormulationBase::setFullyQualifiedName() const {
//   // get the derived name of the formulation
//   std::string logger_prefix = this->loggerPrefix();
//   const std::string suffix = params_.updater_suffix;

//   // add suffix to name if required
//   if (!suffix.empty()) {
//     logger_prefix += ("_" + suffix);
//   }
//   fully_qualified_name_ = logger_prefix;
//   return *fully_qualified_name_;
// }

}  // namespace dyno
