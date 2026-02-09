#pragma once

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/backend/BackendInputPacket.hpp"
#include "dynosam/backend/BackendModule.hpp"
#include "dynosam/backend/Formulation.hpp"
#include "dynosam/backend/RegularBackendDefinitions.hpp"
#include "dynosam/backend/rgbd/VIOFormulation.hpp"

namespace dyno {

struct PoseChangeInput {
  DYNO_POINTER_TYPEDEFS(PoseChangeInput)
};

class PoseChangeVIBackendModule
    : public BackendModuleV1T<MapVision, PoseChangeInput> {
 public:
  // TODO: factory
  using Base = BackendModuleV1T<MapVision, PoseChangeInput>;
  DYNO_POINTER_TYPEDEFS(PoseChangeVIBackendModule)

  std::pair<gtsam::Values, gtsam::NonlinearFactorGraph> getActiveOptimisation()
      const override;

  Accessor::Ptr getAccessor() const override;

 protected:
  using SpinReturn = Base::SpinReturn;

  // just call spin once
  SpinReturn boostrapSpin(PoseChangeInput::ConstPtr input) override;
  SpinReturn nominalSpin(PoseChangeInput::ConstPtr input) override;

  void spinOnce(PoseChangeInput::ConstPtr input);

  VIOFormulation::Ptr formulation_;
  // Formulation display
  // BackendModuleDisplay::Ptr formulation_display_;
};

}  // namespace dyno
