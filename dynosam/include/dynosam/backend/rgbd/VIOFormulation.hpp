#pragma once

#include "dynosam/backend/Accessor.hpp"
#include "dynosam/backend/Formulation.hpp"
#include "dynosam_opt/Map.hpp"

namespace dyno {

class VIOAccessor : public Accessor {
 public:
  DYNO_POINTER_TYPEDEFS(VIOAccessor)

  VIOAccessor() {}
  virtual ~VIOAccessor() {}

  // in these cases we rely mostly on the derived accessor functions for look up
  // these cases should have access to the map rather than just doing a raw
  // query with the key
  StateQuery<gtsam::NavState> getNavState(FrameId frame_id) const;
  StateQuery<gtsam::imuBias::ConstantBias> getImuBias(FrameId frame_id) const;
};

// if derive - also ensure that the DerivedAccessor derives from VIOAccessor
class VIOFormulation : public Formulation<MapVision> {
 public:
  using Base = Formulation<MapVision>;
  using Base::AccessorTypePointer;
  using Base::MapTraitsType;
  using Base::ObjectUpdateContextType;
  using Base::PointUpdateContextType;

  DYNO_POINTER_TYPEDEFS(VIOFormulation)

  VIOFormulation(const FormulationParams& params, typename Map::Ptr map,
                 const NoiseModels& noise_models, const Sensors& sensors,
                 const FormulationHooks& hooks)
      : Base(params, map, noise_models, sensors, hooks) {}
  virtual ~VIOFormulation() {}

  // NOTE: V_C_k should actually be V_W_k as nav state and imu-pim operates in
  // W!!
  gtsam::NavState addStatesInitalise(
      gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors,
      FrameId frame_id_k, Timestamp timestamp_k, const gtsam::Pose3& X_W_k,
      const gtsam::Vector3& V_W_k = gtsam::Vector3(0, 0, 0));

  // Eventually should encpsualte propogate in EgoMotion struct which contains
  // both T_km1_k and imu-preintegration
  gtsam::NavState addStatesPropogate(gtsam::Values& new_values,
                                     gtsam::NonlinearFactorGraph& new_factors,
                                     FrameId frame_id_k, Timestamp timestamp_k,
                                     const gtsam::Pose3& T_k_1_k,
                                     const ImuFrontend::PimPtr& pim = nullptr);

  void addSensorPosePrior(gtsam::NonlinearFactorGraph& new_factors,
                          FrameId frame_id_k, const gtsam::Pose3& X_W_k,
                          gtsam::SharedNoiseModel noise_model);

  void addSensorPose(gtsam::Values& new_values, FrameId frame_id_k,
                     const gtsam::Pose3& X_W_k);

  VIOAccessor::Ptr getAsVIOAccessor() const;

  bool isImuInitalized() const { return imu_states_initalise_; };

  StateQuery<gtsam::NavState> getNavState(FrameId frame_id);

 private:
  gtsam::NavState predictAndAddFactorsVO(
      gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors,
      FrameId frame_id_k, Timestamp timestamp_k, const gtsam::Pose3& T_k_1_k);

  gtsam::NavState predictAndAddFactorsIMU(
      gtsam::Values& new_values, gtsam::NonlinearFactorGraph& new_factors,
      FrameId frame_id_k, Timestamp timestamp_k,
      const ImuFrontend::PimPtr& pim);

 private:
  bool imu_states_initalise_{false};
  FrameId first_frame_{0};
  FrameId last_propogate_frame_{0};
  Timestamp last_propogate_time_{0.0};
};

}  // namespace dyno
