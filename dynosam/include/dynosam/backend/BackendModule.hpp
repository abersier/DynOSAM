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

#pragma once

#include "dynosam/backend/BackendDefinitions.hpp"
#include "dynosam/backend/BackendInputPacket.hpp"
#include "dynosam/backend/BackendOutputPacket.hpp"
#include "dynosam/backend/BackendParams.hpp"
#include "dynosam/backend/Formulation.hpp"
#include "dynosam/visualizer/Visualizer-Definitions.hpp"  //for ImageDisplayQueueOptional,
#include "dynosam_common/DynoState.hpp"
#include "dynosam_common/Exceptions.hpp"
#include "dynosam_common/ModuleBase.hpp"
#include "dynosam_common/SharedModuleInfo.hpp"
#include "dynosam_common/Types.hpp"
#include "dynosam_common/utils/SafeCast.hpp"
#include "dynosam_opt/Map.hpp"

// DECLARE_string(updater_suffix);

namespace dyno {

template <typename DERIVED_INPUT_PACKET, typename MEASUREMENT_TYPE,
          typename BASE_INPUT_PACKET = BackendInputPacket>
struct BackendModuleTraits {
  using DerivedPacketType = DERIVED_INPUT_PACKET;
  using DerivedPacketTypeConstPtr = std::shared_ptr<const DerivedPacketType>;

  using BasePacketType = BASE_INPUT_PACKET;
  // BasePacketType is the type that gets passed to the module via the pipeline
  // and must be a base class since we pass data along the pipelines via
  // poniters
  static_assert(std::is_base_of_v<BasePacketType, DerivedPacketType>);

  using MeasurementType = MEASUREMENT_TYPE;
  using MapType = Map<MeasurementType>;
};

using FrontendUpdateInterface =
    std::function<void(const FrameId, const Timestamp)>;

class Backend {
 public:
  DYNO_POINTER_TYPEDEFS(Backend)

  Backend() {}
  virtual ~Backend() {}
};

// TODO: BackendOutput should become State
template <typename INPUT>
class BackendModuleV1 : public ModuleBase<INPUT, DynoState>, public Backend {
 public:
  using Base = ModuleBase<INPUT, DynoState>;
  using Base::SpinReturn;

  BackendModuleV1(const BackendParams& params, Camera::Ptr camera)
      : Base("backend"),
        backend_params_(params),
        camera_(CHECK_NOTNULL(camera)),
        noise_models_(NoiseModels::fromBackendParams(params)) {}
  virtual ~BackendModuleV1() = default;

  // something about callback here too?
  // and get noise models!
  const BackendParams& getParams() const { return backend_params_; }
  const NoiseModels& getNoiseModels() const { return noise_models_; }

  /**
   * @brief Get the accessor the the underlying formulation, allowing the
   * optimised values to be directly accessed
   *
   * @return Accessor::Ptr
   */
  virtual Accessor::Ptr getAccessor() const = 0;
  virtual std::pair<gtsam::Values, gtsam::NonlinearFactorGraph>
  getActiveOptimisation() const = 0;

  virtual DynoState::Ptr makeOutput() const {
    Accessor::Ptr accessor = this->getAccessor();

    DynoState::Ptr state = std::make_shared<DynoState>();

    const auto camera_trajectory = accessor->getCameraTrajectory();
    // expect frame and timestamp to be from the last entry
    const auto last_camera_entry = camera_trajectory.last();
    state->frame_id = last_camera_entry.frame_id;
    state->timestamp = last_camera_entry.timestamp;

    state->camera_trajectory = camera_trajectory;
    state->object_trajectories = accessor->getMultiObjectTrajectories();

    // TODO: should be global!?
    state->local_static_map = accessor->getFullStaticMap();
    state->dynamic_map = accessor->getDynamicLandmarkEstimates(state->frame_id);

    return state;
  }

 private:
  // called in ModuleBase immediately before the spin function is called
  virtual inline void validateInput(
      const typename Base::InputConstPtr& input) const override {}

 protected:
  // TODO: maybe put in Backend
  const BackendParams backend_params_;
  Camera::Ptr camera_;
  const NoiseModels noise_models_;
};

// For modules that have a single internal map
// ie. not PH
template <typename MAP, typename INPUT>
class BackendModuleV1T : public BackendModuleV1<INPUT> {
 public:
  using MapT = MAP;
  using Base = BackendModuleV1<INPUT>;
  using FormulationT = Formulation<MapT>;

  BackendModuleV1T(const BackendParams& params, Camera::Ptr camera)
      : Base(params, camera), map_(MapT::create()) {}
  virtual ~BackendModuleV1T() = default;

  const typename MapT::Ptr map() { return map_; }

 protected:
  typename MapT::Ptr map_;
};

/**
 * @brief Base class to actually do processing. Data passed to this module from
 * the frontend
 *
 */
class BackendModule
    : public ModuleBase<BackendInputPacket, BackendOutputPacket>,
      public SharedModuleInterface {
 public:
  DYNO_POINTER_TYPEDEFS(BackendModule)

  using Base = ModuleBase<BackendInputPacket, BackendOutputPacket>;
  using Base::SpinReturn;

  BackendModule(const BackendParams& params, ImageDisplayQueue* display_queue);
  virtual ~BackendModule() = default;

  const BackendParams& getParams() const { return base_params_; }
  const NoiseModels& getNoiseModels() const { return noise_models_; }
  const BackendSpinState& getSpinState() const { return spin_state_; }

  /**
   * @brief Get the accessor the the underlying formulation, allowing the
   * optimised values to be directly accessed
   *
   * @return Accessor::Ptr
   */
  virtual Accessor::Ptr getAccessor() = 0;

  void registerFrontendUpdateInterface(const FrontendUpdateInterface& cb) {
    CHECK(cb);
    frontend_update_callback_ = cb;
  }

 protected:
  // called in ModuleBase immediately before the spin function is called
  virtual void validateInput(
      const BackendInputPacket::ConstPtr& input) const override;
  void setFactorParams(const BackendParams& backend_params);

 protected:
  // Redefine base input since these will be cast up by the BackendModuleType
  // class to a new type which we want to refer to as the input type BaseInput
  // is a ConstPtr to the type defined by BackendInputPacket
  using BaseInputConstPtr = Base::InputConstPtr;
  using BaseInput = Base::Input;

 protected:
  const BackendParams base_params_;
  ImageDisplayQueue* display_queue_{nullptr};  //! Optional display queue

  //! Spin state of the backend. Updated in the backend module
  //! base via InputCallback (see BackendModule constructor).
  BackendSpinState spin_state_;

  NoiseModels noise_models_;
  FrontendUpdateInterface frontend_update_callback_;

 private:
  //! Maps which iteration of the backend corresponds with a frame id
  //! Used primarily to handle KF's as the BackendSpinState#iterations
  //! are used as an analog for KF ids
  gtsam::FastMap<int, FrameId> iteration_to_frame_id_;
};

template <class MODULE_TRAITS>
class BackendModuleType : public BackendModule {
 public:
  using ModuleTraits = MODULE_TRAITS;
  // A Dervied BackedInputPacket type (e.g. RGBDOutputPacketType)
  using DerivedPacketType = typename ModuleTraits::DerivedPacketType;
  using DerivedPacketTypeConstPtr =
      typename ModuleTraits::DerivedPacketTypeConstPtr;
  using MeasurementType = typename ModuleTraits::MeasurementType;
  using This = BackendModuleType<ModuleTraits>;
  using Base = BackendModule;

  using MapType = typename ModuleTraits::MapType;
  using FormulationType = Formulation<MapType>;

  DYNO_POINTER_TYPEDEFS(This)

  using Base::SpinReturn;
  // Define the input type to the derived input type, defined in the
  // MODULE_TRAITS this is the derived Input packet that is passed to the
  // boostrap/nominal Spin Impl functions that must be implemented in the
  // derived class that does the provessing on this module
  using InputConstPtr = DerivedPacketTypeConstPtr;
  using OutputConstPtr = Base::OutputConstPtr;

  BackendModuleType(const BackendParams& params,
                    ImageDisplayQueue* display_queue)
      : Base(params, display_queue), map_(MapType::create()) {}

  virtual ~BackendModuleType() {}

  inline const typename MapType::Ptr getMap() { return map_; }

  virtual std::pair<gtsam::Values, gtsam::NonlinearFactorGraph>
  getActiveOptimisation() const = 0;

 protected:
  virtual SpinReturn boostrapSpinImpl(InputConstPtr input) = 0;
  virtual SpinReturn nominalSpinImpl(InputConstPtr input) = 0;

  typename MapType::Ptr map_;

 private:
  SpinReturn boostrapSpin(Base::BaseInputConstPtr base_input) override {
    return boostrapSpinImpl(attemptCast(base_input));
  }

  SpinReturn nominalSpin(Base::BaseInputConstPtr base_input) override {
    return nominalSpinImpl(attemptCast(base_input));
  }

  DerivedPacketTypeConstPtr attemptCast(Base::BaseInputConstPtr base_input) {
    DerivedPacketTypeConstPtr deriverd_input =
        safeCast<Base::BaseInput, DerivedPacketType>(base_input);
    checkAndThrow((bool)deriverd_input,
                  "Failed to cast " + type_name<Base::BaseInput>() + " to " +
                      type_name<DerivedPacketType>() + " in BackendModuleType");
    return deriverd_input;
  }
};

}  // namespace dyno
