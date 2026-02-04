#pragma once

#include "dynosam/frontend/FrontendInputPacket.hpp"
#include "dynosam/frontend/RGBDInstance-Definitions.hpp"
#include "dynosam/frontend/imu/ImuFrontend.hpp"
#include "dynosam/frontend/vision/FeatureTracker.hpp"
#include "dynosam/pipeline/PipelineParams.hpp"
#include "dynosam_common/ModuleBase.hpp"

namespace dyno {

// TODO: Original is in FrontendModule which can be deleted once refactoring
struct InvalidImageContainerException1 : public DynosamException {
  InvalidImageContainerException1(const ImageContainer& container,
                                  const std::string& what)
      : DynosamException("Image container with config: " +
                         container.toString() + "\n was invalid - " + what) {}
};

// For now
struct DynoState {
  DYNO_POINTER_TYPEDEFS(DynoState)
};

class Frontend : public ModuleBase<FrontendInputPacketBase, DynoState> {
 public:
  using Base = ModuleBase<FrontendInputPacketBase, DynoState>;

  DYNO_POINTER_TYPEDEFS(Frontend)
  Frontend(const std::string& name, const DynoParams& params,
           ImageDisplayQueue* display_queue);
  virtual ~Frontend() = default;

 protected:
  bool pushImageToDisplayQueue(const std::string& wname, const cv::Mat& image);

 protected:
  virtual void validateInput(
      const FrontendInputPacketBase::ConstPtr& input) const;

  // TODO: log state

  const DynoParams dyno_params_;
  ImageDisplayQueue* display_queue_;
  RGBDFrontendLogger::UniquePtr logger_;
};

class VIFrontend : public Frontend {
 public:
  DYNO_POINTER_TYPEDEFS(VIFrontend)
  VIFrontend(const std::string& name, const DynoParams& params,
             Camera::Ptr camera, ImageDisplayQueue* display_queue);
  virtual ~VIFrontend() = default;

 protected:
  Camera::Ptr camera_;
  EgoMotionSolver motion_solver_;
  //     // TODO: shared pointer for now during debig phase!
  //     ObjectMotionSolver::Ptr object_motion_solver_;
  FeatureTracker::UniquePtr tracker_;
};

// TODO: include sinks (callback) to backend with VisionImuOutput
class RegularVIFrontend : public VIFrontend {
 public:
  DYNO_POINTER_TYPEDEFS(RegularVIFrontend)
  RegularVIFrontend(const DynoParams& params, Camera::Ptr camera,
                    ImageDisplayQueue* display_queue = nullptr);

 private:
  SpinReturn boostrapSpin(FrontendInputPacketBase::ConstPtr input) override;
  SpinReturn nominalSpin(FrontendInputPacketBase::ConstPtr input) override;
};

class PoseChangeVIFrontend : public VIFrontend {
 public:
  DYNO_POINTER_TYPEDEFS(PoseChangeVIFrontend)
  PoseChangeVIFrontend(const DynoParams& params, Camera::Ptr camera,
                       ImageDisplayQueue* display_queue = nullptr);

 private:
  SpinReturn boostrapSpin(FrontendInputPacketBase::ConstPtr input) override;
  SpinReturn nominalSpin(FrontendInputPacketBase::ConstPtr input) override;
};

}  // namespace dyno
