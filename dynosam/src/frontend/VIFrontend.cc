#include "dynosam/frontend/VIFrontend.hpp"

namespace dyno {

Frontend::Frontend(const std::string& name, const DynoParams& params,
                   ImageDisplayQueue* display_queue)
    : Base(name), dyno_params_(params), display_queue_(display_queue) {
  // TODO: logger!!
}

bool Frontend::pushImageToDisplayQueue(const std::string& wname,
                                       const cv::Mat& image) {
  if (!display_queue_) {
    return false;
  }

  display_queue_->push(ImageToDisplay(wname, image));
  return true;
}

void Frontend::validateInput(
    const FrontendInputPacketBase::ConstPtr& input) const {
  const auto image_container = input->image_container_;

  if (!image_container) {
    throw DynosamException("Image container is null!");
  }

  const bool has_rgb = image_container->hasRgb();
  const bool has_depth_image = image_container->hasDepth();
  const bool has_stereo = image_container->hasRightRgb();

  if (!has_rgb) {
    throw InvalidImageContainerException1(*image_container, "Missing RGB");
  }

  if (!has_depth_image && !has_stereo) {
    throw InvalidImageContainerException1(*image_container,
                                          "Missing Depth or Stereo");
  }
}

VIFrontend::VIFrontend(const std::string& name, const DynoParams& params,
                       Camera::Ptr camera, ImageDisplayQueue* display_queue)
    : Frontend(name, params, display_queue), camera_(CHECK_NOTNULL(camera)) {
  const auto& frontend_params = dyno_params_.frontend_params_;
  tracker_ =
      std::make_unique<FeatureTracker>(frontend_params, camera_, display_queue);
}

RegularVIFrontend::RegularVIFrontend(const DynoParams& params,
                                     Camera::Ptr camera,
                                     ImageDisplayQueue* display_queue)
    : VIFrontend("regular-frontend", params, camera, display_queue) {}

RegularVIFrontend::SpinReturn RegularVIFrontend::boostrapSpin(
    FrontendInputPacketBase::ConstPtr input) {}

RegularVIFrontend::SpinReturn RegularVIFrontend::nominalSpin(
    FrontendInputPacketBase::ConstPtr input) {}

PoseChangeVIFrontend::PoseChangeVIFrontend(const DynoParams& params,
                                           Camera::Ptr camera,
                                           ImageDisplayQueue* display_queue)
    : VIFrontend("pc-frontend", params, camera, display_queue) {}

PoseChangeVIFrontend::SpinReturn PoseChangeVIFrontend::boostrapSpin(
    FrontendInputPacketBase::ConstPtr input) {}

PoseChangeVIFrontend::SpinReturn PoseChangeVIFrontend::nominalSpin(
    FrontendInputPacketBase::ConstPtr input) {}

}  // namespace dyno
