/*
 *   Copyright (c) 2025 ACFR-RPG, University of Sydney, Jesse Morris
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

#include "dynosam_ros/displays/dynamic_slam_displays/FrontendDSDRos.hpp"

#include <pcl_conversions/pcl_conversions.h>

#include "cv_bridge/cv_bridge.h"
#include "dynosam_ros/RosUtils.hpp"
#include "rclcpp/qos.hpp"

namespace dyno {

FrontendDSDRos::FrontendDSDRos(const DisplayParams& params,
                               rclcpp::Node::SharedPtr node,
                               rclcpp::Node::SharedPtr ground_truth_node)
    : FrontendDisplay(), dyno_state_publisher_(params, node),
      camera_frame_id_(params.camera_frame_id) {
  tracking_image_pub_ =
      image_transport::create_publisher(node.get(), "tracking_image");

  // const rclcpp::SensorDataQoS sensor_qos;
  // NOTE: Use RELIABLE QoS (not SensorDataQoS/BEST_EFFORT) so that
  // DynORecon's message_filters subscriber can receive this cloud.
  dense_dynamic_cloud_pub_ =
      node->create_publisher<sensor_msgs::msg::PointCloud2>(
          "dense_labelled_cloud", rclcpp::QoS(10));

  background_stride_ = std::max<int>(1, node->declare_parameter<int>("background_stride", 4));

  if (ground_truth_node) {
    RCLCPP_INFO_STREAM(node->get_logger(), "Creating ground truth publishers");
    ground_truth_publishers_.emplace(params, ground_truth_node);
  }
}

FrontendDSDRos::GroundTruthPublishers::GroundTruthPublishers(
    const DisplayParams& params, rclcpp::Node::SharedPtr ground_truth_node)
    : dyno_state_publisher_(params, CHECK_NOTNULL(ground_truth_node)) {}

void FrontendDSDRos::spinOnce(const RealtimeOutput::ConstPtr& frontend_output) {
  VLOG(20) << "Spinning FrontendDSDRos k=" << frontend_output->state.frame_id;
  dyno_state_publisher_.publish(frontend_output->state);

  // publish debug imagery
  tryPublishDebugImagery(frontend_output);

  // // // publish ground truth
  tryPublishGroundTruth(frontend_output);

  // NOTE: Publish dense labelled cloud for DynORecon (dyno_mpc). Cloud type is
  // PointXYZRGBL; DynORecon reads via pcl::fromROSMsg into PointXYZL, matching
  // fields by name — the extra rgb field is ignored.
  if (frontend_output->dense_labelled_cloud) {
    const auto& cloud = frontend_output->dense_labelled_cloud;
    pcl::PointCloud<pcl::PointXYZRGBL> filtered;
    filtered.reserve(cloud->size());
    int bg_counter = 0;
    for (const auto& p : *cloud) {
      if (p.x == 0.0f && p.y == 0.0f && p.z == 0.0f) continue;
      if (p.label == 0 && bg_counter++ % background_stride_ != 0) continue;
      filtered.push_back(p);
    }
    filtered.is_dense = true;

    sensor_msgs::msg::PointCloud2 cloud_msg;
    pcl::toROSMsg(filtered, cloud_msg);
    cloud_msg.header.stamp = utils::toRosTime(frontend_output->state.timestamp);
    cloud_msg.header.frame_id = camera_frame_id_;
    dense_dynamic_cloud_pub_->publish(cloud_msg);
  }
}

void FrontendDSDRos::tryPublishDebugImagery(
    const RealtimeOutput::ConstPtr& frontend_output) {
  const DebugImagery& debug_imagery = frontend_output->debug_imagery;
  if (debug_imagery.tracking_image.empty()) return;

  std_msgs::msg::Header hdr;
  sensor_msgs::msg::Image::SharedPtr msg =
      cv_bridge::CvImage(hdr, "bgr8", debug_imagery.tracking_image)
          .toImageMsg();
  tracking_image_pub_.publish(msg);
}

void FrontendDSDRos::tryPublishGroundTruth(
    const RealtimeOutput::ConstPtr& frontend_output) {
  // for historical and structural reasons we expect the ground truth packet
  // to be provided in the frontend output for cisualisation
  if (!ground_truth_publishers_ || !frontend_output->ground_truth) return;

  DynoState& ground_truth_dyno_state =
      ground_truth_publishers_->ground_truth_state_;

  const GroundTruthInputPacket& gt_packet =
      frontend_output->ground_truth.value();

  const auto frame_id = gt_packet.frame_id_;
  const auto timestamp = gt_packet.timestamp_;

  // must update frame id and timestamp on state otherwise publishing wont work
  // as it uses the frame id to build the object odometries for this frame!
  ground_truth_dyno_state.frame_id = frame_id;
  ground_truth_dyno_state.timestamp = timestamp;

  for (const auto& object_pose_gt : gt_packet.object_poses_) {
    // check we have a gt motion here
    // in the case that we dont, this might be the first time the object
    // appears...
    if (!object_pose_gt.prev_H_current_world_) {
      continue;
    }

    const auto object_id = object_pose_gt.object_id_;

    gtsam::Pose3 L_W_k_gt = object_pose_gt.L_world_;
    // motion in world k-1 to k
    Motion3ReferenceFrame H_W_km1_k_gt(
        *object_pose_gt.prev_H_current_world_, MotionRepresentationStyle::F2F,
        ReferenceFrame::GLOBAL, frame_id - 1u, frame_id);

    ground_truth_dyno_state.object_trajectories.insert(
        object_id, frame_id, timestamp, {L_W_k_gt, H_W_km1_k_gt});
  }

  ground_truth_dyno_state.camera_trajectory.insert(frame_id, timestamp,
                                                   gt_packet.X_world_);

  ground_truth_publishers_->dyno_state_publisher_.publish(
      ground_truth_dyno_state);
}

}  // namespace dyno
