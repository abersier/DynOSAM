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

#include "dynosam_ros/displays/FrontendDisplayRos.hpp"

#include <pcl/common/transforms.h>
#include <pcl/filters/random_sample.h>
#include <unordered_map>

#include "cv_bridge/cv_bridge.h"
#include "dynosam_ros/RosUtils.hpp"
#include "rclcpp/qos.hpp"

namespace dyno {

FrontendDisplayRos::FrontendDisplayRos(
    const CanonicalSensorRig::ConstPtr& sensor_rig,
    rclcpp::Node::SharedPtr node, rclcpp::Node::SharedPtr ground_truth_node)
    : FrontendDisplay(),
      sensor_rig_(sensor_rig),
      dyno_state_publisher_(sensor_rig, node,
                            // pulish vo, oo transforms and wireframe cameras
                            {true, true, true}) {
  tracking_image_pub_ =
      image_transport::create_publisher(node.get(), "tracking_image");

  // const rclcpp::SensorDataQoS sensor_qos;
  dense_dynamic_cloud_pub_ =
      node->create_publisher<sensor_msgs::msg::PointCloud2>(
          "dense_labelled_cloud", rclcpp::QoS(10));

  labelled_cloud_max_static_points_ =
      ros::Parameter::Builder(node, "labelled_cloud_max_static_points", 2000)
          .description(
              "Max static points published in dense_labelled_cloud (0 = no "
              "limit). Random-sampled before publish. Matches oracle "
              "points_static semantics — tune to match DynORecon throughput.")
          .finish()
          .get<int>();

  labelled_cloud_max_dynamic_points_ =
      ros::Parameter::Builder(node, "labelled_cloud_max_dynamic_points", 800)
          .description(
              "Max dynamic points per object in dense_labelled_cloud (0 = no "
              "limit). Random-sampled per object before publish — object count "
              "and size are unknown so per-object budget is the only fair "
              "limit. Matches oracle points_per_object semantics.")
          .finish()
          .get<int>();

  if (ground_truth_node) {
    RCLCPP_INFO_STREAM(node->get_logger(), "Creating ground truth publishers");
    ground_truth_publishers_.emplace(sensor_rig, ground_truth_node);
  }
}

FrontendDisplayRos::GroundTruthPublishers::GroundTruthPublishers(
    const CanonicalSensorRig::ConstPtr& sensor_rig,
    rclcpp::Node::SharedPtr ground_truth_node)
    : dyno_state_publisher_(sensor_rig, CHECK_NOTNULL(ground_truth_node)) {}

void FrontendDisplayRos::spinOnce(
    const RealtimeOutput::ConstPtr& frontend_output) {
  VLOG(20) << "Spinning FrontendDisplayRos k="
           << frontend_output->state.frame_id;
  dyno_state_publisher_.publish(frontend_output->state);

  tryPublishDebugImagery(frontend_output);
  tryPublishGroundTruth(frontend_output);

  if (frontend_output->dense_labelled_cloud &&
      !frontend_output->dense_labelled_cloud->empty()) {
    const gtsam::Pose3 T_RC = sensor_rig_->getCanonicalExtrinsics();
    const gtsam::Pose3 T_WC =
        frontend_output->state.camera_trajectory.last().data;
    PointCloudLabelRGB cloud_odom;
    pcl::transformPointCloud(*frontend_output->dense_labelled_cloud, cloud_odom,
                             (T_RC * T_WC).matrix().cast<float>());

    // Downsample static and dynamic points independently before publishing.
    // DynoSAM reprojects every depth pixel (~300k pts); oracle sends only
    // ~2000 static pts. Without this, DynORecon's VDB integration is ~150x
    // heavier per frame than in oracle mode, pinning one CPU core.
    // Static = background_label (0); dynamic = any non-zero label.
    if (labelled_cloud_max_static_points_ > 0 ||
        labelled_cloud_max_dynamic_points_ > 0) {
      PointCloudLabelRGB static_cloud, dynamic_cloud;
      static_cloud.reserve(cloud_odom.size());
      dynamic_cloud.reserve(cloud_odom.size());
      for (const auto& pt : cloud_odom) {
        if (pt.label == background_label) static_cloud.push_back(pt);
        else                               dynamic_cloud.push_back(pt);
      }

      cloud_odom.clear();

      auto sample = [](const PointCloudLabelRGB& in, int max_pts,
                       PointCloudLabelRGB& out) {
        if (max_pts > 0 && static_cast<int>(in.size()) > max_pts) {
          pcl::RandomSample<PointLabelRGB> rs;
          rs.setInputCloud(in.makeShared());
          rs.setSample(static_cast<unsigned int>(max_pts));
          rs.filter(out);
        } else {
          out = in;
        }
      };

      // Static: single pool, sample globally.
      PointCloudLabelRGB static_sampled;
      sample(static_cloud, labelled_cloud_max_static_points_, static_sampled);

      // Dynamic: sample per-object so every object gets the same budget
      // regardless of how many objects exist or how large they are.
      // Mirrors oracle's points_per_object semantics.
      PointCloudLabelRGB dynamic_sampled;
      if (labelled_cloud_max_dynamic_points_ > 0 && !dynamic_cloud.empty()) {
        std::unordered_map<uint32_t, PointCloudLabelRGB> per_object;
        for (const auto& pt : dynamic_cloud) per_object[pt.label].push_back(pt);
        for (auto& [label, obj_cloud] : per_object) {
          PointCloudLabelRGB obj_sampled;
          sample(obj_cloud, labelled_cloud_max_dynamic_points_, obj_sampled);
          dynamic_sampled += obj_sampled;
        }
      } else {
        dynamic_sampled = dynamic_cloud;
      }

      cloud_odom = static_sampled + dynamic_sampled;
    }

    sensor_msgs::msg::PointCloud2 pc2_msg;
    pcl::toROSMsg(cloud_odom, pc2_msg);
    pc2_msg.header.frame_id = sensor_rig_->getReferenceFrames().odom_frame;
    pc2_msg.header.stamp = ros::toRosTime(frontend_output->state.timestamp);
    dense_dynamic_cloud_pub_->publish(pc2_msg);
  }
}

void FrontendDisplayRos::tryPublishDebugImagery(
    const RealtimeOutput::ConstPtr& frontend_output) {
  const DebugImagery& debug_imagery = frontend_output->debug_imagery;
  if (debug_imagery.tracking_image.empty()) return;

  std_msgs::msg::Header hdr;
  sensor_msgs::msg::Image::SharedPtr msg =
      cv_bridge::CvImage(hdr, "bgr8", debug_imagery.tracking_image)
          .toImageMsg();
  tracking_image_pub_.publish(msg);
}

void FrontendDisplayRos::tryPublishGroundTruth(
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
