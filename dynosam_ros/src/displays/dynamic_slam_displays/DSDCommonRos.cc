#include "dynosam_ros/displays/dynamic_slam_displays/DSDCommonRos.hpp"

#include <glog/logging.h>

#include "dynosam_common/DynamicObjects.hpp"
#include "dynosam_common/viz/Colour.hpp"
#include "dynosam_ros/RosUtils.hpp"
#include "dynosam_ros/displays/DisplaysCommon.hpp"
#include "tf2/exceptions.h"

namespace dyno {

DynoStatePublisher::DynoStatePublisher(const DisplayParams& params,
                                       rclcpp::Node::SharedPtr node)
    : params_(params), node_(node), tf_buffer_(node->get_clock()) {
  tf_listener_ = std::make_shared<tf2_ros::TransformListener>(tf_buffer_);

  if (params_.physical_camera_frame_id.empty()) {
    // No physical frame configured: pass poses through unchanged in world frame.
    T_map_world_ = gtsam::Pose3();
    T_map_world_ready_ = true;
    output_frame_id_ = params_.world_frame_id;
    RCLCPP_INFO(node_->get_logger(),
                "DynoStatePublisher: physical_camera_frame_id not set — "
                "publishing in world frame without map transform.");
  } else {
    output_frame_id_ = params_.map_frame_id;
    // FrontendInbuiltDisplayRos already broadcasts world→camera each frame.
    // Publishing map→camera here would give "camera" two parents in TF.
    publish_vo_tf_ = false;
  }

  vo_publisher_ =
      node_->create_publisher<nav_msgs::msg::Odometry>("odometry", 1);
  vo_path_publisher_ =
      node_->create_publisher<nav_msgs::msg::Path>("odometry_path", 1);
  tf_broadcaster_ = std::make_unique<tf2_ros::TransformBroadcaster>(*node_);

  object_odom_publisher_ =
      node->create_publisher<ObjectOdometry>("object_odometry", 1);
  multi_object_odom_path_publisher_ =
      node->create_publisher<MultiObjectOdometryPath>("object_odometry_path",
                                                      1);

  static_points_pub_ =
      node->create_publisher<sensor_msgs::msg::PointCloud2>("static_cloud", 1);
  dynamic_points_pub_ =
      node->create_publisher<sensor_msgs::msg::PointCloud2>("dynamic_cloud", 1);
}

void DynoStatePublisher::publish(const DynoState& state) {
  // On the first call look up T_map_world = T_map_camera_0 from TF.
  // DynoSAM initialises its SLAM world frame to identity at the first camera
  // pose, so "world" == the optical frame at t=0.  After this we always output
  // in the map frame so DynORecon's Z-up height filters work correctly.
  if (!T_map_world_ready_) {
    // physical_camera_frame_id is set (checked in constructor); look up the
    // latest available transform (rclcpp::Time(0)) so we don't depend on a
    // specific historical timestamp being in the TF buffer.
    try {
      const auto tf_stamped = tf_buffer_.lookupTransform(
          params_.map_frame_id, params_.physical_camera_frame_id,
          rclcpp::Time(0),
          rclcpp::Duration::from_seconds(0.1));
      const auto& t = tf_stamped.transform;
      T_map_world_ = gtsam::Pose3(
          gtsam::Rot3::Quaternion(t.rotation.w, t.rotation.x,
                                  t.rotation.y, t.rotation.z),
          gtsam::Point3(t.translation.x, t.translation.y, t.translation.z));
      T_map_world_ready_ = true;
      RCLCPP_INFO(node_->get_logger(),
                  "DynoStatePublisher: T_map_world cached from TF "
                  "(%s -> %s).",
                  output_frame_id_.c_str(),
                  params_.physical_camera_frame_id.c_str());
    } catch (const tf2::TransformException& ex) {
      RCLCPP_WARN_THROTTLE(node_->get_logger(), *node_->get_clock(), 2000,
                           "DynoStatePublisher: TF lookup failed, "
                           "skipping frame: %s",
                           ex.what());
      return;
    }
  }

  const FrameId frame_id = state.frame_id;
  const Timestamp timestamp = state.timestamp;

  const gtsam::Pose3 X_W_k = state.camera_trajectory.last().data;
  const gtsam::Pose3 X_map_k = T_map_world_ * X_W_k;

  DisplayCommon::publishOdometry(vo_publisher_, X_map_k, timestamp,
                                 output_frame_id_,
                                 params_.camera_frame_id);
  if (publish_vo_tf_) {
    std_msgs::msg::Header header;
    header.stamp = utils::toRosTime(timestamp);
    header.frame_id = output_frame_id_;
    sendTransform(X_map_k, header, params_.camera_frame_id);
  }

  // publish trajectory — transform each stored world-frame pose to map frame
  auto pose_vector = state.camera_trajectory.toDataVector();
  for (auto& pose : pose_vector) {
    pose = T_map_world_ * pose;
  }
  DisplayCommon::publishOdometryPath(vo_path_publisher_, pose_vector,
                                     timestamp, output_frame_id_);

  // publish local(?) static points
  DisplayCommon::publishPointCloud(static_points_pub_, state.local_static_map,
                                   X_map_k, output_frame_id_);

  DisplayCommon::publishPointCloud(dynamic_points_pub_, state.dynamic_map,
                                   X_map_k, output_frame_id_);

  publishObjects(frame_id, timestamp, state.object_trajectories);
}

void DynoStatePublisher::publishObjects(
    FrameId frame_id, Timestamp timestamp,
    const MultiObjectTrajectories& object_trajectories) {
  // get subset of trajectories that has an object observed at k
  auto object_trajectories_k =
      object_trajectories.trajectoriesAtFrame(frame_id);

  if (object_trajectories_k.empty()) {
    // Always publish so DynORecon's ApproximateTime synchronizer fires every
    // frame. Without this, DynORecon starves until the first object appears and
    // then jumps straight to frame N, skipping all prior static-map integration.
    MultiObjectOdometryPath empty_msg;
    empty_msg.header.stamp = utils::toRosTime(timestamp);
    empty_msg.header.frame_id = output_frame_id_;
    multi_object_odom_path_publisher_->publish(empty_msg);
    return;
  }

  ObjectOdometryMap object_odometries;

  MultiObjectOdometryPath multi_object_odom_paths;
  multi_object_odom_paths.header.stamp =
      utils::toRosTime(object_trajectories.lastTimestamp());
  multi_object_odom_paths.header.frame_id = output_frame_id_;

  for (const auto& [object_id, object_trajectory] : object_trajectories_k) {
    // latest object odometry
    ObjectOdometry object_odometry = constructObjectOdometry(
        object_id, object_trajectory.maxFrame(), object_trajectory,
        T_map_world_);
    object_odom_publisher_->publish(object_odometry);

    if (publish_oo_tf_) {
      sendObjectOdometryTransform(object_odometry);
    }

    // full path for object j
    ObjectOdometryPath object_path;

    std_msgs::msg::ColorRGBA colour_msg;
    convert(Color::uniqueId(object_id), colour_msg);

    // TODO: for now ignore segmenebts becuase the backend will always be
    // segmented due to non-consequative frames
    ObjectOdometryPath path_per_segment;
    path_per_segment.colour = colour_msg;
    path_per_segment.object_id = object_id;
    path_per_segment.path_segment = 0;
    path_per_segment.header = multi_object_odom_paths.header;
    for (FrameId frame_i : object_trajectory.toFrameIds()) {
      path_per_segment.object_odometries.push_back(
          constructObjectOdometry(object_id, frame_i, object_trajectory,
                                  T_map_world_));
    }

    multi_object_odom_paths.paths.push_back(path_per_segment);
    // // construct full paths
    // const auto trajectory_segments = object_trajectory.segments();
    // for (size_t i = 0; i < trajectory_segments.size(); i++) {
    //   size_t segment_id = i + 1;

    //   const auto& segment = trajectory_segments.at(i);

    //   ObjectOdometryPath path_per_segment;
    //   path_per_segment.colour = colour_msg;
    //   path_per_segment.object_id = object_id;
    //   path_per_segment.path_segment = segment_id;
    //   path_per_segment.header = multi_object_odom_paths.header;

    //   for (const auto& entry : segment.trajectory) {
    //     path_per_segment.object_odometries.push_back(
    //         constructObjectOdometry(object_id, entry));
    //   }

    //   multi_object_odom_paths.paths.push_back(path_per_segment);
    // }
  }

  multi_object_odom_path_publisher_->publish(multi_object_odom_paths);
}

ObjectOdometry DynoStatePublisher::constructObjectOdometry(
    ObjectId object_id, FrameId frame_id,
    const PoseWithMotionTrajectory& trajectory,
    const gtsam::Pose3& T_map_world) const {
  CHECK(trajectory.exists(frame_id));

  const PoseWithMotionEntry& pose_with_motion = trajectory.get(frame_id);
  const auto& entry = pose_with_motion.data;
  const auto L_W_k = entry.pose;
  const auto H_W_km1_k = entry.motion;
  const auto timestamp_k = pose_with_motion.timestamp;
  const auto frame_id_k = pose_with_motion.frame_id;

  CHECK_EQ(H_W_km1_k.to(), frame_id);
  CHECK_EQ(frame_id_k, frame_id);
  CHECK_EQ(H_W_km1_k.style(), MotionRepresentationStyle::F2F);

  const auto L_map_k = T_map_world * L_W_k;
  const auto frame_link = output_frame_id_;
  const auto child_link = "object_" + std::to_string(object_id) + "_link";

  ObjectOdometry object_odom;
  utils::convertWithHeader(L_map_k, object_odom.odom, timestamp_k, frame_link,
                           child_link);

  dyno::convert(H_W_km1_k.estimate(), object_odom.h_w_km1_k.pose);

  const std::optional<PoseWithMotionEntry> previous_entry =
      trajectory.getPrevious(pose_with_motion);
  if (previous_entry) {
    // this also implicitly checks that the previous entry is the immediately
    // previous entry
    CHECK_EQ(H_W_km1_k.from(), previous_entry->frame_id);

    const gtsam::Pose3 L_W_km1 = previous_entry->data.pose;
    const Timestamp timestamp_km1 = previous_entry->timestamp;

    gtsam::Vector6 body_velocity =
        calculateBodyMotion(H_W_km1_k, L_W_km1, timestamp_k, timestamp_km1);

    dyno::convert(body_velocity, object_odom.odom.twist.twist);
  }

  // TODO: body velocity
  object_odom.object_id = object_id;
  object_odom.sequence = frame_id_k;

  return object_odom;
}

void DynoStatePublisher::sendObjectOdometryTransform(
    const ObjectOdometry& object_odom) {
  sendTransform(object_odom.odom.pose.pose, object_odom.odom.header,
                object_odom.odom.child_frame_id);
}

}  // namespace dyno
