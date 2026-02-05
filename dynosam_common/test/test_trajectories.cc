
#include <glog/logging.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "dynosam_common/Trajectories.hpp"

using namespace dyno;

using TestTrajectoryInt = TrajectoryBase<int>;

TEST(TrajectoryBase, EmptyTrajectory) {
  TestTrajectoryInt traj;

  EXPECT_TRUE(traj.empty());
  EXPECT_EQ(traj.size(), 0u);
}

TEST(TrajectoryBase, InsertAndAccess) {
  TestTrajectoryInt traj;

  traj.insert(0, 0.0, 42);
  traj.insert(1, 0.1, 43);

  EXPECT_FALSE(traj.empty());
  EXPECT_EQ(traj.size(), 2u);

  EXPECT_EQ(traj.at(0), 42);
  EXPECT_EQ(traj.at(1), 43);
  EXPECT_DOUBLE_EQ(traj.timeAt(1), 0.1);
}

TEST(TrajectoryBase, DuplicateInsertThrows) {
  TestTrajectoryInt traj;

  traj.insert(0, 0.0, 42);

  EXPECT_THROW(traj.insert(0, 0.1, 43), TrajectoryEntryAlreadyExists);
}

TEST(TrajectoryBase, Exists) {
  TestTrajectoryInt traj;

  traj.insert(5, 0.5, 10);

  EXPECT_TRUE(traj.exists(5));
  EXPECT_FALSE(traj.exists(6));
}

TEST(TrajectoryBase, FirstAndLast) {
  TestTrajectoryInt traj;

  traj.insert(10, 1.0, 100);
  traj.insert(12, 1.2, 120);
  traj.insert(11, 1.1, 110);

  EXPECT_EQ(traj.first().frame_id, 10);
  EXPECT_EQ(traj.last().frame_id, 12);

  EXPECT_EQ(traj.first().data, 100);
  EXPECT_EQ(traj.last().data, 120);
}

TEST(TrajectoryBase, Update) {
  TestTrajectoryInt traj;
  traj.insert(0, 0.0, 1);

  EXPECT_TRUE(traj.update(0, 99));
  EXPECT_EQ(traj.at(0), 99);

  EXPECT_FALSE(traj.update(1, 10));
}

TEST(TrajectoryBase, GetPrevious) {
  TestTrajectoryInt traj;

  traj.insert(1, 0.1, 10);
  traj.insert(3, 0.3, 30);
  traj.insert(5, 0.5, 50);

  TestTrajectoryInt::Entry prev;

  EXPECT_TRUE(traj.getPrevious(prev, traj.get(3)));
  EXPECT_EQ(prev.frame_id, 1);

  EXPECT_FALSE(traj.getPrevious(prev, traj.first()));
}

TEST(TrajectoryBase, Segments) {
  TestTrajectoryInt traj;

  traj.insert(0, 0.0, 0);
  traj.insert(1, 0.1, 1);
  traj.insert(3, 0.3, 3);
  traj.insert(4, 0.4, 4);
  traj.insert(6, 0.6, 6);

  auto segments = traj.segments();

  ASSERT_EQ(segments.size(), 3u);

  EXPECT_EQ(segments[0].start_frame, 0);
  EXPECT_EQ(segments[0].end_frame, 1);
  EXPECT_TRUE(segments[0].check());

  EXPECT_EQ(segments[1].start_frame, 3);
  EXPECT_EQ(segments[1].end_frame, 4);
  EXPECT_TRUE(segments[1].check());

  EXPECT_EQ(segments[2].start_frame, 6);
  EXPECT_EQ(segments[2].end_frame, 6);
  EXPECT_TRUE(segments[2].check());
}

TEST(TrajectoryBase, ToVector) {
  TestTrajectoryInt traj;

  traj.insert(2, 0.2, 20);
  traj.insert(1, 0.1, 10);

  auto vec = traj.toVector();

  ASSERT_EQ(vec.size(), 2u);
  EXPECT_EQ(vec[0].frame_id, 1);
  EXPECT_EQ(vec[1].frame_id, 2);
}
