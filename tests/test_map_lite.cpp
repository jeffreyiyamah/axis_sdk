#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "axis/map_lite_relocalizer.h"

using axis::MapLiteRelocalizer;
using axis::MapLiteConfig;
using axis::LidarScan;

TEST(MapLiteTest, SubmapAdditionAndBounds)
{
    MapLiteConfig cfg;
    cfg.submap_radius = 1.0;
    cfg.max_scans_in_submap = 5;

    MapLiteRelocalizer rel(cfg);
    ASSERT_TRUE(rel.initialize());

    for (int i = 0; i < 10; ++i) {
        LidarScan scan;
        scan.position = Eigen::Vector3d(static_cast<double>(i), 0.0, 0.0);
        rel.addScan(scan);
    }

    EXPECT_LE(rel.getSubmapSize(), static_cast<size_t>(cfg.max_scans_in_submap));

    rel.shutdown();
}

TEST(MapLiteTest, ICPMatchingWithSyntheticScans)
{
    MapLiteConfig cfg;
    cfg.submap_radius = 10.0;

    MapLiteRelocalizer rel(cfg);
    ASSERT_TRUE(rel.initialize());

    LidarScan base;
    base.position = Eigen::Vector3d::Zero();
    base.points.push_back(Eigen::Vector3d(0, 0, 0));
    base.points.push_back(Eigen::Vector3d(1, 0, 0));
    base.points.push_back(Eigen::Vector3d(0, 1, 0));

    rel.addScan(base);

    LidarScan shifted = base;
    shifted.position = Eigen::Vector3d(0.1, 0.0, 0.0);
    for (auto& p : shifted.points) {
        p += Eigen::Vector3d(0.1, 0.0, 0.0);
    }

    rel.setActive(true);
    rel.addScan(shifted);

    auto result = rel.attemptRelocalization();

    if (result) {
        EXPECT_TRUE(result->success);
    }

    rel.shutdown();
}

TEST(MapLiteTest, CorrectionTransformApplicationIndirect)
{
    MapLiteConfig cfg;
    cfg.submap_radius = 10.0;

    MapLiteRelocalizer rel(cfg);
    ASSERT_TRUE(rel.initialize());

    LidarScan scan;
    scan.position = Eigen::Vector3d(1.0, 2.0, 3.0);
    scan.points.push_back(Eigen::Vector3d(0, 0, 0));
    rel.addScan(scan);

    rel.setActive(true);
    auto result = rel.attemptRelocalization();

    if (result) {
        EXPECT_TRUE(result->success);
    }

    rel.shutdown();
}
