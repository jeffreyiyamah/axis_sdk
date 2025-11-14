// tests/test_node_manager.cpp
#include "axis/mode_manager.h"
#include <gtest/gtest.h>
#include <iostream>

TEST(ModeManagerTest, BasicTransitions) {
    axis::ModeManager manager(axis::OperatingMode::NOMINAL);

    double t = 0.0;                     // simulation time
    const double HYSTERESIS = 1.1;      // > RECOVERY_HYSTERESIS_SEC

    auto tick = [&](double dt) {
        t += dt;
        manager.evaluateTransition(t);
    };

    // ------------------------------------------------------------------
    // 1. Lose GPS → DEAD_RECKONING
    // ------------------------------------------------------------------
    manager.updateSensorStatus("GPS", false);
    tick(0.1);                                          // evaluate immediately
    EXPECT_EQ(manager.getCurrentMode(), axis::OperatingMode::DEAD_RECKONING)
        << "Should enter DEAD_RECKONING after GPS loss";

    // ------------------------------------------------------------------
    // 2. LiDAR becomes healthy → LIDAR_ASSIST
    //    Need 2 ticks: one to mark recovery start, one to check hysteresis
    // ------------------------------------------------------------------
    manager.updateSensorStatus("LiDAR", true);
    tick(0.01);                                         // mark recovery time
    tick(HYSTERESIS);                                   // wait for hysteresis
    EXPECT_EQ(manager.getCurrentMode(), axis::OperatingMode::LIDAR_ASSIST)
        << "Should enter LIDAR_ASSIST after LiDAR recovers";

    // ------------------------------------------------------------------
    // 3. LoopClosure becomes healthy → RELOCALIZED
    //    LoopClosure doesn't use hysteresis, so 1 tick is enough
    // ------------------------------------------------------------------
    manager.updateSensorStatus("LoopClosure", true);
    tick(0.01);                                         // evaluate immediately
    EXPECT_EQ(manager.getCurrentMode(), axis::OperatingMode::RELOCALIZED)
        << "Should enter RELOCALIZED after loop closure";

    // ------------------------------------------------------------------
    // 4. GPS recovers → NOMINAL
    //    GPS uses hysteresis, so need 2 ticks again
    // ------------------------------------------------------------------
    manager.updateSensorStatus("GPS", true);
    tick(0.01);                                         // mark recovery time
    tick(HYSTERESIS);                                   // wait for hysteresis
    EXPECT_EQ(manager.getCurrentMode(), axis::OperatingMode::NOMINAL)
        << "Should return to NOMINAL after GPS recovers";

    // ------------------------------------------------------------------
    // 5. Diagnostics
    // ------------------------------------------------------------------
    std::cout << "\n" << manager.getDiagnostics(t) << std::endl;
}