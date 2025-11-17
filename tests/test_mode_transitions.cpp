#include <gtest/gtest.h>
#include "axis/mode_manager.h"

using axis::ModeManager;
using axis::OperatingMode;

TEST(ModeTransitionsTest, AllStateTransitions)
{
    ModeManager manager(OperatingMode::NOMINAL);
    double t = 0.0;

    auto tick = [&](double dt) {
        t += dt;
        manager.evaluateTransition(t);
    };

    manager.updateSensorStatus("GPS", false);
    tick(0.1);
    EXPECT_EQ(manager.getCurrentMode(), OperatingMode::DEAD_RECKONING);

    manager.updateSensorStatus("LiDAR", true);
    tick(0.01);
    tick(1.1);
    EXPECT_EQ(manager.getCurrentMode(), OperatingMode::LIDAR_ASSIST);

    manager.updateSensorStatus("LoopClosure", true);
    tick(0.01);
    EXPECT_EQ(manager.getCurrentMode(), OperatingMode::RELOCALIZED);

    manager.updateSensorStatus("GPS", true);
    tick(0.01);
    tick(1.1);
    EXPECT_EQ(manager.getCurrentMode(), OperatingMode::NOMINAL);
}

TEST(ModeTransitionsTest, SensorFailureScenarios)
{
    ModeManager manager(OperatingMode::NOMINAL);
    double t = 0.0;

    auto tick = [&](double dt) {
        t += dt;
        manager.evaluateTransition(t);
    };

    manager.updateSensorStatus("IMU", false);
    tick(0.1);
    EXPECT_EQ(manager.getCurrentMode(), OperatingMode::FAIL_SAFE);
}

TEST(ModeTransitionsTest, TimeoutBasedTransitions)
{
    ModeManager manager(OperatingMode::DEAD_RECKONING);
    
    // Ensure GPS is not available (otherwise it will recover and transition to NOMINAL)
    manager.updateSensorStatus("GPS", false);
    
    double t = 0.0;

    auto tick = [&](double dt) {
        t += dt;
        manager.evaluateTransition(t);
    };

    for (int i = 0; i < 70; ++i) {
        tick(1.0);
    }

    EXPECT_EQ(manager.getCurrentMode(), OperatingMode::FAIL_SAFE);
}

TEST(ModeTransitionsTest, ModeHysteresis)
{
    ModeManager manager(OperatingMode::DEAD_RECKONING);
    double t = 0.0;

    auto tick = [&](double dt) {
        t += dt;
        manager.evaluateTransition(t);
    };

    manager.updateSensorStatus("GPS", true);
    tick(0.01);
    EXPECT_EQ(manager.getCurrentMode(), OperatingMode::DEAD_RECKONING);

    tick(1.1);
    EXPECT_EQ(manager.getCurrentMode(), OperatingMode::NOMINAL);
}
