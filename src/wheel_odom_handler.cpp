#include "axis/wheel_odom_handler.h"
#include <iostream>
#include <cmath>

namespace axis {
namespace {
constexpr double STALE_THRESH = 0.1; // 100ms
constexpr double MAX_SLIP = 2.0;  // m/s slip
}

WheelOdomHandler::WheelOdomHandler() : SensorInterface() {}

void WheelOdomHandler::feedWheelOdom(double v_left, double v_right, double timestamp) {
    lastVLeft_ = v_left;
    lastVRight_ = v_right;
    timestamp_ = timestamp;
    // Update forward and angular velocities
    lastForwardVel_ = 0.5 * (v_left + v_right);
    lastAngularVel_ = (v_right - v_left) / axleWidth_;
    updateHealth(v_left, v_right, timestamp);
}

std::optional<Eigen::VectorXd> WheelOdomHandler::processMeasurement() {
    if (!isHealthy()) return std::nullopt;

    Eigen::VectorXd z(2);
    z << lastForwardVel_, lastAngularVel_;
    return z;
}

bool WheelOdomHandler::isHealthy() const {
    if (timestamp_ <= 0.0) return false;
    // If slip is unreasonable, or zero update
    double slip = std::abs(lastVLeft_ - lastVRight_);
    if (slip > MAX_SLIP) return false;
    return true;
}

double WheelOdomHandler::getLastUpdateTime() const {
    return timestamp_;
}

void WheelOdomHandler::updateHealth(double v_left, double v_right, double ts) {
    if (ts <= 0.0) {
        health_ = SensorHealth::OFFLINE;
        std::cerr << "Wheel odom: Invalid timestamp.\n";
    } else if (std::abs(v_left - v_right) > MAX_SLIP) {
        health_ = SensorHealth::DEGRADED;
        std::cerr << "Wheel odom: Suspected wheel slip.\n";
    } else {
        health_ = SensorHealth::ONLINE;
    }
}

} // namespace axis
