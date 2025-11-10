#include "axis/imu_handler.h"
#include <iostream>
#include <cmath>

namespace axis {

ImuHandler::ImuHandler() : SensorInterface() {}

void ImuHandler::feedIMU(const Eigen::Vector3d& accel, const Eigen::Vector3d& gyro, double timestamp) {
    lastAccel_ = accel;
    lastGyro_ = gyro;
    timestamp_ = timestamp;
    updateHealth(accel, gyro, timestamp);
}

std::optional<Eigen::VectorXd> ImuHandler::processMeasurement() {
    if (!isHealthy()) {
        std::cerr << "IMU unhealthy. Dropping measurement.\n";
        return std::nullopt;
    }

    Eigen::VectorXd z(6);
    z << lastAccel_.x(), lastAccel_.y(), lastAccel_.z(),
         lastGyro_.x(),  lastGyro_.y(),  lastGyro_.z();
    return z;
}

bool ImuHandler::isHealthy() const {
    // Example: check timestamp, accel & gyro within plausible range
    double now = timestamp_; // In production, use a monotonic system clock
    constexpr double STALE_THRESH = 0.1; // 100ms
    constexpr double ACC_MAX = 100.0; // m/s^2
    constexpr double GYRO_MAX = 20.0;  // rad/s
    if (now <= 0.0) return false;
    if (std::abs(lastAccel_.maxCoeff()) > ACC_MAX || std::abs(lastAccel_.minCoeff()) < -ACC_MAX) return false;
    if (std::abs(lastGyro_.maxCoeff()) > GYRO_MAX || std::abs(lastGyro_.minCoeff()) < -GYRO_MAX) return false;
    // Simple freshness: user should update timestamp_ externally
    return true;
}

double ImuHandler::getLastUpdateTime() const {
    return timestamp_;
}

void ImuHandler::updateHealth(const Eigen::Vector3d& accel, const Eigen::Vector3d& gyro, double ts) {
    // Staleness and saturation checks
    constexpr double ACC_MAX = 100.0;
    constexpr double GYRO_MAX = 20.0;
    if (ts <= 0.0) {
        health_ = SensorHealth::OFFLINE;
        std::cerr << "IMU: Invalid timestamp.\n";
    } else if ((accel.array().abs() > ACC_MAX).any() || (gyro.array().abs() > GYRO_MAX).any()) {
        health_ = SensorHealth::DEGRADED;
        std::cerr << "IMU: Sensor saturation detected.\n";
    } else {
        health_ = SensorHealth::ONLINE;
    }
}

} // namespace axis
