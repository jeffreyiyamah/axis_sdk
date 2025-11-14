#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <string>
#include <array>
#include <cstdint>

namespace axis {

using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;

enum class SensorHealth {
    ONLINE,
    DEGRADED,
    OFFLINE,
    UNAVAILABLE
};

enum class OperatingMode {
    NOMINAL,
    DEAD_RECKONING,
    LIDAR_ASSIST,
    RELOCALIZED,
    FAIL_SAFE
};

struct PoseMessage {
    double timestamp{0.0};
    std::string frame_id;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
    Eigen::Vector3d velocity = Eigen::Vector3d::Zero();
    std::array<double, 36> covariance{};
    double confidence{0.0};
    OperatingMode mode{OperatingMode::NOMINAL};

    PoseMessage() = default;
    PoseMessage(double ts, const std::string& fid,
                const Eigen::Vector3d& pos,
                const Eigen::Quaterniond& ori,
                const Eigen::Vector3d& vel,
                const std::array<double,36>& cov,
                double conf, OperatingMode m)
        : timestamp(ts), frame_id(fid), position(pos), orientation(ori),
          velocity(vel), covariance(cov), confidence(conf), mode(m) {}
};

struct SensorMeasurement {
    double timestamp{0.0};
    bool valid{false};

    SensorMeasurement() = default;
    SensorMeasurement(double ts, bool v) : timestamp(ts), valid(v) {}
};

} // namespace axis
