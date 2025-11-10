#pragma once
#include "axis/sensor_interface.h"
#include <Eigen/Dense>
#include <optional>

namespace axis {

struct ImuMeasurement {
    Eigen::Vector3d accel;
    Eigen::Vector3d gyro;
    double timestamp;
    bool valid;
};

class ImuHandler : public SensorInterface {
public:
    ImuHandler();
    void feedIMU(const Eigen::Vector3d& accel, const Eigen::Vector3d& gyro, double timestamp);
    std::optional<Eigen::VectorXd> processMeasurement() override;
    bool isHealthy() const override;
    double getLastUpdateTime() const override;
private:
    Eigen::Vector3d lastAccel_{Eigen::Vector3d::Zero()};
    Eigen::Vector3d lastGyro_{Eigen::Vector3d::Zero()};
    double lastTemp_{0.0};
    int noiseWindow_{10};
    void updateHealth(const Eigen::Vector3d& accel, const Eigen::Vector3d& gyro, double timestamp);
};

} // namespace axis
