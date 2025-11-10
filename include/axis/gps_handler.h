#pragma once
#include "axis/sensor_interface.h"
#include <Eigen/Dense>
#include <optional>
#include <cstdint>
#include <array>

namespace axis {

struct GpsMeasurement {
    Eigen::Vector3d positionENU;
    Eigen::Matrix3d covariance;
    double timestamp;
    bool valid;
};

class GpsHandler : public SensorInterface {
public:
    GpsHandler();
    void feedGPS(double lat, double lon, double alt, const Eigen::Matrix3d& covariance, double timestamp, int fixType, int numSatellites);
    std::optional<Eigen::VectorXd> processMeasurement() override;
    bool isHealthy() const override;
    double getLastUpdateTime() const override;
private:
    // Origin for ENU conversion
    double originLat_{0.0}, originLon_{0.0}, originAlt_{0.0};
    bool originSet_{false};
    Eigen::Vector3d lastPosENU_{Eigen::Vector3d::Zero()};
    Eigen::Matrix3d lastCov_{Eigen::Matrix3d::Identity()};
    int lastFixType_{0};
    int lastNumSat_{0};
    double lastHDOP_{99.9};
    void updateHealth(int fixType, int numSat, double hdop);
    Eigen::Vector3d geoToENU(double lat, double lon, double alt);
};

} // namespace axis
