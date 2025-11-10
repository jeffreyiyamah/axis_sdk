#include "axis/gps_handler.h"
#include <iostream>
#include <cmath>

namespace axis {
namespace {
constexpr double DEG2RAD = M_PI / 180.0;
constexpr double EARTH_RADIUS = 6378137.0;
constexpr double MIN_HDOP = 2.5;
constexpr int MIN_FIX = 2; // >=2D fix
constexpr int MIN_SATS = 5;
} // namespace

GpsHandler::GpsHandler() : SensorInterface() {}

void GpsHandler::feedGPS(double lat, double lon, double alt, const Eigen::Matrix3d& covariance,
                         double timestamp, int fixType, int numSatellites) {
    if (!originSet_ && fixType >= MIN_FIX) {
        originLat_ = lat;
        originLon_ = lon;
        originAlt_ = alt;
        originSet_ = true;
    }
    lastPosENU_ = originSet_ ? geoToENU(lat, lon, alt) : Eigen::Vector3d::Zero();
    lastCov_ = covariance;
    lastFixType_ = fixType;
    lastNumSat_ = numSatellites;
    timestamp_ = timestamp;
    // For simplicity, HDOP as sqrt(trace(cov)/3)
    lastHDOP_ = std::sqrt(covariance.trace()/3.0);
    updateHealth(fixType, numSatellites, lastHDOP_);
}

std::optional<Eigen::VectorXd> GpsHandler::processMeasurement() {
    if (!isHealthy()) {
        std::cerr << "GPS health degraded or offline. Measurement dropped.\n";
        health_ = SensorHealth::DEGRADED;
        return std::nullopt;
    }

    // Pack position into Eigen::VectorXd (3x1)
    Eigen::VectorXd measurement(3);
    measurement << lastPosENU_.x(), lastPosENU_.y(), lastPosENU_.z();

    // Optional: store covariance elsewhere (e.g., member or pass via EKF separately)
    // For now, just return the measurement vector
    return measurement;
}

bool GpsHandler::isHealthy() const {
    if (!originSet_) return false;
    if (lastFixType_ < MIN_FIX || lastNumSat_ < MIN_SATS || lastHDOP_ > MIN_HDOP) return false;
    if (timestamp_ <= 0.0) return false;
    return true;
}

double GpsHandler::getLastUpdateTime() const {
    return timestamp_;
}

void GpsHandler::updateHealth(int fixType, int numSat, double hdop) {
    if (timestamp_ <= 0.0) {
        health_ = SensorHealth::OFFLINE;
        std::cerr << "GPS: Invalid timestamp.\n";
    } else if (fixType < MIN_FIX || numSat < MIN_SATS || hdop > MIN_HDOP) {
        health_ = SensorHealth::DEGRADED;
        std::cerr << "GPS: Poor fix or few sats or high HDOP.\n";
    } else {
        health_ = SensorHealth::ONLINE;
    }
}

Eigen::Vector3d GpsHandler::geoToENU(double lat, double lon, double alt) {
    // Flat earth approximation for demo
    double dLat = (lat - originLat_) * DEG2RAD;
    double dLon = (lon - originLon_) * DEG2RAD;
    double dAlt = alt - originAlt_;
    double x = dLon * EARTH_RADIUS * std::cos(originLat_ * DEG2RAD);
    double y = dLat * EARTH_RADIUS;
    double z = dAlt;
    return Eigen::Vector3d(x, y, z);
}

} // namespace axis
