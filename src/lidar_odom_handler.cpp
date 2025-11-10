#include "axis/lidar_odom_handler.h"
#include <iostream>

namespace axis { 
namespace {
constexpr double MIN_MATCH_SCORE = 0.3;
constexpr int MIN_FEATURES = 30;
}

 // ← ADD THIS

LidarOdomHandler::LidarOdomHandler() : SensorInterface() {}

void LidarOdomHandler::feedLidarOdom(const Eigen::Matrix<double, 6, 1>& deltaPose,
                                     const Eigen::Matrix<double, 6, 6>& covariance,
                                     double timestamp, double matchScore, int featureCount) {
    lastDeltaPose_ = deltaPose;
    lastCov_ = covariance;
    lastMatchScore_ = matchScore;
    lastFeatureCount_ = featureCount;
    timestamp_ = timestamp;
    updateHealth(matchScore, featureCount);
}

std::optional<Eigen::VectorXd> LidarOdomHandler::processMeasurement() {
    if (!isHealthy()) return std::nullopt;

    Eigen::VectorXd z(6);
    z = lastDeltaPose_;
    return z;
}

bool LidarOdomHandler::isHealthy() const {
    if (lastMatchScore_ < MIN_MATCH_SCORE || lastFeatureCount_ < MIN_FEATURES) return false;
    if (timestamp_ <= 0.0) return false;
    return true;
}

double LidarOdomHandler::getLastUpdateTime() const {
    return timestamp_;
}

void LidarOdomHandler::updateHealth(double matchScore, int featureCount) {
    if (timestamp_ <= 0.0) {
        health_ = SensorHealth::OFFLINE;
        std::cerr << "LiDAR odom: Invalid timestamp.\n";
    } else if (matchScore < MIN_MATCH_SCORE || featureCount < MIN_FEATURES) {
        health_ = SensorHealth::DEGRADED;
        std::cerr << "LiDAR odom: Poor feature/match score.\n";
    } else {
        health_ = SensorHealth::ONLINE;
    }
}

}  // namespace axis  ← ADD THIS