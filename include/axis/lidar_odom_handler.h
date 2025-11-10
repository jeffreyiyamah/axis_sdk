#pragma once
#include "axis/sensor_interface.h"
#include <Eigen/Dense>
#include <optional>

namespace axis {

struct LidarOdomMeasurement {
    Eigen::Matrix<double, 6, 1> deltaPose; // [dx, dy, dz, droll, dpitch, dyaw]
    Eigen::Matrix<double, 6, 6> covariance;
    double timestamp;
    double matchScore;
    int featureCount;
    bool valid;
};

class LidarOdomHandler : public SensorInterface {
    public:
        LidarOdomHandler();
        void feedLidarOdom(const Eigen::Matrix<double, 6, 1>& deltaPose,
                        const Eigen::Matrix<double, 6, 6>& covariance,
                        double timestamp, double matchScore, int featureCount);
        std::optional<Eigen::VectorXd> processMeasurement() override;
        bool isHealthy() const override;
        double getLastUpdateTime() const override;
    private:
        Eigen::Matrix<double, 6, 1> lastDeltaPose_{Eigen::Matrix<double, 6, 1>::Zero()};
        Eigen::Matrix<double, 6, 6> lastCov_{Eigen::Matrix<double, 6, 6>::Identity()};
        double lastMatchScore_{0.0};
        int lastFeatureCount_{0};
        void updateHealth(double matchScore, int featureCount);
    };
} // namespace axis