#pragma once
#include <Eigen/Dense>
#include <optional>
#include "axis/types.h"

namespace axis {

class SensorInterface {
public:
    virtual ~SensorInterface() = default;

    virtual std::optional<Eigen::VectorXd> processMeasurement() = 0;
    virtual bool isHealthy() const = 0;
    virtual double getLastUpdateTime() const = 0;

protected:
    double timestamp_{0.0};
    SensorHealth health_{SensorHealth::UNAVAILABLE};
    Eigen::MatrixXd covariance_;
};

} // namespace axis