#pragma once
#include "axis/sensor_interface.h"
#include <optional>

namespace axis {

struct WheelOdomMeasurement {
    double forwardVel;   // [m/s]
    double angularVel;   // [rad/s]
    double heading;      // delta heading
    double timestamp;
    bool valid;
};

class WheelOdomHandler : public SensorInterface {
    public:
        WheelOdomHandler();
        void feedWheelOdom(double v_left, double v_right, double timestamp);
        std::optional<Eigen::VectorXd> processMeasurement() override;
        bool isHealthy() const override;
        double getLastUpdateTime() const override;
    private:
        double lastVLeft_{0.0}, lastVRight_{0.0}, lastHeading_{0.0};
        double lastForwardVel_{0.0}, lastAngularVel_{0.0};
        double axleWidth_{0.5}; // meters, can be parameterized
        void updateHealth(double v_left, double v_right, double timestamp);
    };
} // namespace axis