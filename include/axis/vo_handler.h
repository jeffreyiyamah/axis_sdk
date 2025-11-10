#pragma once
#include "axis/sensor_interface.h"
#include <optional>

namespace axis {

struct VoMeasurement {
    // Placeholder for future VO measurement struct
    bool valid;
    double timestamp;
};

class VoHandler : public SensorInterface {
public:
    VoHandler() = default;
    // Placeholder for feedVO()
    void feedVO();
    std::optional<Eigen::VectorXd> processMeasurement() override { return std::nullopt; }
    bool isHealthy() const override { return false; }
    double getLastUpdateTime() const override { return timestamp_; }
};

} // namespace axis