#pragma once
#include <string>
#include <memory>
#include <unordered_map>
#include <Eigen/Core>

namespace axis {

class ConfigParser {
public:
    explicit ConfigParser(const std::string& yaml_path);
    ~ConfigParser();

    // Load/parsing
    void loadConfig();

    // Sensor enable/disable getters
    bool isIMUEnabled() const;
    bool isWheelOdomEnabled() const;
    bool isLidarOdomEnabled() const;
    bool isGPSEnabled() const;
    bool isVOEnabled() const;

    // ML Model
    std::string getMLModelPath() const;
    std::unordered_map<std::string, double> getMLParams() const;

    // EKF tuning
    Eigen::MatrixXd getEKFInitialCovariance() const;
    Eigen::MatrixXd getEKFProcessNoise() const;
    double getEKFUpdateRate() const;

    // Map-Lite
    int getSubmapSize() const;
    double getICPThreshold() const;

    // Diagnostics
    double getDiagPubRate() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace axis
