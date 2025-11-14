#pragma once
#include <string>
#include <memory>
#include <unordered_map>
#include <Eigen/Core>
#include <yaml-cpp/yaml.h>

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

    // --------------------------------------------------------------------
    // Generic YAML accessors (AxisLocalizer uses these)
    // --------------------------------------------------------------------
    template<typename T>
    T get(const std::string& key, const T& default_value) const;

    bool getBool(const std::string& key, bool default_value) const;
    double getDouble(const std::string& key, double default_value) const;
    int getInt(const std::string& key, int default_value) const;
    std::string getString(const std::string& key,
                          const std::string& default_value) const;

private:
    // Forward declaration ONLY — real definition lives in .cpp
    class Impl;

    std::unique_ptr<Impl> impl_;
};

} // namespace axis
