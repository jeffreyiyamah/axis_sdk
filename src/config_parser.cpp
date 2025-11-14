#include "axis/config_parser.h"
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <fstream>
#include <stdexcept>

namespace axis {

// ================================================================
// Impl definition
// ================================================================
class ConfigParser::Impl {
public:
    Impl(const std::string& path) : yaml_path_(path) {
        loadConfig();
    }

    void loadConfig() {
        node_ = YAML::LoadFile(yaml_path_);
    }

    // ------------------------------------------------------------
    // Existing Strongly-Typed API
    // ------------------------------------------------------------

    // Sensor flags
    bool isIMUEnabled() const        { return getFlag("imu"); }
    bool isWheelOdomEnabled() const  { return getFlag("wheel_odom"); }
    bool isLidarOdomEnabled() const  { return getFlag("lidar_odom"); }
    bool isGPSEnabled() const        { return getFlag("gps"); }
    bool isVOEnabled() const         { return getFlag("visual_odom"); }

    // ML
    std::string getMLModelPath() const {
        return node_["ml_model"]["path"].as<std::string>("");
    }

    std::unordered_map<std::string, double> getMLParams() const {
        std::unordered_map<std::string, double> params;
        if (node_["ml_model"]["params"]) {
            for (const auto& it : node_["ml_model"]["params"].as<YAML::Node>()) {
                params[it.first.as<std::string>()] = it.second.as<double>();
            }
        }
        return params;
    }

    // EKF
    Eigen::MatrixXd getEKFInitialCovariance() const {
        return toMatrix(node_["ekf"]["initial_covariance"]);
    }

    Eigen::MatrixXd getEKFProcessNoise() const {
        return toMatrix(node_["ekf"]["process_noise"]);
    }

    double getEKFUpdateRate() const {
        return node_["ekf"]["update_rate"].as<double>(50.0);
    }

    // Map-Lite
    int getSubmapSize() const {
        return node_["maplite"]["submap_size"].as<int>(10);
    }

    double getICPThreshold() const {
        return node_["maplite"]["icp_threshold"].as<double>(1.0);
    }

    // Diagnostics
    double getDiagPubRate() const {
        return node_["diagnostics"]["publish_rate"].as<double>(1.0);
    }

    // Expose the raw YAML node for generic getters
    const YAML::Node& root() const { return node_; }

private:
    bool getFlag(const char* key) const {
        return node_["sensors"][key].as<bool>(false);
    }

    Eigen::MatrixXd toMatrix(const YAML::Node& n) const {
        if (!n || !n.IsSequence()) return Eigen::MatrixXd();
        const size_t rows = n.size();
        if (rows == 0) return Eigen::MatrixXd();
        const size_t cols = n[0].size();

        Eigen::MatrixXd m(rows, cols);
        for (size_t i = 0; i < rows; ++i)
            for (size_t j = 0; j < cols; ++j)
                m(i, j) = n[i][j].as<double>();
        return m;
    }

    std::string yaml_path_;
    YAML::Node node_;
};

// ================================================================
// ConfigParser public API
// ================================================================

ConfigParser::ConfigParser(const std::string& yaml_path)
    : impl_(std::make_unique<Impl>(yaml_path)) {}

ConfigParser::~ConfigParser() = default;

void ConfigParser::loadConfig() { impl_->loadConfig(); }

bool ConfigParser::isIMUEnabled() const { return impl_->isIMUEnabled(); }
bool ConfigParser::isWheelOdomEnabled() const { return impl_->isWheelOdomEnabled(); }
bool ConfigParser::isLidarOdomEnabled() const { return impl_->isLidarOdomEnabled(); }
bool ConfigParser::isGPSEnabled() const { return impl_->isGPSEnabled(); }
bool ConfigParser::isVOEnabled() const { return impl_->isVOEnabled(); }

std::string ConfigParser::getMLModelPath() const { return impl_->getMLModelPath(); }
std::unordered_map<std::string, double> ConfigParser::getMLParams() const { return impl_->getMLParams(); }

Eigen::MatrixXd ConfigParser::getEKFInitialCovariance() const { return impl_->getEKFInitialCovariance(); }
Eigen::MatrixXd ConfigParser::getEKFProcessNoise() const { return impl_->getEKFProcessNoise(); }
double ConfigParser::getEKFUpdateRate() const { return impl_->getEKFUpdateRate(); }

int ConfigParser::getSubmapSize() const { return impl_->getSubmapSize(); }
double ConfigParser::getICPThreshold() const { return impl_->getICPThreshold(); }

double ConfigParser::getDiagPubRate() const { return impl_->getDiagPubRate(); }

// ================================================================
// Generic YAML Accessors (used by AxisLocalizer)
// ================================================================

// Helper: resolve "a.b.c" YAML paths
static YAML::Node resolveNodePath(const YAML::Node& root, const std::string& path) {
    YAML::Node current = root;
    size_t start = 0;

    while (true) {
        size_t dot = path.find('.', start);
        std::string key = (dot == std::string::npos)
                          ? path.substr(start)
                          : path.substr(start, dot - start);

        if (!current[key]) return YAML::Node();  // missing

        current = current[key];

        if (dot == std::string::npos) break;
        start = dot + 1;
    }

    return current;
}

template<typename T>
T ConfigParser::get(const std::string& key, const T& default_value) const {
    try {
        YAML::Node node = resolveNodePath(impl_->root(), key);
        if (!node) return default_value;
        return node.as<T>();
    } catch (...) {
        return default_value;
    }
}

bool ConfigParser::getBool(const std::string& key, bool default_value) const {
    return get<bool>(key, default_value);
}

double ConfigParser::getDouble(const std::string& key, double default_value) const {
    return get<double>(key, default_value);
}

int ConfigParser::getInt(const std::string& key, int default_value) const {
    return get<int>(key, default_value);
}

std::string ConfigParser::getString(const std::string& key,
                                    const std::string& default_value) const {
    return get<std::string>(key, default_value);
}

} // namespace axis
