#include "axis/config_parser.h"
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>
#include <fstream>
#include <stdexcept>

namespace axis {

class ConfigParser::Impl {
public:
    Impl(const std::string& path) : yaml_path_(path) { loadConfig(); }
    void loadConfig() {
        node_ = YAML::LoadFile(yaml_path_);
    }

    // Sensor flags
    bool isIMUEnabled() const { return getFlag("imu"); }
    bool isWheelOdomEnabled() const { return getFlag("wheel_odom"); }
    bool isLidarOdomEnabled() const { return getFlag("lidar_odom"); }
    bool isGPSEnabled() const { return getFlag("gps"); }
    bool isVOEnabled() const { return getFlag("visual_odom"); }

    // ML
    std::string getMLModelPath() const { return node_["ml_model"]["path"].as<std::string>(""); }
    std::unordered_map<std::string, double> getMLParams() const {
        std::unordered_map<std::string, double> params;
        if (node_["ml_model"]["params"]) {
            for(const auto& it : node_["ml_model"]["params"].as<YAML::Node>()) {
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
private:
    bool getFlag(const char* key) const {
        return node_["sensors"][key].as<bool>(false);
    }
    Eigen::MatrixXd toMatrix(const YAML::Node& n) const {
        if (!n || !n.IsSequence()) return Eigen::MatrixXd();
        const size_t rows = n.size();
        if (rows == 0) return Eigen::MatrixXd();
        size_t cols = n[0].size();
        Eigen::MatrixXd m(rows, cols);
        for (size_t i = 0; i < rows; ++i)
            for(size_t j = 0; j < cols; ++j)
                m(i,j) = n[i][j].as<double>();
        return m;
    }
    std::string yaml_path_;
    YAML::Node node_;
};

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

} // namespace axis
