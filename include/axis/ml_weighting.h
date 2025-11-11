#pragma once
#include <Eigen/Dense>
#include <string>
#include <vector>
#include <memory>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cmath>

namespace axis {

/**
 * @brief Sensor features for ML-based weighting
 */
struct SensorFeatures {
    double imu_noise_variance{0.0};
    double lidar_return_density{0.0};
    double vo_feature_count{0.0};
    double gps_snr{0.0};
    double gps_satellite_count{0.0};
    double ekf_innovation_imu{0.0};
    double ekf_innovation_wheel{0.0};
    double ekf_innovation_lidar{0.0};
    double ekf_innovation_vo{0.0};
    double ekf_innovation_gps{0.0};
    
    Eigen::VectorXd toVector() const {
        Eigen::VectorXd vec(10);
        vec << imu_noise_variance, lidar_return_density, vo_feature_count,
               gps_snr, gps_satellite_count,
               ekf_innovation_imu, ekf_innovation_wheel, ekf_innovation_lidar,
               ekf_innovation_vo, ekf_innovation_gps;
        return vec;
    }
};

/**
 * @brief GBDT tree node for inference
 */
struct GBDTNode {
    int feature_index{-1};
    double threshold{0.0};
    double leaf_value{0.0};
    std::unique_ptr<GBDTNode> left_child;
    std::unique_ptr<GBDTNode> right_child;
    bool is_leaf{false};
    
    double predict(const Eigen::VectorXd& features) const {
        if (is_leaf) {
            return leaf_value;
        }
        if (features[feature_index] <= threshold) {
            return left_child ? left_child->predict(features) : 0.0;
        } else {
            return right_child ? right_child->predict(features) : 0.0;
        }
    }
};

/**
 * @brief GBDT tree model
 */
struct GBDTTree {
    std::unique_ptr<GBDTNode> root;
    double predict(const Eigen::VectorXd& features) const {
        return root ? root->predict(features) : 0.0;
    }
};

/**
 * @brief MLP layer for neural network inference
 */
struct MLPLayer {
    Eigen::MatrixXd weights;
    Eigen::VectorXd bias;
    std::string activation{"relu"};
    
    Eigen::VectorXd forward(const Eigen::VectorXd& input) const {
        Eigen::VectorXd output = weights * input + bias;
        if (activation == "relu") {
            for (int i = 0; i < output.size(); ++i) {
                output[i] = std::max(0.0, output[i]);
            }
        } else if (activation == "linear") {
            // No activation
        }
        return output;
    }
};

/**
 * @brief MLP model
 */
struct MLPModel {
    std::vector<MLPLayer> layers;
    double output_scale{1.0};
    double output_bias{0.0};
    
    Eigen::VectorXd predict(const Eigen::VectorXd& input) const {
        Eigen::VectorXd current = input;
        for (const auto& layer : layers) {
            current = layer.forward(current);
        }
        return current.array() * output_scale + output_bias;
    }
};

/**
 * @brief ML-based sensor weighting engine
 * 
 * Supports both GBDT (Gradient Boosted Decision Trees) and MLP (Multi-Layer Perceptron)
 * models for computing sensor weights based on current sensor features and EKF performance.
 * 
 * Higher weights indicate higher trust in the sensor (lower measurement covariance).
 */
class MLWeightingEngine {
public:
    enum class ModelType {
        GBDT,
        MLP
    };
    
    MLWeightingEngine();
    ~MLWeightingEngine() = default;
    
    /**
     * @brief Load model from JSON file
     * @param json_path Path to JSON model file
     * @return true if loading successful
     */
    bool loadModel(const std::string& json_path);
    
    /**
     * @brief Compute sensor weights from features
     * @param features Current sensor features
     * @return Weight vector [w_imu, w_wheel, w_lidar, w_vo, w_gps]
     */
    Eigen::VectorXd computeWeights(const SensorFeatures& features);
    
    /**
     * @brief Check if model is loaded and ready
     */
    bool isModelLoaded() const { return model_loaded_; }
    
    /**
     * @brief Get model type
     */
    ModelType getModelType() const { return model_type_; }
    
    /**
     * @brief Set weight clipping bounds
     * @param min_weight Minimum allowed weight (default: 0.1)
     * @param max_weight Maximum allowed weight (default: 2.0)
     */
    void setWeightBounds(double min_weight, double max_weight);
    
    /**
     * @brief Get current weight bounds
     */
    std::pair<double, double> getWeightBounds() const {
        return {min_weight_, max_weight_};
    }

private:
    ModelType model_type_{ModelType::GBDT};
    bool model_loaded_{false};
    
    // GBDT model
    std::vector<GBDTTree> gbdt_trees_;
    double gbdt_learning_rate_{0.1};
    double gbdt_init_prediction_{0.0};
    
    // MLP model
    MLPModel mlp_model_;
    
    // Weight bounds
    double min_weight_{0.1};
    double max_weight_{2.0};
    
    // JSON parsing helpers
    bool parseGBDTModel(const std::string& json_content);
    bool parseMLPModel(const std::string& json_content);
    std::unique_ptr<GBDTNode> parseGBDTNode(const std::string& node_json);
    
    // Inference methods
    Eigen::VectorXd predictGBDT(const Eigen::VectorXd& features);
    Eigen::VectorXd predictMLP(const Eigen::VectorXd& features);
    
    // Utility methods
    Eigen::VectorXd normalizeAndClipWeights(const Eigen::VectorXd& raw_weights);
    std::string extractJsonValue(const std::string& json, const std::string& key);
    std::vector<std::string> splitJsonArray(const std::string& array_str);
};

} // namespace axis
