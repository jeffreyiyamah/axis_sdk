#pragma once
#include <Eigen/Dense>
#include <string>
#include <fstream>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <queue>
#include <condition_variable>
#include <functional>
#include <map>
#include "axis/types.h"

namespace axis {

/**
 * @brief Pose quality metrics
 */
struct PoseQuality {
    double covariance_trace{0.0};
    double position_uncertainty{0.0};
    double orientation_uncertainty{0.0};
    double confidence_score{0.0};
    OperatingMode mode{OperatingMode::NOMINAL};
    double timestamp{0.0};
    
    PoseQuality() = default;
    PoseQuality(double trace, double pos_unc, double ori_unc, double conf, 
                OperatingMode m, double ts)
        : covariance_trace(trace), position_uncertainty(pos_unc), 
          orientation_uncertainty(ori_unc), confidence_score(conf), 
          mode(m), timestamp(ts) {}
};

/**
 * @brief Diagnostics configuration
 */
struct DiagnosticsConfig {
    double publish_rate{10.0};           // Hz
    bool enable_file_logging{true};
    std::string log_file_path{"axis_diagnostics.log"};
    bool enable_console_output{false};
    bool enable_pose_publishing{true};
    bool enable_quality_publishing{true};
    bool enable_health_publishing{true};
    bool enable_mode_publishing{true};
    
    DiagnosticsConfig() = default;
};

/**
 * @brief Callback types for different diagnostics topics
 */
using PoseCallback = std::function<void(const PoseMessage&)>;
using QualityCallback = std::function<void(const PoseQuality&)>;
using HealthCallback = std::function<void(const std::map<std::string, SensorHealth>&)>;
using ModeCallback = std::function<void(OperatingMode)>;

/**
 * @brief Diagnostics publisher for Axis SDK
 * 
 * Handles publishing of pose, quality metrics, sensor health, and mode information.
 * Supports both file logging and callback-based publishing for integration with
 * ROS or other middleware.
 */
class DiagnosticsPublisher {
public:
    DiagnosticsPublisher();
    explicit DiagnosticsPublisher(const DiagnosticsConfig& config);
    ~DiagnosticsPublisher();
    
    /**
     * @brief Initialize the diagnostics publisher
     * @return true if initialization successful
     */
    bool initialize();
    
    /**
     * @brief Shutdown the diagnostics publisher
     */
    void shutdown();
    
    /**
     * @brief Publish pose message
     * @param pose Current pose estimate
     */
    void publishPose(const PoseMessage& pose);
    
    /**
     * @brief Publish pose quality metrics
     * @param quality Pose quality information
     */
    void publishQuality(const PoseQuality& quality);
    
    /**
     * @brief Publish sensor health status
     * @param health Map of sensor names to health status
     */
    void publishHealth(const std::map<std::string, SensorHealth>& health);
    
    /**
     * @brief Publish current operating mode
     * @param mode Current operating mode
     */
    void publishMode(OperatingMode mode);
    
    /**
     * @brief Set callback for pose messages
     */
    void setPoseCallback(PoseCallback callback) { pose_callback_ = callback; }
    
    /**
     * @brief Set callback for quality metrics
     */
    void setQualityCallback(QualityCallback callback) { quality_callback_ = callback; }
    
    /**
     * @brief Set callback for sensor health
     */
    void setHealthCallback(HealthCallback callback) { health_callback_ = callback; }
    
    /**
     * @brief Set callback for mode changes
     */
    void setModeCallback(ModeCallback callback) { mode_callback_ = callback; }
    
    /**
     * @brief Get configuration
     */
    const DiagnosticsConfig& getConfig() const { return config_; }
    
    /**
     * @brief Set configuration
     */
    void setConfig(const DiagnosticsConfig& config);
    
    /**
     * @brief Check if publisher is initialized
     */
    bool isInitialized() const { return initialized_; }

private:
    DiagnosticsConfig config_;
    bool initialized_{false};
    std::atomic<bool> should_shutdown_{false};
    
    // Threading
    std::unique_ptr<std::thread> worker_thread_;
    std::mutex queue_mutex_;
    std::queue<std::function<void()>> message_queue_;
    std::condition_variable queue_cv_;
    
    // File logging
    std::unique_ptr<std::ofstream> log_file_;
    std::mutex log_mutex_;
    
    // Callbacks
    PoseCallback pose_callback_;
    QualityCallback quality_callback_;
    HealthCallback health_callback_;
    ModeCallback mode_callback_;
    
    // Cached latest values
    PoseMessage latest_pose_;
    PoseQuality latest_quality_;
    std::map<std::string, SensorHealth> latest_health_;
    OperatingMode latest_mode_{OperatingMode::NOMINAL};
    mutable std::mutex cache_mutex_;
    
    // Private methods
    void workerLoop();
    void processMessages();
    
    // Logging methods
    void logToFile(const PoseMessage& pose);
    void logToFile(const PoseQuality& quality);
    void logToFile(const std::map<std::string, SensorHealth>& health);
    void logToFile(OperatingMode mode);
    
    // Utility methods
    std::string poseToString(const PoseMessage& pose) const;
    std::string qualityToString(const PoseQuality& quality) const;
    std::string healthToString(const std::map<std::string, SensorHealth>& health) const;
    std::string modeToString(OperatingMode mode) const;
    std::string getCurrentTimestamp() const;
    
    // Publishing methods (called from worker thread)
    void publishPoseInternal(const PoseMessage& pose);
    void publishQualityInternal(const PoseQuality& quality);
    void publishHealthInternal(const std::map<std::string, SensorHealth>& health);
    void publishModeInternal(OperatingMode mode);
};

/**
 * @brief Convenience functions for creating diagnostics
 */
DiagnosticsConfig createDefaultDiagnosticsConfig();
DiagnosticsConfig createHighRateDagnosticsConfig();  // 50Hz publishing
DiagnosticsConfig createLowRateDiagnosticsConfig();   // 1Hz publishing

} // namespace axis
