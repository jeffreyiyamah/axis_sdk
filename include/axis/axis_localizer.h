#pragma once
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <string>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <map>
#include <queue>
#include <condition_variable>
#include <functional>
#include "axis/types.h"
#include "axis/ekf_state.h"
#include "axis/config_parser.h"
#include "axis/mode_manager.h"
#include "axis/ml_weighting.h"
#include "axis/map_lite_relocalizer.h"
#include "axis/diagnostics.h"

#ifdef WITH_PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#endif

namespace axis {

/**
 * @brief Sensor measurement structures
 */
struct IMUMeasurement {
    Eigen::Vector3d accel = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyro = Eigen::Vector3d::Zero();
    double timestamp{0.0};
    bool valid{false};
    
    IMUMeasurement() = default;
    IMUMeasurement(const Eigen::Vector3d& a, const Eigen::Vector3d& g, double ts, bool v = true)
        : accel(a), gyro(g), timestamp(ts), valid(v) {}
};

struct GPSMeasurement {
    double latitude{0.0};
    double longitude{0.0};
    double altitude{0.0};
    Eigen::Matrix3d covariance = Eigen::Matrix3d::Identity();
    double timestamp{0.0};
    bool valid{false};
    
    GPSMeasurement() = default;
    GPSMeasurement(double lat, double lon, double alt, const Eigen::Matrix3d& cov, double ts, bool v = true)
        : latitude(lat), longitude(lon), altitude(alt), covariance(cov), timestamp(ts), valid(v) {}
};

struct LidarOdometryMeasurement {
    Eigen::Isometry3d delta_pose = Eigen::Isometry3d::Identity();
    axis::Matrix6d covariance = axis::Matrix6d::Identity();
    double timestamp{0.0};
    bool valid{false};
    
    LidarOdometryMeasurement() = default;
    LidarOdometryMeasurement(const Eigen::Isometry3d& pose, const axis::Matrix6d& cov, double ts, bool v = true)
        : delta_pose(pose), covariance(cov), timestamp(ts), valid(v) {}
};

struct WheelOdometryMeasurement {
    double velocity_left{0.0};
    double velocity_right{0.0};
    double timestamp{0.0};
    bool valid{false};
    
    WheelOdometryMeasurement() = default;
    WheelOdometryMeasurement(double v_left, double v_right, double ts, bool v = true)
        : velocity_left(v_left), velocity_right(v_right), timestamp(ts), valid(v) {}
};

struct VisualOdometryMeasurement {
    Eigen::Isometry3d delta_pose = Eigen::Isometry3d::Identity();
    axis::Matrix6d covariance = axis::Matrix6d::Identity();
    int feature_count{0};
    double timestamp{0.0};
    bool valid{false};
    
    VisualOdometryMeasurement() = default;
    VisualOdometryMeasurement(const Eigen::Isometry3d& pose, const axis::Matrix6d& cov, int features, double ts, bool v = true)
        : delta_pose(pose), covariance(cov), feature_count(features), timestamp(ts), valid(v) {}
};

/**
 * @brief Main Axis localizer class that integrates all components
 * 
 * This class provides the main interface for the Axis SDK, integrating
 * EKF state estimation, sensor handling, mode management, ML-based weighting,
 * Map-Lite relocalization, and diagnostics publishing.
 */
class AxisLocalizer {
public:
    /**
     * @brief Constructor
     * @param config_path Path to configuration YAML file
     */
    explicit AxisLocalizer(const std::string& config_path);
    
    /**
     * @brief Destructor
     */
    ~AxisLocalizer();
    
    /**
     * @brief Initialize the localizer
     * @return true if initialization successful
     */
    bool initialize();
    
    /**
     * @brief Start the localizer (begin processing)
     * @return true if start successful
     */
    bool start();
    
    /**
     * @brief Stop the localizer
     */
    void stop();
    
    /**
     * @brief Reset the localizer (reinitialize filter)
     * @return true if reset successful
     */
    bool reset();
    
    /**
     * @brief Check if localizer is initialized
     */
    bool isInitialized() const { return initialized_; }
    
    /**
     * @brief Check if localizer is running
     */
    bool isRunning() const { return running_.load(); }
    
    // Sensor feed methods
    /**
     * @brief Feed IMU measurement
     * @param accel Linear acceleration (m/s^2)
     * @param gyro Angular velocity (rad/s)
     * @param timestamp Measurement timestamp
     */
    void feedIMU(const Eigen::Vector3d& accel, const Eigen::Vector3d& gyro, double timestamp);
    
    /**
     * @brief Feed GPS measurement
     * @param latitude Latitude (degrees)
     * @param longitude Longitude (degrees)
     * @param altitude Altitude (meters)
     * @param covariance Position covariance matrix (3x3)
     * @param timestamp Measurement timestamp
     */
    void feedGPS(double latitude, double longitude, double altitude, 
                const Eigen::Matrix3d& covariance, double timestamp);
    
    /**
     * @brief Feed LiDAR odometry measurement
     * @param delta_pose Delta pose transform
     * @param covariance Pose covariance matrix (6x6)
     * @param timestamp Measurement timestamp
     */
    void feedLidarOdom(const Eigen::Isometry3d& delta_pose, 
                      const axis::Matrix6d& covariance, double timestamp);
    
    /**
     * @brief Feed wheel odometry measurement
     * @param velocity_left Left wheel velocity (m/s)
     * @param velocity_right Right wheel velocity (m/s)
     * @param timestamp Measurement timestamp
     */
    void feedWheelOdom(double velocity_left, double velocity_right, double timestamp);
    
    /**
     * @brief Feed visual odometry measurement
     * @param delta_pose Delta pose transform
     * @param covariance Pose covariance matrix (6x6)
     * @param feature_count Number of visual features tracked
     * @param timestamp Measurement timestamp
     */
    void feedVisualOdom(const Eigen::Isometry3d& delta_pose, 
                       const axis::Matrix6d& covariance, int feature_count, double timestamp);
    
    /**
     * @brief Feed LiDAR scan for Map-Lite relocalization
     * @param pointcloud Point cloud data
     * @param timestamp Scan timestamp
     */
    void feedLidarScan(
#ifdef WITH_PCL
        pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud,
#else
        const std::vector<Eigen::Vector3d>& points,
#endif
        double timestamp);
    
    // Output methods
    /**
     * @brief Get current pose estimate
     * @return Current pose message
     */
    PoseMessage getPose() const;
    
    /**
     * @brief Get current operating mode
     * @return Current operating mode
     */
    OperatingMode getMode() const;
    
    /**
     * @brief Get sensor health status
     * @return Map of sensor names to health status
     */
    std::map<std::string, SensorHealth> getSensorHealth() const;
    
    /**
     * @brief Get current ML weights
     * @return Weight vector [w_imu, w_wheel, w_lidar, w_vo, w_gps]
     */
    Eigen::VectorXd getMLWeights() const;
    
    /**
     * @brief Get last relocalization result
     * @return Optional relocalization result
     */
    std::optional<RelocalizationResult> getLastRelocalizationResult() const;

private:
    // Configuration
    std::string config_path_;
    bool initialized_{false};
    std::atomic<bool> running_{false};
    std::atomic<bool> should_shutdown_{false};
    
    // Components
    std::unique_ptr<ConfigParser> config_parser_;
    std::unique_ptr<EKFState> ekf_;
    std::unique_ptr<ModeManager> mode_manager_;
    std::unique_ptr<MLWeightingEngine> ml_weighting_;
    std::unique_ptr<MapLiteRelocalizer> map_lite_relocalizer_;
    std::unique_ptr<DiagnosticsPublisher> diagnostics_;
    std::mutex ml_mutex_;
    std::mutex relocalization_mutex_;

    
    // Sensor queues
    std::mutex imu_mutex_;
    std::queue<IMUMeasurement> imu_queue_;
    
    std::mutex gps_mutex_;
    std::queue<GPSMeasurement> gps_queue_;
    
    std::mutex lidar_odom_mutex_;
    std::queue<LidarOdometryMeasurement> lidar_odom_queue_;
    
    std::mutex wheel_odom_mutex_;
    std::queue<WheelOdometryMeasurement> wheel_odom_queue_;
    
    std::mutex visual_odom_mutex_;
    std::queue<VisualOdometryMeasurement> visual_odom_queue_;
    
    std::mutex lidar_scan_mutex_;
    std::queue<LidarScan> lidar_scan_queue_;
    
    // Threading
    std::unique_ptr<std::thread> main_thread_;
    std::unique_ptr<std::thread> ml_update_thread_;
    std::unique_ptr<std::thread> relocalization_thread_;
    std::condition_variable main_cv_;
    std::condition_variable ml_cv_;
    std::condition_variable relocalization_cv_;
    
    // Timing
    std::chrono::steady_clock::time_point last_ml_update_;
    std::chrono::steady_clock::time_point last_relocalization_attempt_;
    std::chrono::steady_clock::time_point last_diagnostics_update_;
    double ml_update_interval_{2.0};      // seconds
    double relocalization_interval_{2.0}; // seconds
    double diagnostics_interval_{0.1};    // seconds (10 Hz)
    
    // State
    mutable std::mutex state_mutex_;
    PoseMessage current_pose_;
    Eigen::VectorXd current_weights_;
    std::map<std::string, SensorHealth> sensor_health_;
    
    // Reference point for GPS (origin)
    Eigen::Vector3d gps_origin_ = Eigen::Vector3d::Zero();
    bool gps_origin_set_{false};
    
    // Private methods
    void mainLoop();
    void mlUpdateLoop();
    void relocalizationLoop();
    
    // Processing methods
    void processIMUMeasurements();
    void processGPSMeasurements();
    void processLidarOdometryMeasurements();
    void processWheelOdometryMeasurements();
    void processVisualOdometryMeasurements();
    void processLidarScans();
    
    // ML weighting
    void updateMLWeights();
    SensorFeatures extractSensorFeatures() const;
    void applyMLWeightsToEKF();
    
    // Relocalization
    void attemptRelocalization();
    void handleRelocalizationResult(const RelocalizationResult& result);
    
    // Mode management
    
    void updateOperatingMode();
    void checkSensorHealth();
    
    // Diagnostics
    void updateDiagnostics();
    PoseQuality computePoseQuality() const;
    
    // Utility methods
    Eigen::Vector3d gpsToENU(double latitude, double longitude, double altitude) const;
    void setGPSOrigin(double latitude, double longitude, double altitude);
    axis::Matrix6d isometryToMatrix6d(const Eigen::Isometry3d& pose) const;
    Eigen::Isometry3d matrix6dToIsometry(const axis::Matrix6d& matrix) const;
    
    // Configuration parsing
    bool loadConfiguration();
    void setupDiagnosticsCallbacks();
    
    // Cleanup
    void cleanup();

    inline void setMode(OperatingMode m) {
    if (mode_manager_) {
        mode_manager_->requestTransition(m);  // Now valid!
    }
}
};

} // namespace axis
