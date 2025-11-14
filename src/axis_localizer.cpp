#include "axis/axis_localizer.h"
#include "axis/imu_handler.h"
#include "axis/gps_handler.h"
#include "axis/lidar_odom_handler.h"
#include "axis/wheel_odom_handler.h"
#include "axis/vo_handler.h"
#include <iostream>
#include <cmath>
#include <algorithm>

namespace axis {

// Constants for GPS to ENU conversion
constexpr double EARTH_RADIUS = 6378137.0; // meters
constexpr double DEG_TO_RAD = M_PI / 180.0;

AxisLocalizer::AxisLocalizer(const std::string& config_path)
    : config_path_(config_path),
      last_ml_update_(std::chrono::steady_clock::now()),
      last_relocalization_attempt_(std::chrono::steady_clock::now()),
      last_diagnostics_update_(std::chrono::steady_clock::now()) {
    
    // Initialize default weights
    current_weights_ = Eigen::VectorXd::Ones(5);
    
    // Initialize sensor health
    sensor_health_["imu"] = SensorHealth::UNAVAILABLE;
    sensor_health_["gps"] = SensorHealth::UNAVAILABLE;
    sensor_health_["lidar_odom"] = SensorHealth::UNAVAILABLE;
    sensor_health_["wheel_odom"] = SensorHealth::UNAVAILABLE;
    sensor_health_["visual_odom"] = SensorHealth::UNAVAILABLE;
}

AxisLocalizer::~AxisLocalizer() {
    stop();
    cleanup();
}

bool AxisLocalizer::initialize() {
    if (initialized_) {
        return true;
    }
    
    std::cout << "Initializing AxisLocalizer..." << std::endl;
    
    // Load configuration
    if (!loadConfiguration()) {
        std::cerr << "Failed to load configuration" << std::endl;
        return false;
    }
    
    // Initialize components
    try {
        // EKF State
        ekf_ = std::make_unique<EKFState>();
        
        // Mode Manager
        mode_manager_ = std::make_unique<ModeManager>();
        
        // ML Weighting Engine
        ml_weighting_ = std::make_unique<MLWeightingEngine>();
        std::string ml_model_path = config_parser_->get<std::string>("ml_weighting.model_path", "config/ml_model.json");
        if (!ml_weighting_->loadModel(ml_model_path)) {
            std::cerr << "Warning: Failed to load ML model, using default weights" << std::endl;
        }
        
        // ---------------------------------------------------------------
        //  Map-Lite relocalizer creation (only if LiDAR odometry is enabled)
        // ---------------------------------------------------------------
        if (config_parser_ && config_parser_->get<bool>("sensors.lidar_odom", false)) {
            MapLiteConfig maplite_config;

            maplite_config.submap_radius = config_parser_->get<double>("maplite.submap_radius", 50.0);
            maplite_config.icp_max_correspondence_distance = config_parser_->get<double>("maplite.icp_max_correspondence_distance", 0.5);
            maplite_config.max_scans_in_submap = config_parser_->get<int>   ("maplite.max_scans_in_submap", 100);
            maplite_config.icp_max_iterations = config_parser_->get<int>   ("maplite.icp_max_iterations", 50);
            maplite_config.match_score_threshold = config_parser_->get<double>("maplite.match_score_threshold", 0.7);
            maplite_config.min_time_between_attempts = config_parser_->get<double>("maplite.min_time_between_attempts", 1.0);

            map_lite_relocalizer_ = std::make_unique<MapLiteRelocalizer>(maplite_config);

            if (!map_lite_relocalizer_->initialize()) {
                std::cerr << "Warning: Failed to initialize Map-Lite relocalizer\n";
            }
        }
        
        // Diagnostics
        DiagnosticsConfig diagnostics_config;
        diagnostics_config.publish_rate = config_parser_->getDouble("diagnostics.publish_rate", 10.0);
        diagnostics_config.enable_file_logging = config_parser_->getBool("diagnostics.enable_file_logging", true);
        diagnostics_config.log_file_path = config_parser_->getString("diagnostics.log_file_path", "axis_diagnostics.log");
        
        diagnostics_ = std::make_unique<DiagnosticsPublisher>(diagnostics_config);
        if (!diagnostics_->initialize()) {
            std::cerr << "Warning: Failed to initialize diagnostics publisher" << std::endl;
        }
        
        setupDiagnosticsCallbacks();
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during initialization: " << e.what() << std::endl;
        return false;
    }
    
    initialized_ = true;
    std::cout << "AxisLocalizer initialized successfully" << std::endl;
    return true;
}

bool AxisLocalizer::start() {
    if (!initialized_) {
        std::cerr << "Cannot start: not initialized" << std::endl;
        return false;
    }
    
    if (running_.load()) {
        std::cout << "Already running" << std::endl;
        return true;
    }
    
    std::cout << "Starting AxisLocalizer..." << std::endl;
    
    should_shutdown_.store(false);
    running_.store(true);
    
    // Start worker threads
    main_thread_ = std::make_unique<std::thread>(&AxisLocalizer::mainLoop, this);
    ml_update_thread_ = std::make_unique<std::thread>(&AxisLocalizer::mlUpdateLoop, this);
    
    if (map_lite_relocalizer_) {
        relocalization_thread_ = std::make_unique<std::thread>(&AxisLocalizer::relocalizationLoop, this);
    }
    
    std::cout << "AxisLocalizer started successfully" << std::endl;
    return true;
}

void AxisLocalizer::stop() {
    if (!running_.load()) {
        return;
    }
    
    std::cout << "Stopping AxisLocalizer..." << std::endl;
    
    should_shutdown_.store(true);
    running_.store(false);
    
    // Notify all threads
    main_cv_.notify_all();
    ml_cv_.notify_all();
    relocalization_cv_.notify_all();
    
    // Wait for threads to finish
    if (main_thread_ && main_thread_->joinable()) {
        main_thread_->join();
    }
    
    if (ml_update_thread_ && ml_update_thread_->joinable()) {
        ml_update_thread_->join();
    }
    
    if (relocalization_thread_ && relocalization_thread_->joinable()) {
        relocalization_thread_->join();
    }
    
    // Stop components
    if (map_lite_relocalizer_) {
        map_lite_relocalizer_->shutdown();
    }
    
    if (diagnostics_) {
        diagnostics_->shutdown();
    }
    
    std::cout << "AxisLocalizer stopped" << std::endl;
}

bool AxisLocalizer::reset() {
    std::cout << "Resetting AxisLocalizer..." << std::endl;
    
    // Reset EKF
    if (ekf_) {
        Eigen::VectorXd initial_state = Eigen::VectorXd::Zero(15);
        ekf_->reset(initial_state);
    }
    
    // Reset mode manager
    if (mode_manager_) {
        mode_manager_->reset();
    }
    
    // Clear queues
    {
        std::lock_guard<std::mutex> lock(imu_mutex_);
        std::queue<IMUMeasurement> empty;
        imu_queue_.swap(empty);
    }
    
    {
        std::lock_guard<std::mutex> lock(gps_mutex_);
        std::queue<GPSMeasurement> empty;
        gps_queue_.swap(empty);
    }
    
    // Reset other state
    gps_origin_set_ = false;
    
    std::cout << "AxisLocalizer reset complete" << std::endl;
    return true;
}

// Sensor feed methods
void AxisLocalizer::feedIMU(const Eigen::Vector3d& accel, const Eigen::Vector3d& gyro, double timestamp) {
    if (!running_.load()) {
        return;
    }
    
    IMUMeasurement measurement(accel, gyro, timestamp);
    
    std::lock_guard<std::mutex> lock(imu_mutex_);
    imu_queue_.push(measurement);
    main_cv_.notify_one();
}

void AxisLocalizer::feedGPS(double latitude, double longitude, double altitude, 
                           const Eigen::Matrix3d& covariance, double timestamp) {
    if (!running_.load()) {
        return;
    }
    
    // Set GPS origin on first measurement
    if (!gps_origin_set_) {
        setGPSOrigin(latitude, longitude, altitude);
        gps_origin_set_ = true;
    }
    
    GPSMeasurement measurement(latitude, longitude, altitude, covariance, timestamp);
    
    std::lock_guard<std::mutex> lock(gps_mutex_);
    gps_queue_.push(measurement);
    main_cv_.notify_one();
}

void AxisLocalizer::feedLidarOdom(const Eigen::Isometry3d& delta_pose, 
                                 const axis::Matrix6d& covariance, double timestamp) {
    if (!running_.load()) {
        return;
    }
    
    LidarOdometryMeasurement measurement(delta_pose, covariance, timestamp);
    
    std::lock_guard<std::mutex> lock(lidar_odom_mutex_);
    lidar_odom_queue_.push(measurement);
    main_cv_.notify_one();
}

void AxisLocalizer::feedWheelOdom(double velocity_left, double velocity_right, double timestamp) {
    if (!running_.load()) {
        return;
    }
    
    WheelOdometryMeasurement measurement(velocity_left, velocity_right, timestamp);
    
    std::lock_guard<std::mutex> lock(wheel_odom_mutex_);
    wheel_odom_queue_.push(measurement);
    main_cv_.notify_one();
}

void AxisLocalizer::feedVisualOdom(const Eigen::Isometry3d& delta_pose, 
                                  const axis::Matrix6d& covariance, int feature_count, double timestamp) {
    if (!running_.load()) {
        return;
    }
    
    VisualOdometryMeasurement measurement(delta_pose, covariance, feature_count, timestamp);
    
    std::lock_guard<std::mutex> lock(visual_odom_mutex_);
    visual_odom_queue_.push(measurement);
    main_cv_.notify_one();
}

void AxisLocalizer::feedLidarScan(
#ifdef WITH_PCL
    pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud,
#else
    const std::vector<Eigen::Vector3d>& points,
#endif
    double timestamp) {
    if (!running_.load() || !map_lite_relocalizer_) {
        return;
    }
    
    LidarScan scan;
    scan.timestamp = timestamp;
    
    // Get current pose for scan
    auto current_pose = getPose();
    scan.position = current_pose.position;
    scan.orientation = current_pose.orientation;
    
#ifdef WITH_PCL
    scan.pointcloud = pointcloud;
#else
    scan.points = points;
#endif
    
    std::lock_guard<std::mutex> lock(lidar_scan_mutex_);
    lidar_scan_queue_.push(scan);
    relocalization_cv_.notify_one();
}

// Output methods
PoseMessage AxisLocalizer::getPose() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return current_pose_;
}

OperatingMode AxisLocalizer::getMode() const {
    if (mode_manager_) {
        return mode_manager_->getCurrentMode();
    }
    return OperatingMode::NOMINAL;
}

std::map<std::string, SensorHealth> AxisLocalizer::getSensorHealth() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return sensor_health_;
}

Eigen::VectorXd AxisLocalizer::getMLWeights() const {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return current_weights_;
}

std::optional<RelocalizationResult> AxisLocalizer::getLastRelocalizationResult() const {
    if (map_lite_relocalizer_) {
        return map_lite_relocalizer_->getLastResult();
    }
    return std::nullopt;
}

// Private methods
void AxisLocalizer::mainLoop() {
    auto last_update = std::chrono::steady_clock::now();
    
    while (!should_shutdown_.load()) {
        // Process sensor measurements
        processIMUMeasurements();
        processGPSMeasurements();
        processLidarOdometryMeasurements();
        processWheelOdometryMeasurements();
        processVisualOdometryMeasurements();
        
        // Update operating mode
        updateOperatingMode();
        
        // Update diagnostics
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - last_diagnostics_update_);
        if (elapsed.count() >= diagnostics_interval_) {
            updateDiagnostics();
            last_diagnostics_update_ = now;
        }
        
        // Sleep for short interval
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

void AxisLocalizer::mlUpdateLoop(){
    while (!should_shutdown_.load()) {
        std::unique_lock<std::mutex> lock(ml_mutex_);          // <-- use ml_mutex_
        ml_cv_.wait_for(lock, std::chrono::duration<double>(ml_update_interval_));

        if (should_shutdown_.load()) break;

        updateMLWeights();
    }
}

void AxisLocalizer::relocalizationLoop(){
    if (!map_lite_relocalizer_) return;

    while (!should_shutdown_.load()) {
        std::unique_lock<std::mutex> lock(relocalization_mutex_); 
        relocalization_cv_.wait_for(lock,
                                   std::chrono::duration<double>(relocalization_interval_));

        if (should_shutdown_.load()) break;

        processLidarScans();
        attemptRelocalization();
    }
}

void AxisLocalizer::processIMUMeasurements() {
    std::lock_guard<std::mutex> lock(imu_mutex_);
    
    while (!imu_queue_.empty()) {
        auto measurement = imu_queue_.front();
        imu_queue_.pop();
        
        if (measurement.valid && ekf_) {
            // EKF predict step
            auto current_state = ekf_->getState();
            double dt = 0.01; // Assume 100Hz IMU, should be computed from timestamps
            
            ekf_->predict(measurement.accel, measurement.gyro, dt);
            
            // Update current pose
            auto new_state = ekf_->getState();
            {
                std::lock_guard<std::mutex> state_lock(state_mutex_);
                current_pose_.timestamp = measurement.timestamp;
                current_pose_.position = new_state.position;
                current_pose_.orientation = new_state.orientation;
                current_pose_.velocity = new_state.velocity;
                current_pose_.mode = getMode();
            }
            
            // Update sensor health
            sensor_health_["imu"] = SensorHealth::ONLINE;
        }
    }
}

void AxisLocalizer::processGPSMeasurements() {
    std::lock_guard<std::mutex> lock(gps_mutex_);
    
    while (!gps_queue_.empty()) {
        auto measurement = gps_queue_.front();
        gps_queue_.pop();
        
        if (measurement.valid && ekf_ && gps_origin_set_) {
            // Convert GPS to ENU coordinates
            Eigen::Vector3d position_enu = gpsToENU(measurement.latitude, measurement.longitude, measurement.altitude);
            
            // EKF update step
            Eigen::VectorXd z(3);
            z << position_enu.x(), position_enu.y(), position_enu.z();
            
            Eigen::MatrixXd H = Eigen::MatrixXd::Identity(3, 15);
            Eigen::MatrixXd R = measurement.covariance;
            
            // Apply ML weighting
            if (current_weights_.size() >= 5) {
                R = R / current_weights_[4]; // GPS weight index 4
            }
            
            ekf_->update(z, H, R);
            
            sensor_health_["gps"] = SensorHealth::ONLINE;
        }
    }
}

void AxisLocalizer::processLidarOdometryMeasurements() {
    std::lock_guard<std::mutex> lock(lidar_odom_mutex_);
    
    while (!lidar_odom_queue_.empty()) {
        auto measurement = lidar_odom_queue_.front();
        lidar_odom_queue_.pop();
        
        if (measurement.valid && ekf_) {
            // Extract pose and velocity from delta pose
            Eigen::Vector3d position_delta = measurement.delta_pose.translation();
            Eigen::Quaterniond orientation_delta = Eigen::Quaterniond(measurement.delta_pose.linear());
            
            // Create measurement vector (simplified - would need proper odometry model)
            Eigen::VectorXd z(6);
            z << position_delta.x(), position_delta.y(), position_delta.z(),
                 orientation_delta.x(), orientation_delta.y(), orientation_delta.z();
            
            Eigen::MatrixXd H = Eigen::MatrixXd::Identity(6, 15);
            Eigen::MatrixXd R = measurement.covariance;
            
            // Apply ML weighting
            if (current_weights_.size() >= 5) {
                R = R / current_weights_[2]; // LiDAR weight index 2
            }
            
            ekf_->update(z, H, R);
            
            sensor_health_["lidar_odom"] = SensorHealth::ONLINE;
        }
    }
}

void AxisLocalizer::processWheelOdometryMeasurements() {
    std::lock_guard<std::mutex> lock(wheel_odom_mutex_);
    
    while (!wheel_odom_queue_.empty()) {
        auto measurement = wheel_odom_queue_.front();
        wheel_odom_queue_.pop();
        
        if (measurement.valid && ekf_) {
            // Simple wheel odometry update (would need proper kinematic model)
            double linear_velocity = (measurement.velocity_left + measurement.velocity_right) / 2.0;
            
            Eigen::VectorXd z(3);
            z << linear_velocity, 0.0, 0.0; // Simplified velocity measurement
            
            Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, 15);
            H(0, 3) = 1.0; // Velocity x component
            
            Eigen::MatrixXd R = Eigen::MatrixXd::Identity(3, 3) * 0.1; // Simple covariance
            
            // Apply ML weighting
            if (current_weights_.size() >= 5) {
                R = R / current_weights_[1]; // Wheel weight index 1
            }
            
            ekf_->update(z, H, R);
            
            sensor_health_["wheel_odom"] = SensorHealth::ONLINE;
        }
    }
}

void AxisLocalizer::processVisualOdometryMeasurements() {
    std::lock_guard<std::mutex> lock(visual_odom_mutex_);
    
    while (!visual_odom_queue_.empty()) {
        auto measurement = visual_odom_queue_.front();
        visual_odom_queue_.pop();
        
        if (measurement.valid && ekf_) {
            // Similar to LiDAR odometry processing
            Eigen::Vector3d position_delta = measurement.delta_pose.translation();
            
            Eigen::VectorXd z(6);
            z << position_delta.x(), position_delta.y(), position_delta.z(),
                 0.0, 0.0, 0.0; // Simplified orientation
            
            Eigen::MatrixXd H = Eigen::MatrixXd::Identity(6, 15);
            Eigen::MatrixXd R = measurement.covariance;
            
            // Apply ML weighting
            if (current_weights_.size() >= 5) {
                R = R / current_weights_[3]; // VO weight index 3
            }
            
            ekf_->update(z, H, R);
            
            sensor_health_["visual_odom"] = SensorHealth::ONLINE;
        }
    }
}

void AxisLocalizer::processLidarScans() {
    std::lock_guard<std::mutex> lock(lidar_scan_mutex_);
    
    while (!lidar_scan_queue_.empty()) {
        auto scan = lidar_scan_queue_.front();
        lidar_scan_queue_.pop();
        
        if (map_lite_relocalizer_) {
            map_lite_relocalizer_->addScan(scan);
        }
    }
}

void AxisLocalizer::updateMLWeights() {
    if (!ml_weighting_ || !ml_weighting_->isModelLoaded()) {
        return;
    }
    
    // Extract current sensor features
    SensorFeatures features = extractSensorFeatures();
    
    // Compute new weights
    Eigen::VectorXd new_weights = ml_weighting_->computeWeights(features);
    
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        current_weights_ = new_weights;
    }
}

SensorFeatures AxisLocalizer::extractSensorFeatures() const {
    SensorFeatures features;
    
    // Extract features from current sensor states
    // This is a simplified implementation - would need proper feature extraction
    
    features.imu_noise_variance = 0.01; // Placeholder
    features.lidar_return_density = 1000.0; // Placeholder
    features.vo_feature_count = 50.0; // Placeholder
    features.gps_snr = 30.0; // Placeholder
    features.gps_satellite_count = 8.0; // Placeholder
    
    // EKF innovation residuals (simplified)
    if (ekf_) {
        auto covariance = ekf_->getCovariance();
        features.ekf_innovation_imu = covariance.block<3, 3>(0, 0).trace();
        features.ekf_innovation_wheel = covariance.block<3, 3>(3, 3).trace();
        features.ekf_innovation_lidar = covariance.block<3, 3>(6, 6).trace();
        features.ekf_innovation_vo = covariance.block<3, 3>(9, 9).trace();
        features.ekf_innovation_gps = covariance.block<3, 3>(12, 12).trace();
    }
    
    return features;
}

void AxisLocalizer::attemptRelocalization() {
    if (!map_lite_relocalizer_ || !map_lite_relocalizer_->isActive()) {
        return;
    }
    
    auto result = map_lite_relocalizer_->attemptRelocalization();
    if (result && result->success) {
        handleRelocalizationResult(*result);
    }
}

void AxisLocalizer::handleRelocalizationResult(const RelocalizationResult& result) {
    if (!ekf_) {
        return;
    }
    
    // Apply relocalization as GPS-like update
    Eigen::VectorXd z(6);
    z << result.transform.translation().x(),
         result.transform.translation().y(),
         result.transform.translation().z(),
         0.0, 0.0, 0.0; // Simplified orientation
    
    Eigen::MatrixXd H = Eigen::MatrixXd::Identity(6, 15);
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(6, 6) * 0.01; // Low covariance for high confidence
    
    ekf_->update(z, H, R);
    
    // Update mode
    if (mode_manager_) {
        setMode(OperatingMode::RELOCALIZED);
    }
}

void AxisLocalizer::updateOperatingMode() {
    if (!mode_manager_) {
        return;
    }
    
    // Check sensor health and update mode
    checkSensorHealth();
    
    // Simple mode logic - would be more sophisticated in practice
    if (sensor_health_["gps"] == SensorHealth::ONLINE) {
        setMode(OperatingMode::NOMINAL);
    } else if (sensor_health_["lidar_odom"] == SensorHealth::ONLINE) {
        setMode(OperatingMode::LIDAR_ASSIST);
        if (map_lite_relocalizer_) {
            map_lite_relocalizer_->setActive(true);
        }
    } else {
        setMode(OperatingMode::DEAD_RECKONING);
        if (map_lite_relocalizer_) {
            map_lite_relocalizer_->setActive(false);
        }
    }
}

void AxisLocalizer::checkSensorHealth() {
    // Simple health check based on measurement age
    auto now = std::chrono::steady_clock::now();
    const double max_age = 1.0; // seconds
    
    // Check each sensor - this is simplified
    for (auto& [sensor, health] : sensor_health_) {
        if (health == SensorHealth::ONLINE) {
            // Would check measurement timestamps here
            health = SensorHealth::ONLINE; // Placeholder
        }
    }
}

void AxisLocalizer::updateDiagnostics() {
    if (!diagnostics_) {
        return;
    }
    
    // Update pose
    auto pose = getPose();
    diagnostics_->publishPose(pose);
    
    // Update quality
    auto quality = computePoseQuality();
    diagnostics_->publishQuality(quality);
    
    // Update health
    diagnostics_->publishHealth(sensor_health_);
    
    // Update mode
    diagnostics_->publishMode(getMode());
}

PoseQuality AxisLocalizer::computePoseQuality() const {
    PoseQuality quality;
    
    if (ekf_) {
        auto covariance = ekf_->getCovariance();
        quality.covariance_trace = covariance.trace();
        quality.position_uncertainty = covariance.block<3, 3>(0, 0).trace();
        quality.orientation_uncertainty = covariance.block<3, 3>(6, 6).trace();
        
        // Compute confidence score (simplified)
        quality.confidence_score = std::max(0.0, std::min(1.0, 1.0 - quality.covariance_trace / 10.0));
    }
    
    quality.mode = getMode();
    quality.timestamp = current_pose_.timestamp;
    
    return quality;
}

Eigen::Vector3d AxisLocalizer::gpsToENU(double latitude, double longitude, double altitude) const {
    // Simple GPS to ENU conversion (approximate)
    double lat_rad = latitude * DEG_TO_RAD;
    double lon_rad = longitude * DEG_TO_RAD;
    
    double lat_origin_rad = gps_origin_.x() * DEG_TO_RAD;
    double lon_origin_rad = gps_origin_.y() * DEG_TO_RAD;
    
    double dlat = (latitude - gps_origin_.x()) * DEG_TO_RAD;
    double dlon = (longitude - gps_origin_.y()) * DEG_TO_RAD;
    double dalt = altitude - gps_origin_.z();
    
    double east = EARTH_RADIUS * dlon * std::cos(lat_origin_rad);
    double north = EARTH_RADIUS * dlat;
    double up = dalt;
    
    return Eigen::Vector3d(east, north, up);
}

void AxisLocalizer::setGPSOrigin(double latitude, double longitude, double altitude) {
    gps_origin_ = Eigen::Vector3d(latitude, longitude, altitude);
    std::cout << "GPS origin set to: " << latitude << ", " << longitude << ", " << altitude << std::endl;
}

axis::Matrix6d AxisLocalizer::isometryToMatrix6d(const Eigen::Isometry3d& pose) const {
    axis::Matrix6d matrix = axis::Matrix6d::Identity();
    
    // Position part
    matrix.block<3, 3>(0, 0) = pose.linear();
    matrix.block<3, 1>(0, 3) = pose.translation();
    
    // Orientation part (simplified)
    matrix.block<3, 3>(3, 3) = pose.linear();
    
    return matrix;
}

Eigen::Isometry3d AxisLocalizer::matrix6dToIsometry(const axis::Matrix6d& matrix) const {
    Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
    
    pose.linear() = matrix.block<3, 3>(0, 0);
    pose.translation() = matrix.block<3, 1>(0, 3);
    
    return pose;
}

bool AxisLocalizer::loadConfiguration() {
    try {
        config_parser_ = std::make_unique<ConfigParser>(config_path_);
        
        // Load timing parameters
        ml_update_interval_ = config_parser_->getDouble("ml_weighting.update_interval", 2.0);
        relocalization_interval_ = config_parser_->getDouble("maplite.min_time_between_attempts", 2.0);
        diagnostics_interval_ = 1.0 / config_parser_->getDouble("diagnostics.publish_rate", 10.0);
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load configuration: " << e.what() << std::endl;
        return false;
    }
}

void AxisLocalizer::setupDiagnosticsCallbacks() {
    if (!diagnostics_) {
        return;
    }
    
    // Set up callbacks for different topics
    diagnostics_->setPoseCallback([](const PoseMessage& pose) {
        // Could publish to ROS topics here
    });
    
    diagnostics_->setQualityCallback([](const PoseQuality& quality) {
        // Could publish quality metrics
    });
    
    diagnostics_->setHealthCallback([](const std::map<std::string, SensorHealth>& health) {
        // Could publish sensor health
    });
    
    diagnostics_->setModeCallback([](OperatingMode mode) {
        // Could publish mode changes
    });
}

void AxisLocalizer::cleanup() {
    ekf_.reset();
    mode_manager_.reset();
    ml_weighting_.reset();
    map_lite_relocalizer_.reset();
    diagnostics_.reset();
    config_parser_.reset();
}

} // namespace axis
