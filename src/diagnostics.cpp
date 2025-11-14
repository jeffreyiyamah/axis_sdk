#include "axis/diagnostics.h"
#include <iostream>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <chrono>

namespace axis {

DiagnosticsPublisher::DiagnosticsPublisher() = default;

DiagnosticsPublisher::DiagnosticsPublisher(const DiagnosticsConfig& config)
    : config_(config) {
}

DiagnosticsPublisher::~DiagnosticsPublisher() {
    shutdown();
}

bool DiagnosticsPublisher::initialize() {
    if (initialized_) {
        return true;
    }
    
    // Initialize file logging
    if (config_.enable_file_logging) {
        log_file_ = std::make_unique<std::ofstream>(config_.log_file_path, std::ios::app);
        if (!log_file_->is_open()) {
            std::cerr << "Failed to open diagnostics log file: " << config_.log_file_path << std::endl;
            return false;
        }
        
        // Write header
        *log_file_ << "\n=== Axis SDK Diagnostics Session Started at " 
                   << getCurrentTimestamp() << " ===\n" << std::flush;
    }
    
    // Start worker thread
    should_shutdown_.store(false);
    worker_thread_ = std::make_unique<std::thread>(&DiagnosticsPublisher::workerLoop, this);
    
    initialized_ = true;
    return true;
}

void DiagnosticsPublisher::shutdown() {
    if (initialized_) {
        should_shutdown_.store(true);
        queue_cv_.notify_all();
        
        if (worker_thread_ && worker_thread_->joinable()) {
            worker_thread_->join();
        }
        
        // Close log file
        if (log_file_ && log_file_->is_open()) {
            *log_file_ << "=== Axis SDK Diagnostics Session Ended at " 
                       << getCurrentTimestamp() << " ===\n" << std::flush;
            log_file_->close();
        }
        
        initialized_ = false;
    }
}

void DiagnosticsPublisher::publishPose(const PoseMessage& pose) {
    if (!initialized_) {
        return;
    }
    
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        latest_pose_ = pose;
    }
    
    if (config_.enable_pose_publishing) {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        message_queue_.push([this, pose]() { publishPoseInternal(pose); });
        queue_cv_.notify_one();
    }
}

void DiagnosticsPublisher::publishQuality(const PoseQuality& quality) {
    if (!initialized_) {
        return;
    }
    
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        latest_quality_ = quality;
    }
    
    if (config_.enable_quality_publishing) {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        message_queue_.push([this, quality]() { publishQualityInternal(quality); });
        queue_cv_.notify_one();
    }
}

void DiagnosticsPublisher::publishHealth(const std::map<std::string, SensorHealth>& health) {
    if (!initialized_) {
        return;
    }
    
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        latest_health_ = health;
    }
    
    if (config_.enable_health_publishing) {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        message_queue_.push([this, health]() { publishHealthInternal(health); });
        queue_cv_.notify_one();
    }
}

void DiagnosticsPublisher::publishMode(OperatingMode mode) {
    if (!initialized_) {
        return;
    }
    
    {
        std::lock_guard<std::mutex> lock(cache_mutex_);
        latest_mode_ = mode;
    }
    
    if (config_.enable_mode_publishing) {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        message_queue_.push([this, mode]() { publishModeInternal(mode); });
        queue_cv_.notify_one();
    }
}

void DiagnosticsPublisher::setConfig(const DiagnosticsConfig& config) {
    config_ = config;
    
    // Reinitialize file logging if path changed
    if (config_.enable_file_logging && log_file_) {
        log_file_->close();
        log_file_ = std::make_unique<std::ofstream>(config_.log_file_path, std::ios::app);
        if (!log_file_->is_open()) {
            std::cerr << "Failed to reopen diagnostics log file: " << config_.log_file_path << std::endl;
        }
    }
}

void DiagnosticsPublisher::workerLoop() {
    using Clock = std::chrono::steady_clock;
    using Duration = Clock::duration;  // <-- This is nanoseconds

    auto publish_interval_seconds = 1.0 / config_.publish_rate;
    auto publish_interval = std::chrono::duration_cast<Duration>(
        std::chrono::duration<double>(publish_interval_seconds)
    );

    auto next_publish = Clock::now();

    while (!should_shutdown_.load()) {
        auto now = Clock::now();

        if (now >= next_publish) {
            processMessages();
            next_publish = now + publish_interval;  // Now same duration type!
        }

        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.wait_until(lock, next_publish);
    }
}

void DiagnosticsPublisher::processMessages() {
    std::queue<std::function<void()>> messages_to_process;
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        messages_to_process = std::move(message_queue_);
        message_queue_ = std::queue<std::function<void()>>();
    }
    
    while (!messages_to_process.empty()) {
        auto message = messages_to_process.front();
        messages_to_process.pop();
        
        try {
            message();
        } catch (const std::exception& e) {
            std::cerr << "Error processing diagnostics message: " << e.what() << std::endl;
        }
    }
}

void DiagnosticsPublisher::publishPoseInternal(const PoseMessage& pose) {
    // Call callback if set
    if (pose_callback_) {
        pose_callback_(pose);
    }
    
    // Log to file
    if (config_.enable_file_logging && log_file_) {
        logToFile(pose);
    }
    
    // Console output
    if (config_.enable_console_output) {
        std::cout << "[POSE] " << poseToString(pose) << std::endl;
    }
}

void DiagnosticsPublisher::publishQualityInternal(const PoseQuality& quality) {
    // Call callback if set
    if (quality_callback_) {
        quality_callback_(quality);
    }
    
    // Log to file
    if (config_.enable_file_logging && log_file_) {
        logToFile(quality);
    }
    
    // Console output
    if (config_.enable_console_output) {
        std::cout << "[QUALITY] " << qualityToString(quality) << std::endl;
    }
}

void DiagnosticsPublisher::publishHealthInternal(const std::map<std::string, SensorHealth>& health) {
    // Call callback if set
    if (health_callback_) {
        health_callback_(health);
    }
    
    // Log to file
    if (config_.enable_file_logging && log_file_) {
        logToFile(health);
    }
    
    // Console output
    if (config_.enable_console_output) {
        std::cout << "[HEALTH] " << healthToString(health) << std::endl;
    }
}

void DiagnosticsPublisher::publishModeInternal(OperatingMode mode) {
    // Call callback if set
    if (mode_callback_) {
        mode_callback_(mode);
    }
    
    // Log to file
    if (config_.enable_file_logging && log_file_) {
        logToFile(mode);
    }
    
    // Console output
    if (config_.enable_console_output) {
        std::cout << "[MODE] " << modeToString(mode) << std::endl;
    }
}

void DiagnosticsPublisher::logToFile(const PoseMessage& pose) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    if (log_file_ && log_file_->is_open()) {
        *log_file_ << "[" << getCurrentTimestamp() << "] POSE: " 
                   << poseToString(pose) << std::flush;
    }
}

void DiagnosticsPublisher::logToFile(const PoseQuality& quality) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    if (log_file_ && log_file_->is_open()) {
        *log_file_ << "[" << getCurrentTimestamp() << "] QUALITY: " 
                   << qualityToString(quality) << std::flush;
    }
}

void DiagnosticsPublisher::logToFile(const std::map<std::string, SensorHealth>& health) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    if (log_file_ && log_file_->is_open()) {
        *log_file_ << "[" << getCurrentTimestamp() << "] HEALTH: " 
                   << healthToString(health) << std::flush;
    }
}

void DiagnosticsPublisher::logToFile(OperatingMode mode) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    if (log_file_ && log_file_->is_open()) {
        *log_file_ << "[" << getCurrentTimestamp() << "] MODE: " 
                   << modeToString(mode) << std::flush;
    }
}

std::string DiagnosticsPublisher::poseToString(const PoseMessage& pose) const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "pos=[" << pose.position.x() << "," << pose.position.y() << "," << pose.position.z() << "] ";
    oss << "ori=[" << pose.orientation.w() << "," << pose.orientation.x() << "," 
        << pose.orientation.y() << "," << pose.orientation.z() << "] ";
    oss << "vel=[" << pose.velocity.x() << "," << pose.velocity.y() << "," << pose.velocity.z() << "] ";
    oss << "conf=" << std::setprecision(3) << pose.confidence << " ";
    oss << "mode=" << static_cast<int>(pose.mode);
    return oss.str();
}

std::string DiagnosticsPublisher::qualityToString(const PoseQuality& quality) const {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);
    oss << "cov_trace=" << quality.covariance_trace << " ";
    oss << "pos_unc=" << quality.position_uncertainty << " ";
    oss << "ori_unc=" << quality.orientation_uncertainty << " ";
    oss << "conf=" << std::setprecision(3) << quality.confidence_score << " ";
    oss << "mode=" << static_cast<int>(quality.mode);
    return oss.str();
}

std::string DiagnosticsPublisher::healthToString(const std::map<std::string, SensorHealth>& health) const {
    std::ostringstream oss;
    for (const auto& [sensor, status] : health) {
        oss << sensor << "=" << static_cast<int>(status) << " ";
    }
    return oss.str();
}

std::string DiagnosticsPublisher::modeToString(OperatingMode mode) const {
    switch (mode) {
        case OperatingMode::NOMINAL: return "NOMINAL";
        case OperatingMode::DEAD_RECKONING: return "DEAD_RECKONING";
        case OperatingMode::LIDAR_ASSIST: return "LIDAR_ASSIST";
        case OperatingMode::RELOCALIZED: return "RELOCALIZED";
        case OperatingMode::FAIL_SAFE: return "FAIL_SAFE";
        default: return "UNKNOWN";
    }
}

std::string DiagnosticsPublisher::getCurrentTimestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::ostringstream oss;
    oss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    oss << "." << std::setfill('0') << std::setw(3) << ms.count();
    return oss.str();
}

// Convenience functions
DiagnosticsConfig createDefaultDiagnosticsConfig() {
    DiagnosticsConfig config;
    config.publish_rate = 10.0;
    config.enable_file_logging = true;
    config.log_file_path = "axis_diagnostics.log";
    config.enable_console_output = false;
    return config;
}

DiagnosticsConfig createHighRateDagnosticsConfig() {
    DiagnosticsConfig config;
    config.publish_rate = 50.0;
    config.enable_file_logging = true;
    config.log_file_path = "axis_diagnostics_highrate.log";
    config.enable_console_output = false;
    return config;
}

DiagnosticsConfig createLowRateDiagnosticsConfig() {
    DiagnosticsConfig config;
    config.publish_rate = 1.0;
    config.enable_file_logging = true;
    config.log_file_path = "axis_diagnostics_lowrate.log";
    config.enable_console_output = true;
    return config;
}

} // namespace axis
