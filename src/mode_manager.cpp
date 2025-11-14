#include "mode_manager.h"
#include <algorithm>
#include <sstream>
#include <iomanip>
#include <iostream>

namespace axis {

ModeManager::ModeManager(OperatingMode initial_mode)
    : current_mode_(initial_mode),
      mode_entry_time_(0.0),
      on_enter_mode_(nullptr),
      on_exit_mode_(nullptr) {
    // Initialize sensor status
    sensor_status_["IMU"] = true;
    sensor_status_["GPS"] = true;
    sensor_status_["LiDAR"] = false;
    sensor_status_["WheelOdom"] = false;
    sensor_status_["LoopClosure"] = false;
}

void ModeManager::updateSensorStatus(const std::string& sensor_name, bool is_healthy) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    bool previous_status = sensor_status_[sensor_name];
    sensor_status_[sensor_name] = is_healthy;
    
    // Track recovery time for hysteresis - but DON'T set to 0.0!
    // Instead, remove from map so evaluateTransition will add it with proper time
    if (!previous_status && is_healthy) {
        sensor_recovery_time_.erase(sensor_name);  // Remove entry - will be added on next tick
    }
}

void ModeManager::requestTransition(OperatingMode new_mode) {
    if (current_mode_ == new_mode) {
        return;
    }

    std::cout << "[ModeManager] Transition: "
              << modeToString(current_mode_) << " -> " << modeToString(new_mode)
              << std::endl;

    current_mode_ = new_mode;
}

bool ModeManager::evaluateTransition(double current_time) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    // Update recovery times for sensors that are healthy but not yet tracked
    for (const auto& [sensor, is_healthy] : sensor_status_) {
        if (is_healthy && sensor_recovery_time_.find(sensor) == sensor_recovery_time_.end()) {
            // Sensor is healthy but not in recovery map - add it now
            sensor_recovery_time_[sensor] = current_time;
        }
    }
    
    OperatingMode current = current_mode_.load();
    
    switch (current) {
        case OperatingMode::NOMINAL:
            return shouldTransitionFromNominal(current_time);
            
        case OperatingMode::DEAD_RECKONING:
            return shouldTransitionFromDeadReckoning(current_time);
            
        case OperatingMode::LIDAR_ASSIST:
            return shouldTransitionFromLidarAssist(current_time);
            
        case OperatingMode::RELOCALIZED:
            return shouldTransitionFromRelocalized(current_time);
            
        case OperatingMode::FAIL_SAFE:
            return shouldTransitionFromFailSafe(current_time);
            
        default:
            return false;
    }
}

bool ModeManager::shouldTransitionFromNominal(double current_time) const {
    // Check for critical sensor failure (IMU is critical)
    if (!isSensorHealthy("IMU")) {
        const_cast<ModeManager*>(this)->transitionToMode(
            OperatingMode::FAIL_SAFE, 
            "Critical sensor (IMU) failure", 
            current_time);
        return true;
    }
    
    // Check if GPS is lost
    if (!isSensorHealthy("GPS")) {
        const_cast<ModeManager*>(this)->transitionToMode(
            OperatingMode::DEAD_RECKONING, 
            "GPS signal lost", 
            current_time);
        return true;
    }
    
    // Check if both GPS and LiDAR are lost (need at least one)
    if (!isSensorHealthy("GPS") && !isSensorHealthy("LiDAR")) {
        const_cast<ModeManager*>(this)->transitionToMode(
            OperatingMode::FAIL_SAFE, 
            "Both GPS and LiDAR unavailable", 
            current_time);
        return true;
    }
    
    return false;
}

bool ModeManager::shouldTransitionFromDeadReckoning(double current_time) const {
    // Check timeout
    double time_in_mode = current_time - mode_entry_time_;
    if (time_in_mode > DEAD_RECKONING_TIMEOUT_SEC) {
        const_cast<ModeManager*>(this)->transitionToMode(
            OperatingMode::FAIL_SAFE, 
            "Dead reckoning timeout exceeded (>60s)", 
            current_time);
        return true;
    }
    
    // Check if GPS recovered with hysteresis
    if (hasRecoveredWithHysteresis("GPS", current_time)) {
        const_cast<ModeManager*>(this)->transitionToMode(
            OperatingMode::NOMINAL, 
            "GPS signal recovered", 
            current_time);
        return true;
    }
    
    // Check if LiDAR is healthy (can assist with drift correction)
    if (isSensorHealthy("LiDAR") && hasRecoveredWithHysteresis("LiDAR", current_time)) {
        const_cast<ModeManager*>(this)->transitionToMode(
            OperatingMode::LIDAR_ASSIST, 
            "LiDAR available for drift correction", 
            current_time);
        return true;
    }
    
    // Check if IMU failed
    if (!isSensorHealthy("IMU")) {
        const_cast<ModeManager*>(this)->transitionToMode(
            OperatingMode::FAIL_SAFE, 
            "IMU failure during dead reckoning", 
            current_time);
        return true;
    }
    
    return false;
}

bool ModeManager::shouldTransitionFromLidarAssist(double current_time) const {
    // Check if GPS recovered
    if (hasRecoveredWithHysteresis("GPS", current_time)) {
        const_cast<ModeManager*>(this)->transitionToMode(
            OperatingMode::NOMINAL, 
            "GPS signal recovered", 
            current_time);
        return true;
    }
    
    // Check if loop closure detected (indicates successful relocalization)
    if (isSensorHealthy("LoopClosure")) {
        const_cast<ModeManager*>(this)->transitionToMode(
            OperatingMode::RELOCALIZED, 
            "Loop closure detected, Map-Lite correction applied", 
            current_time);
        return true;
    }
    
    // Check if LiDAR lost
    if (!isSensorHealthy("LiDAR")) {
        const_cast<ModeManager*>(this)->transitionToMode(
            OperatingMode::DEAD_RECKONING, 
            "LiDAR signal lost", 
            current_time);
        return true;
    }
    
    // Check if IMU failed
    if (!isSensorHealthy("IMU")) {
        const_cast<ModeManager*>(this)->transitionToMode(
            OperatingMode::FAIL_SAFE, 
            "IMU failure", 
            current_time);
        return true;
    }
    
    return false;
}

bool ModeManager::shouldTransitionFromRelocalized(double current_time) const {
    // Check if GPS recovered
    if (hasRecoveredWithHysteresis("GPS", current_time)) {
        const_cast<ModeManager*>(this)->transitionToMode(
            OperatingMode::NOMINAL, 
            "GPS signal recovered after relocalization", 
            current_time);
        return true;
    }
    
    // Check if IMU failed
    if (!isSensorHealthy("IMU")) {
        const_cast<ModeManager*>(this)->transitionToMode(
            OperatingMode::FAIL_SAFE, 
            "IMU failure", 
            current_time);
        return true;
    }
    
    // Stay in RELOCALIZED mode until GPS returns
    return false;
}

bool ModeManager::shouldTransitionFromFailSafe(double current_time) const {
    // FAIL_SAFE requires manual reset or sensor recovery
    // Check if all critical sensors are back online
    if (isSensorHealthy("IMU") && 
        (hasRecoveredWithHysteresis("GPS", current_time) || 
         hasRecoveredWithHysteresis("LiDAR", current_time))) {
        const_cast<ModeManager*>(this)->transitionToMode(
            OperatingMode::NOMINAL, 
            "Critical sensors recovered", 
            current_time);
        return true;
    }
    
    return false;
}

void ModeManager::forceMode(OperatingMode new_mode, const std::string& reason) {
    std::lock_guard<std::mutex> lock(mutex_);
    double current_time = 0.0; // Timestamp will be set by caller if needed
    transitionToMode(new_mode, reason, current_time);
}

OperatingMode ModeManager::getCurrentMode() const {
    return current_mode_.load();
}

double ModeManager::getModeUptime(double current_time) const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_time - mode_entry_time_;
}

void ModeManager::setOnEnterModeCallback(ModeCallback callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    on_enter_mode_ = callback;
}

void ModeManager::setOnExitModeCallback(ModeCallback callback) {
    std::lock_guard<std::mutex> lock(mutex_);
    on_exit_mode_ = callback;
}

std::map<std::string, bool> ModeManager::getSensorStatus() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return sensor_status_;
}

std::vector<ModeTransition> ModeManager::getTransitionHistory(size_t count) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    size_t num_to_return = std::min(count, transition_history_.size());
    std::vector<ModeTransition> result;
    result.reserve(num_to_return);
    
    // Return most recent transitions first
    for (size_t i = 0; i < num_to_return; ++i) {
        result.push_back(transition_history_[transition_history_.size() - 1 - i]);
    }
    
    return result;
}

std::string ModeManager::getDiagnostics(double current_time) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(2);
    
    // Current mode and uptime
    oss << "=== Mode Manager Diagnostics ===\n";
    oss << "Current Mode: " << modeToString(current_mode_.load()) << "\n";
    oss << "Mode Uptime: " << (current_time - mode_entry_time_) << " seconds\n\n";
    
    // Sensor availability
    oss << "Sensor Status:\n";
    for (const auto& [sensor, healthy] : sensor_status_) {
        oss << "  " << sensor << ": " << (healthy ? "HEALTHY" : "UNHEALTHY") << "\n";
    }
    oss << "\n";
    
    // Transition history
    oss << "Recent Transitions (last " << std::min(size_t(10), transition_history_.size()) << "):\n";
    size_t count = std::min(size_t(10), transition_history_.size());
    for (size_t i = 0; i < count; ++i) {
        const auto& trans = transition_history_[transition_history_.size() - 1 - i];
        oss << "  [" << trans.timestamp << "s] " 
            << modeToString(trans.from_mode) << " -> " 
            << modeToString(trans.to_mode) << " (" << trans.reason << ")\n";
    }
    
    return oss.str();
}

void ModeManager::reset(OperatingMode initial_mode) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    current_mode_.store(initial_mode);
    mode_entry_time_ = 0.0;
    
    // Reset sensor status to defaults
    sensor_status_["IMU"] = true;
    sensor_status_["GPS"] = true;
    sensor_status_["LiDAR"] = false;
    sensor_status_["WheelOdom"] = false;
    sensor_status_["LoopClosure"] = false;
    
    sensor_recovery_time_.clear();
    transition_history_.clear();
}

void ModeManager::transitionToMode(OperatingMode new_mode, const std::string& reason, double current_time) {
    OperatingMode old_mode = current_mode_.load();
    
    if (old_mode == new_mode) {
        return; // No transition needed
    }
    
    // Call exit callback
    if (on_exit_mode_) {
        on_exit_mode_(old_mode);
    }
    
    // Update mode
    current_mode_.store(new_mode);
    mode_entry_time_ = current_time;
    
    // Record transition
    transition_history_.emplace_back(current_time, old_mode, new_mode, reason);
    if (transition_history_.size() > MAX_HISTORY_SIZE) {
        transition_history_.erase(transition_history_.begin());
    }
    
    // Reset recovery times for new mode
    sensor_recovery_time_.clear();
    
    // Call enter callback
    if (on_enter_mode_) {
        on_enter_mode_(new_mode);
    }
}

bool ModeManager::isSensorHealthy(const std::string& sensor_name) const {
    auto it = sensor_status_.find(sensor_name);
    return it != sensor_status_.end() && it->second;
}

bool ModeManager::hasRecoveredWithHysteresis(const std::string& sensor_name, double current_time) const {
    if (!isSensorHealthy(sensor_name)) {
        return false;
    }
    
    auto it = sensor_recovery_time_.find(sensor_name);
    if (it == sensor_recovery_time_.end()) {
        return false; // Not yet tracked in recovery map
    }
    
    double time_since_recovery = current_time - it->second;
    return time_since_recovery >= RECOVERY_HYSTERESIS_SEC;
}

std::string ModeManager::modeToString(OperatingMode mode) const {
    switch (mode) {
        case OperatingMode::NOMINAL:
            return "NOMINAL";
        case OperatingMode::DEAD_RECKONING:
            return "DEAD_RECKONING";
        case OperatingMode::LIDAR_ASSIST:
            return "LIDAR_ASSIST";
        case OperatingMode::RELOCALIZED:
            return "RELOCALIZED";
        case OperatingMode::FAIL_SAFE:
            return "FAIL_SAFE";
        default:
            return "UNKNOWN";
    }
}

} // namespace axis