#pragma once

#include "types.h"
#include <atomic>
#include <chrono>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace axis {

/**
 * @brief Transition event record for diagnostics
 */
struct ModeTransition {
    double timestamp;
    OperatingMode from_mode;
    OperatingMode to_mode;
    std::string reason;

    ModeTransition(double ts, OperatingMode from, OperatingMode to, const std::string& r)
        : timestamp(ts), from_mode(from), to_mode(to), reason(r) {}
};

/**
 * @brief Manages operating mode state machine for Axis SDK
 * 
 * Thread-safe state machine that handles transitions between operating modes
 * based on sensor health and timing constraints.
 */
class ModeManager {
public:
    using ModeCallback = std::function<void(OperatingMode)>;

    /**
     * @brief Construct a new Mode Manager
     * @param initial_mode Starting operating mode (default: NOMINAL)
     */
    explicit ModeManager(OperatingMode initial_mode = OperatingMode::NOMINAL);

    /**
     * @brief Update the health status of a sensor
     * @param sensor_name Name of the sensor (e.g., "IMU", "GPS", "LiDAR")
     * @param is_healthy True if sensor is operational, false otherwise
     */
    void updateSensorStatus(const std::string& sensor_name, bool is_healthy);

    /**
     * @brief Evaluate if a mode transition is needed based on current state
     * @param current_time Current timestamp in seconds
     * @return True if a transition occurred, false otherwise
     */
    bool evaluateTransition(double current_time);

    /**
     * @brief Force a mode change (manual override)
     * @param new_mode Target operating mode
     * @param reason Reason for the forced transition
     */
    void forceMode(OperatingMode new_mode, const std::string& reason = "Manual override");

    /**
     * @brief Get the current operating mode
     * @return Current OperatingMode
     */
    OperatingMode getCurrentMode() const;

    /**
     * @brief Get time spent in current mode
     * @return Duration in seconds
     */
    double getModeUptime(double current_time) const;

    /**
     * @brief Register callback for mode entry events
     * @param callback Function to call when entering a new mode
     */
    void setOnEnterModeCallback(ModeCallback callback);

    /**
     * @brief Register callback for mode exit events
     * @param callback Function to call when exiting a mode
     */
    void setOnExitModeCallback(ModeCallback callback);

    /**
     * @brief Get sensor availability summary
     * @return Map of sensor names to their health status
     */
    std::map<std::string, bool> getSensorStatus() const;

    /**
     * @brief Get recent transition history
     * @param count Maximum number of transitions to return (default: 10)
     * @return Vector of recent transitions, newest first
     */
    std::vector<ModeTransition> getTransitionHistory(size_t count = 10) const;

    /**
     * @brief Get diagnostic information as a formatted string
     * @param current_time Current timestamp for uptime calculation
     * @return Formatted diagnostic string
     */
    std::string getDiagnostics(double current_time) const;

    /**
     * @brief Reset to initial state
     * @param initial_mode Mode to reset to (default: NOMINAL)
     */
    void reset(OperatingMode initial_mode = OperatingMode::NOMINAL);

private:
    // Core state
    std::atomic<OperatingMode> current_mode_;
    std::map<std::string, bool> sensor_status_;
    mutable std::mutex mutex_;

    // Timing
    double mode_entry_time_;
    double last_gps_healthy_time_;
    
    // Hysteresis tracking
    std::map<std::string, double> sensor_recovery_time_;
    static constexpr double RECOVERY_HYSTERESIS_SEC = 1.0;
    static constexpr double DEAD_RECKONING_TIMEOUT_SEC = 60.0;

    // Callbacks
    ModeCallback on_enter_mode_;
    ModeCallback on_exit_mode_;

    // Diagnostics
    std::vector<ModeTransition> transition_history_;
    static constexpr size_t MAX_HISTORY_SIZE = 50;

    // Helper methods
    void transitionToMode(OperatingMode new_mode, const std::string& reason, double current_time);
    bool shouldTransitionFromNominal(double current_time) const;
    bool shouldTransitionFromDeadReckoning(double current_time) const;
    bool shouldTransitionFromLidarAssist(double current_time) const;
    bool shouldTransitionFromRelocalized(double current_time) const;
    bool shouldTransitionFromFailSafe(double current_time) const;
    
    bool isSensorHealthy(const std::string& sensor_name) const;
    bool hasRecoveredWithHysteresis(const std::string& sensor_name, double current_time) const;
    
    std::string modeToString(OperatingMode mode) const;
};

} // namespace axis
