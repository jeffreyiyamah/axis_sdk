#pragma once
#include <Eigen/Dense>
#include <optional>
#include "axis/types.h"

/**
 * @file ekf_state.h
 * @brief Extended Kalman Filter state for Axis SDK.
 *
 * State vector layout [size 15]:
 *   - position:      [0:3]   (meters, local frame)
 *   - velocity:      [3:6]   (m/s, local frame)
 *   - orientation:   [6:10]  (quaternion x, y, z, w) (body frame wrt local)
 *   - accel_bias:    [10:13] (m/s^2)
 *   - gyro_bias:     [13:16] (rad/s)
 *
 * Coordinate frames:
 *    - Body frame: Robot (right, forward, up)
 *    - Local frame: EKF origin/start (ENU, meters)
 * Units:
 *    - position: meters; velocity: m/s; orientation: unit quaternion; accel/gyro bias: SI
 */
namespace axis {

class EKFState {
public:
    EKFState();
    explicit EKFState(const Eigen::VectorXd& initial_state);

    // Predict step: propagate state with IMU, compensate biases, propagate covariance
    void predict(const Eigen::Vector3d& accel, const Eigen::Vector3d& gyro, double dt);
    // Update step: generic EKF measurement update
    void update(const Eigen::VectorXd& measurement, const Eigen::MatrixXd& H, const Eigen::MatrixXd& R);
    // Get current [position, velocity, orientation]
    struct StateSnapshot {
        Eigen::Vector3d position;
        Eigen::Vector3d velocity;
        Eigen::Quaterniond orientation;
        Eigen::Vector3d accel_bias;
        Eigen::Vector3d gyro_bias;
    };
    StateSnapshot getState() const;
    Eigen::Matrix<double,15,15> getCovariance() const;
    void reset(const Eigen::VectorXd& initial_state);

    // Health check helper (e.g. time gating or numerical problems)
    SensorHealth checkHealth() const;

private:
    Eigen::Matrix<double,15,1> state_;  // See doc for layout
    Eigen::Matrix<double,15,15> P_;     // Covariance
    SensorHealth health_{SensorHealth::ONLINE};

    void conditionCovariance(); // enforce positive-definite, lower bounds
};

} // namespace axis
