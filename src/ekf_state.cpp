#include "axis/ekf_state.h"
#include "axis/types.h"
#include "axis/utils/math_utils.h"
#include <cmath>
#include <algorithm>

namespace axis {

namespace {
constexpr double MIN_COV = 1e-6; // Covariance lower bound
}

EKFState::EKFState() : state_(Eigen::Matrix<double,16,1>::Zero()), P_(Eigen::Matrix<double,16,16>::Identity()*1e-2) {}

EKFState::EKFState(const Eigen::VectorXd& initial_state)
    : state_(initial_state.head<16>()), P_(Eigen::Matrix<double,16,16>::Identity()*10.0) {}

void EKFState::predict(const Eigen::Vector3d& accel_in, const Eigen::Vector3d& gyro_in, double dt) {
    // Unpack
    Eigen::Vector3d pos = state_.segment<3>(0);
    Eigen::Vector3d vel = state_.segment<3>(3);
    Eigen::Quaterniond q(state_.segment<4>(6));
    q.normalize();
    Eigen::Vector3d accel_bias = state_.segment<3>(10);
    Eigen::Vector3d gyro_bias = state_.segment<3>(13);

    // Bias-correct
    Eigen::Vector3d acc = accel_in - accel_bias;
    Eigen::Vector3d gyro = gyro_in - gyro_bias;

    // State propagation (simple model)
    Eigen::Matrix3d R = axis::quaternionToRotation(q);
    vel += R * acc * dt;
    pos += vel * dt;
    Eigen::Quaterniond dq;
    Eigen::Vector3d omega = gyro * dt * 0.5;
    dq.w() = 1.0;
    dq.vec() = omega;
    dq.normalize();
    q = (q * dq).normalized();

    // Bias as random walk (no change)

    // Write back
    state_.segment<3>(0) = pos;
    state_.segment<3>(3) = vel;
    state_.segment<4>(6) = Eigen::Vector4d(q.x(), q.y(), q.z(), q.w());

    // Covariance propagation (simple - no input noise)
    // TODO: Insert reasonable F, Q computation
    P_ = P_ + Eigen::Matrix<double,16,16>::Identity() * MIN_COV * dt; // crude inflate
    conditionCovariance();
}

void EKFState::update(const Eigen::VectorXd& y, const Eigen::MatrixXd& H, const Eigen::MatrixXd& R) {
    const Eigen::VectorXd hx = H * state_;
    Eigen::VectorXd innov = y - hx;
    Eigen::MatrixXd S = H * P_ * H.transpose() + R;
    Eigen::MatrixXd K = P_ * H.transpose() * S.inverse();
    state_ += K * innov;
    P_ = (Eigen::Matrix<double,16,16>::Identity() - K * H) * P_;
    // Normalize quaternion
    Eigen::Vector4d qv = state_.segment<4>(6);
    state_.segment<4>(6) = axis::normalizeQuaternion(qv);
    conditionCovariance();
}

EKFState::StateSnapshot EKFState::getState() const {
    Eigen::Vector3d pos = state_.segment<3>(0);
    Eigen::Vector3d vel = state_.segment<3>(3);
    Eigen::Vector4d qv = state_.segment<4>(6);
    Eigen::Quaterniond quat(qv[0], qv[1], qv[2], qv[3]);
    quat.normalize();
    Eigen::Vector3d accel_bias = state_.segment<3>(10);
    Eigen::Vector3d gyro_bias = state_.segment<3>(13);
    return {pos, vel, quat, accel_bias, gyro_bias};
}

Eigen::Matrix<double,16,16> EKFState::getCovariance() const {
    return P_;
}

void EKFState::reset(const Eigen::VectorXd& initial_state) {
    state_ = initial_state.head<16>();
    P_.setIdentity();
    P_ *= 1e-2;
    health_ = SensorHealth::ONLINE;
}

void EKFState::conditionCovariance() {
    for (int i = 0; i < 15; ++i)
        if(P_(i, i) < MIN_COV)
            P_(i, i) = MIN_COV;
    // TODO: Test for symmetry and positive-definite (future improvement)
}

SensorHealth EKFState::checkHealth() const {
    // Example: Check for large covariance or NaN
    if(!P_.allFinite() || P_.maxCoeff() > 1e6)
        return SensorHealth::DEGRADED;
    return health_;
}

} // namespace axis
