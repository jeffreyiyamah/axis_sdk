// tests/test_ekf.cpp
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "axis/ekf_state.h"
#include "axis/utils/math_utils.h"

using axis::EKFState;
using Eigen::Vector3d;
using Eigen::VectorXd;
using Eigen::MatrixXd;

// -----------------------------------------------------------------------------
// Helper: zero state with unit quaternion (x,y,z,w order)
static VectorXd ZeroState()
{
    VectorXd x = VectorXd::Zero(16);
    x.segment<4>(6) << 0.0, 0.0, 0.0, 1.0;
    return x;
}

// -----------------------------------------------------------------------------
// 1. Predict step moves velocity/position

TEST(EKFTest, PredictStepWithIMUInputs)
{
    EKFState ekf(ZeroState());

    Vector3d accel(0.0, 0.0, 1.0);   // 1 m/s² up
    Vector3d gyro (0.0, 0.0, 0.1);   // 0.1 rad/s
    double   dt = 0.01;

    ekf.predict(accel, gyro, dt);
    auto s = ekf.getState();

    EXPECT_GT(s.velocity.z(), 0.0);
    EXPECT_GT(s.position.z(), 0.0);
}

// -----------------------------------------------------------------------------
// 2. GPS update pulls position to measurement

TEST(EKFTest, UpdateStepWithGPSLikeMeasurement)
{
    EKFState ekf(ZeroState());

    
    Vector3d accel(0.0, 0.0, 0.0);
    Vector3d gyro = Vector3d::Zero();
    double dt = 0.05;   // 20 Hz

    for (int i = 0; i < 40; ++i) { 
        ekf.predict(accel, gyro, dt);
    }

    VectorXd z(3);  z << 1.0, 2.0, 3.0;
    MatrixXd H = MatrixXd::Zero(3, 16);
    H.block<3,3>(0,0) = MatrixXd::Identity(3,3);

    MatrixXd R = MatrixXd::Identity(3,3) * 0.01;

    ekf.update(z, H, R);

    auto state = ekf.getState();

    const double tol = 1e-2;
    EXPECT_NEAR(state.position.x(), 1.0, tol);
    EXPECT_NEAR(state.position.y(), 2.0, tol);
    EXPECT_NEAR(state.position.z(), 3.0, tol);
}


TEST(EKFTest, CovarianceGrowthAndBounds)
{
    EKFState ekf(ZeroState());
    auto P0 = ekf.getCovariance();

    Vector3d a = Vector3d::Zero();
    Vector3d g = Vector3d::Zero();
    for (int i = 0; i < 100; ++i) {
        ekf.predict(a, g, 0.01);
    }

    auto P = ekf.getCovariance();
    EXPECT_GT(P.trace(), P0.trace());

    for (int i = 0; i < 16; ++i) {
        EXPECT_GE(P(i,i), 1e-6);
    }
}


TEST(EKFTest, QuaternionNormalization)
{
    VectorXd x = VectorXd::Zero(16);
    x.segment<4>(6) << 0.1, 0.2, 0.3, 0.4;
    EKFState ekf(x);

    VectorXd z = VectorXd::Zero(3);
    MatrixXd H = MatrixXd::Zero(3,16);
    MatrixXd R = MatrixXd::Identity(3,3);
    ekf.update(z, H, R);

    EXPECT_NEAR(ekf.getState().orientation.coeffs().norm(), 1.0, 1e-6);
}