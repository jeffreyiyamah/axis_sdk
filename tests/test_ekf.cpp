#include <gtest/gtest.h>
#include <Eigen/Dense>
#include "axis/ekf_state.h"
#include "axis/utils/math_utils.h"

using axis::EKFState;

TEST(EKFTest, PredictStepWithIMUInputs)
{
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(15);
    x0.segment<4>(6) << 0.0, 0.0, 0.0, 1.0;
    EKFState ekf(x0);

    Eigen::Vector3d accel(0.0, 0.0, 1.0);
    Eigen::Vector3d gyro(0.0, 0.0, 0.1);
    double dt = 0.01;

    ekf.predict(accel, gyro, dt);
    auto state = ekf.getState();

    EXPECT_GT(state.velocity.z(), 0.0);
    EXPECT_GT(state.position.z(), 0.0);
}

TEST(EKFTest, UpdateStepWithGPSLikeMeasurement)
{
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(15);
    x0.segment<4>(6) << 0.0, 0.0, 0.0, 1.0;
    EKFState ekf(x0);

    Eigen::VectorXd z(3);
    z << 1.0, 2.0, 3.0;

    Eigen::MatrixXd H = Eigen::MatrixXd::Identity(3, 15);
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(3, 3) * 0.01;

    ekf.update(z, H, R);
    auto state = ekf.getState();

    EXPECT_NEAR(state.position.x(), 1.0, 1e-1);
    EXPECT_NEAR(state.position.y(), 2.0, 1e-1);
    EXPECT_NEAR(state.position.z(), 3.0, 1e-1);
}

TEST(EKFTest, CovarianceGrowthAndBounds)
{
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(15);
    x0.segment<4>(6) << 0.0, 0.0, 0.0, 1.0;
    EKFState ekf(x0);

    auto P0 = ekf.getCovariance();

    Eigen::Vector3d accel = Eigen::Vector3d::Zero();
    Eigen::Vector3d gyro = Eigen::Vector3d::Zero();

    for (int i = 0; i < 100; ++i) {
        ekf.predict(accel, gyro, 0.01);
    }

    auto P = ekf.getCovariance();

    EXPECT_GT(P.trace(), P0.trace());

    for (int i = 0; i < 15; ++i) {
        EXPECT_GE(P(i, i), 1e-6);
    }
}

TEST(EKFTest, QuaternionNormalization)
{
    Eigen::VectorXd x0 = Eigen::VectorXd::Zero(15);
    x0.segment<4>(6) << 0.1, 0.2, 0.3, 0.4;
    EKFState ekf(x0);

    Eigen::VectorXd z = Eigen::VectorXd::Zero(3);
    Eigen::MatrixXd H = Eigen::MatrixXd::Zero(3, 15);
    Eigen::MatrixXd R = Eigen::MatrixXd::Identity(3, 3);

    ekf.update(z, H, R);

    auto state = ekf.getState();
    double norm = state.orientation.coeffs().norm();
    EXPECT_NEAR(norm, 1.0, 1e-6);
}
