#pragma once
#include <Eigen/Dense>
#include <cmath>

namespace axis {

/**
 * @brief Create skew-symmetric matrix from a vector (for cross product)
 */
Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& v);

/**
 * @brief Convert quaternion (Eigen::Quaterniond or Vector4d [x,y,z,w]) to 3x3 rotation matrix
 */
Eigen::Matrix3d quaternionToRotation(const Eigen::Quaterniond& q);
Eigen::Matrix3d quaternionToRotation(const Eigen::Vector4d& quat);

/**
 * @brief Normalize quaternion vector [x,y,z,w]
 */
Eigen::Vector4d normalizeQuaternion(const Eigen::Vector4d& quat);

/**
 * @brief Wrap angle to [-pi, pi]
 */
double wrapAngle(double angle);

} // namespace axis
