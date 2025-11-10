#include "axis/utils/math_utils.h"
#include <cmath>

namespace axis {

Eigen::Matrix3d skewSymmetric(const Eigen::Vector3d& v) {
    Eigen::Matrix3d m;
    m <<     0, -v.z(),  v.y(),
          v.z(),     0, -v.x(),
         -v.y(),  v.x(),    0;
    return m;
}

Eigen::Matrix3d quaternionToRotation(const Eigen::Quaterniond& q) {
    return q.normalized().toRotationMatrix();
}

Eigen::Matrix3d quaternionToRotation(const Eigen::Vector4d& quat_v) {
    Eigen::Quaterniond q(quat_v[0], quat_v[1], quat_v[2], quat_v[3]);
    return q.normalized().toRotationMatrix();
}

Eigen::Vector4d normalizeQuaternion(const Eigen::Vector4d& quat_v) {
    Eigen::Quaterniond q(quat_v[0], quat_v[1], quat_v[2], quat_v[3]);
    q.normalize();
    return Eigen::Vector4d(q.x(), q.y(), q.z(), q.w());
}

double wrapAngle(double angle) {
    // Fast wrap to [-pi, pi]
    return std::remainder(angle, 2 * M_PI);
}

} // namespace axis
