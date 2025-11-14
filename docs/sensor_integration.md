# Sensor Integration Guide

This document explains how to integrate new sensor types and wire them into Axis SDK.

## Adding a New Sensor Type

1. **Define measurement struct**
   - Add a struct in `axis_localizer.h` similar to `IMUMeasurement`, `GPSMeasurement`, etc.
   - Include fields for raw data, covariance, timestamp, and `valid` flag.

2. **Extend AxisLocalizer queues**
   - Add a mutex and `std::queue<YourMeasurement>` to `AxisLocalizer`.
   - Add a `feedYourSensor(...)` method to push measurements into the queue.

3. **Implement processing method**
   - Add `processYourSensorMeasurements()` in `axis_localizer.cpp`.
   - Pop measurements and construct an EKF measurement vector `z`, matrix `H`, and covariance `R`.
   - Call `ekf_->update(z, H, R);`.

4. **Update sensor health**
   - Maintain entries in `sensor_health_` map for your sensor.
   - Set health to `SensorHealth::ONLINE` when valid data is consumed.

5. **Expose configuration**
   - Add a YAML key under `sensors.*` (e.g., `sensors.your_sensor: true`).
   - Use `ConfigParser` to gate processing and enable/disable the sensor.

## Sensor Health Criteria

Typical criteria for setting `SensorHealth`:

- **ONLINE**
  - Recent valid measurement within a time threshold.
  - No numeric failures (NaNs, infinities, extreme covariance).

- **DEGRADED**
  - High noise or covariance.
  - Inconsistent readings (large innovations) but still usable.

- **OFFLINE / UNAVAILABLE**
  - No measurements received for a timeout period.
  - Driver reports hard failure.

Health is tracked in `sensor_health_` and published through diagnostics.

## Measurement Format Specifications

### IMU

- Acceleration: `Eigen::Vector3d` (m/s²) in body frame.
- Angular velocity: `Eigen::Vector3d` (rad/s) in body frame.
- Timestamp: double seconds (monotonic).

### GPS

- Latitude/longitude in degrees, altitude in meters.
- Covariance: `Eigen::Matrix3d` position covariance (ENU after conversion).
- Converted to ENU in `gpsToENU` before EKF update.

### Wheel Odometry

- Left/right wheel velocities (m/s).
- Simple linear velocity measurement fused on x-velocity.

### LiDAR / Visual Odometry

- `Eigen::Isometry3d` delta pose in local frame.
- `axis::Matrix6d` covariance on position/orientation.
- Fused as a 6D measurement with ML-weighted covariance.

### LiDAR Scans

- PCL point cloud (`pcl::PointCloud<pcl::PointXYZ>::Ptr`) when built with `WITH_PCL`.
- `std::vector<Eigen::Vector3d>` point list when PCL is disabled.

## Integrating with ML Weighting

- Extend `SensorFeatures` in `ml_weighting.h` when adding new signals that should affect weights.
- Update `AxisLocalizer::extractSensorFeatures()` to populate new feature fields from EKF covariance or sensor diagnostics.
- Re-train ML model to consume new features and export using the JSON schema described in `ml_model_training.md`.
