# Axis SDK API Reference

This document summarizes the main public classes and methods exposed by Axis SDK.

## AxisLocalizer

`class axis::AxisLocalizer`

- **Constructor**
  - `explicit AxisLocalizer(const std::string& config_path);`

- **Lifecycle**
  - `bool initialize();`
  - `bool start();`
  - `void stop();`
  - `bool reset();`
  - `bool isInitialized() const;`
  - `bool isRunning() const;`

- **Sensor Inputs**
  - `void feedIMU(const Eigen::Vector3d& accel, const Eigen::Vector3d& gyro, double timestamp);`
  - `void feedGPS(double latitude, double longitude, double altitude, const Eigen::Matrix3d& covariance, double timestamp);`
  - `void feedLidarOdom(const Eigen::Isometry3d& delta_pose, const axis::Matrix6d& covariance, double timestamp);`
  - `void feedWheelOdom(double velocity_left, double velocity_right, double timestamp);`
  - `void feedVisualOdom(const Eigen::Isometry3d& delta_pose, const axis::Matrix6d& covariance, int feature_count, double timestamp);`
  - `void feedLidarScan(points_or_cloud, double timestamp);` (PCL or Eigen points depending on build).

- **Outputs**
  - `PoseMessage getPose() const;`
  - `OperatingMode getMode() const;`
  - `std::map<std::string, SensorHealth> getSensorHealth() const;`
  - `Eigen::VectorXd getMLWeights() const;`
  - `std::optional<RelocalizationResult> getLastRelocalizationResult() const;`

### Usage Examples by Sensor Type

- **IMU-only**
  - Call `feedIMU` at IMU rate (e.g., 100 Hz).
  - EKF predict step runs in `processIMUMeasurements` using a fixed dt; timestamps will be used for more advanced models.

- **GPS**
  - Call `feedGPS` at GPS rate (e.g., 1–10 Hz).
  - The first GPS measurement sets the origin; subsequent GPS updates are converted into ENU and update the EKF position block.

- **Wheel Odometry**
  - Call `feedWheelOdom` at wheel encoder rate.
  - The EKF update uses a simple velocity measurement in the x component of the local frame.

- **LiDAR Odometry**
  - Call `feedLidarOdom` with delta pose and 6x6 covariance.
  - Measurements are treated as pose deltas and fused as a 6D measurement with ML-weighted covariance.

- **Visual Odometry**
  - Call `feedVisualOdom` with delta pose, covariance, and feature count.
  - Position deltas update the EKF; feature count can be used in ML features.

- **LiDAR Scans for Map-Lite**
  - Call `feedLidarScan` with a point cloud or vector of points.
  - Scans are queued and consumed by the relocalization thread.

## EKFState

`class axis::EKFState`

- `EKFState();`
- `explicit EKFState(const Eigen::VectorXd& initial_state);`
- `void predict(const Eigen::Vector3d& accel, const Eigen::Vector3d& gyro, double dt);`
- `void update(const Eigen::VectorXd& measurement, const Eigen::MatrixXd& H, const Eigen::MatrixXd& R);`
- `StateSnapshot getState() const;`
- `Eigen::Matrix<double,15,15> getCovariance() const;`
- `void reset(const Eigen::VectorXd& initial_state);`
- `SensorHealth checkHealth() const;`

## ModeManager

`class axis::ModeManager`

- `explicit ModeManager(OperatingMode initial_mode = OperatingMode::NOMINAL);`
- `void updateSensorStatus(const std::string& sensor_name, bool is_healthy);`
- `bool evaluateTransition(double current_time);`
- `void forceMode(OperatingMode new_mode, const std::string& reason = "Manual override");`
- `OperatingMode getCurrentMode() const;`
- `void requestTransition(OperatingMode new_mode);`
- `double getModeUptime(double current_time) const;`
- `void setOnEnterModeCallback(ModeCallback callback);`
- `void setOnExitModeCallback(ModeCallback callback);`
- `std::map<std::string, bool> getSensorStatus() const;`
- `std::vector<ModeTransition> getTransitionHistory(size_t count = 10) const;`
- `std::string getDiagnostics(double current_time) const;`
- `void reset(OperatingMode initial_mode = OperatingMode::NOMINAL);`

## MLWeightingEngine

`class axis::MLWeightingEngine`

- `MLWeightingEngine();`
- `bool loadModel(const std::string& json_path);`
- `Eigen::VectorXd computeWeights(const SensorFeatures& features);`
- `bool isModelLoaded() const;`
- `ModelType getModelType() const;`
- `void setWeightBounds(double min_weight, double max_weight);`
- `std::pair<double,double> getWeightBounds() const;`

### Configuration parameters

From YAML (`ml_weighting.*`):

- `model_path`: path to JSON model file.
- `min_weight`, `max_weight`: clipping bounds for normalized weights.
- `update_interval`: seconds between ML weight recomputation.

## MapLiteRelocalizer

`class axis::MapLiteRelocalizer`

- `MapLiteRelocalizer();`
- `explicit MapLiteRelocalizer(const MapLiteConfig& config);`
- `bool initialize();`
- `void shutdown();`
- `void addScan(const LidarScan& scan);`
- `std::optional<RelocalizationResult> attemptRelocalization();`
- `bool isActive() const;`
- `void setActive(bool active);`
- `size_t getSubmapSize() const;`
- `const MapLiteConfig& getConfig() const;`
- `void setConfig(const MapLiteConfig& config);`
- `void setRelocalizationCallback(RelocalizationCallback callback);`
- `std::optional<RelocalizationResult> getLastResult() const;`

## Configuration Reference

### Global YAML keys

- `sensors.*` — enable/disable individual sensor modalities.
- `ekf.*` — EKF initialization and process noise.
- `ml_weighting.*` — ML model path and bounds.
- `maplite.*` — Map-Lite radius, ICP thresholds, and loop-closure settings.
- `diagnostics.*` — diagnostics publish rate and logging options.
