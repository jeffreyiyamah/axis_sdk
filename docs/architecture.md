# Axis SDK Architecture

## System Overview

Axis SDK provides a modular localization stack centered around an extended Kalman filter (EKF) with pluggable sensors and optional Map-Lite LiDAR relocalization and ML-based sensor weighting.

- **axis_core**: static library containing EKF, mode manager, ML weighting, Map-Lite, config parser, diagnostics, and AxisLocalizer.
- **axis_node**: reference executable that wires configuration, spins threads, and runs the main loop.

## Component Diagram

- **AxisLocalizer**
  - owns `EKFState`
  - owns `ModeManager`
  - owns `MLWeightingEngine`
  - optionally owns `MapLiteRelocalizer`
  - owns `DiagnosticsPublisher`
  - uses `ConfigParser` for YAML config

- **Sensors (inputs)**
  - IMU, Wheel Odom, LiDAR Odom, Visual Odom, GPS, LiDAR Scans

- **Outputs**
  - `PoseMessage`
  - diagnostics (pose, quality, health, mode)

## Component Interaction Flow

### Initialization

1. `AxisLocalizer` loads YAML via `ConfigParser`.
2. Constructs `EKFState`, `ModeManager`, `MLWeightingEngine`, optional `MapLiteRelocalizer`, and `DiagnosticsPublisher`.
3. ML model is loaded from JSON if configured.
4. Map-Lite parameters are populated from `maplite.*` keys.

### Runtime Loop

- **Main loop (10 ms tick)**
  - drain sensor queues (IMU, GPS, LiDAR odom, wheel odom, visual odom)
  - call EKF `predict` for IMU
  - call EKF `update` for other sensors (with ML-weighted covariances)
  - update `PoseMessage` and sensor health
  - evaluate mode via `ModeManager` and simple GPS/LiDAR rules
  - publish diagnostics

- **ML weighting thread**
  - wakes at `ml_weighting.update_interval`
  - builds `SensorFeatures` from EKF covariance and simple sensor stats
  - calls `MLWeightingEngine::computeWeights`
  - stores weights for use in EKF measurement covariances

- **Relocalization thread (when Map-Lite enabled)**
  - drains LiDAR scans
  - maintains rolling submap of scans
  - periodically attempts ICP-based relocalization
  - on success, generates `RelocalizationResult` and triggers an EKF update and mode change to `RELOCALIZED`.

## Threading Model

- **Threads**
  - Main processing thread (`AxisLocalizer::mainLoop`)
  - ML update thread (`mlUpdateLoop`)
  - Optional relocalization thread (`relocalizationLoop`)

- **Synchronization**
  - Per-sensor mutex + queue for IMU/GPS/odometry/scan feeds.
  - `state_mutex_` guards current pose and ML weights.
  - `ml_mutex_` and `relocalization_mutex_` guard condition-variable waits.

- **Shutdown**
  - `should_shutdown_` atomic flag requested by `stop()`.
  - All condition variables notified; threads join before cleanup.

## Coordinate Frames

- **Body frame**
  - Robot-fixed frame (right, forward, up).

- **Local frame**
  - EKF origin frame (approximate ENU) defined by initial GPS origin.

- **GPS to ENU**
  - `AxisLocalizer::gpsToENU` converts lat/lon/alt to ENU using a spherical Earth approximation and stored origin.

- **LiDAR / VO frames**
  - LiDAR odometry and visual odometry are handled as delta poses in the local frame; Map-Lite operates in the same local frame for relocalization transforms.
