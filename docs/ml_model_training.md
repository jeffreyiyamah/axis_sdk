# ML Model Training for Axis SDK

This document describes how to train and export ML models for `MLWeightingEngine`.

## Overview

Axis SDK uses ML-based sensor weighting to adaptively scale measurement covariances for IMU, wheel odometry, LiDAR, visual odometry, and GPS.

- Inputs: `SensorFeatures` (10D vector).
- Outputs: 5D weight vector `[w_imu, w_wheel, w_lidar, w_vo, w_gps]`.

Models can be implemented as:

- Gradient-boosted decision trees (GBDT).
- Multi-layer perceptron (MLP).

## Feature Engineering

Current `SensorFeatures` fields:

- `imu_noise_variance`
- `lidar_return_density`
- `vo_feature_count`
- `gps_snr`
- `gps_satellite_count`
- `ekf_innovation_imu`
- `ekf_innovation_wheel`
- `ekf_innovation_lidar`
- `ekf_innovation_vo`
- `ekf_innovation_gps`

Recommended practices:

- Normalize raw signals (e.g., z-score or min-max) before training.
- Clip outliers using robust statistics (e.g., percentile-based limits).
- Derive additional features such as innovation ratios or temporal trends if needed.

## JSON Export Format

`MLWeightingEngine` expects a custom JSON format.

### Common top-level fields

- `model_type`: `"GBDT"` or `"MLP"`.
- `output_scale`: scalar applied to network outputs.
- `output_bias`: scalar added after scaling.

### MLP schema

```jsonc
{
  "model_type": "MLP",
  "layers": [
    {
      "weights": [[...], [...]], // 2D array (rows = out_dim, cols = in_dim)
      "bias":    [...],          // 1D array of length out_dim
      "activation": "relu"      // or "linear"
    },
    {
      "weights": [[...]],
      "bias":    [...],
      "activation": "linear"
    }
  ],
  "output_scale": 1.0,
  "output_bias": 0.0
}
```

### GBDT schema

GBDT models are represented as a list of trees with nested nodes.

```jsonc
{
  "model_type": "GBDT",
  "learning_rate": 0.1,
  "init_prediction": 0.0,
  "trees": [
    {
      "is_leaf": false,
      "feature_index": 0,
      "threshold": 0.5,
      "left_child": { "is_leaf": true, "value": 0.8 },
      "right_child": { "is_leaf": true, "value": 1.2 }
    }
  ]
}
```

Each node uses:

- `is_leaf`: boolean.
- `value`: leaf prediction (when `is_leaf == true`).
- `feature_index`, `threshold`: split parameters (when not leaf).
- `left_child`, `right_child`: nested node objects.

## Training Data Collection

- Log `SensorFeatures` and ground-truth or heuristic target weights during operation.
- Recommended to log:
  - Raw sensor measurements.
  - EKF covariance and innovations.
  - Final fused pose quality metrics.

Targets can be:

- Hand-designed weights based on heuristics.
- Weights inferred from performance metrics (e.g., pose error vs. ground truth).

## Python Notebook Template

A typical workflow in a notebook:

1. Load logs into pandas DataFrame.
2. Build feature matrix `X` from `SensorFeatures` fields.
3. Build target matrix `Y` (5D weights).
4. Train model (e.g., scikit-learn GBDT or small MLP in PyTorch).
5. Export weights/trees into the JSON format above.

Example outline:

```python
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

# 1) Load data
logs = pd.read_csv("axis_logs.csv")

feature_cols = [
    "imu_noise_variance",
    "lidar_return_density",
    "vo_feature_count",
    "gps_snr",
    "gps_satellite_count",
    "ekf_innovation_imu",
    "ekf_innovation_wheel",
    "ekf_innovation_lidar",
    "ekf_innovation_vo",
    "ekf_innovation_gps",
]

X = logs[feature_cols].values
Y = logs[["w_imu", "w_wheel", "w_lidar", "w_vo", "w_gps"]].values

# 2) Train per-output GBDT models (example)
models = []
for i in range(Y.shape[1]):
    m = GradientBoostingRegressor(n_estimators=20, learning_rate=0.1)
    m.fit(X, Y[:, i])
    models.append(m)

# 3) Export to Axis JSON (left as an exercise)
#    You would traverse the sklearn trees and emit the GBDT schema.
```

The notebook should finish by writing `ml_model.json` into `config/` for consumption by Axis SDK.
