# Axis SDK

Axis SDK is a modular localization library built around an extended Kalman filter, sensor fusion, ML-based sensor weighting, and optional LiDAR Map-Lite relocalization.

## Build Instructions

### Prerequisites

- C++17 compiler
- CMake >= 3.10
- Eigen3
- yaml-cpp
- Optional: PCL (for LiDAR point clouds, build with `-DUSE_PCL=ON`)

### Configure and Build

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j
ctest
```

To enable PCL-based Map-Lite:

```bash
cmake .. -DCMAKE_BUILD_TYPE=Release -DUSE_PCL=ON
```

## Quick Start Example

1. Choose a configuration, e.g. IMU+GPS+LiDAR:

   - `config/imu_gps_lidar.yaml`

2. Run the reference node (example):

```cpp
#include "axis/axis_localizer.h"

int main() {
    axis::AxisLocalizer localizer("config/imu_gps_lidar.yaml");
    if (!localizer.initialize()) return 1;
    if (!localizer.start()) return 1;

    // Feed sensors in your own loop (pseudo code)
    // localizer.feedIMU(accel, gyro, t);
    // localizer.feedGPS(lat, lon, alt, cov, t);
    // localizer.feedLidarOdom(delta_pose, cov6, t);

    // Query pose
    auto pose = localizer.getPose();

    localizer.stop();
    return 0;
}
```

## Hardware Requirements

Axis SDK is designed to run on embedded platforms such as NVIDIA Jetson.

Typical tested configuration:

- **Jetson Xavier / Orin class**
  - 6+ CPU cores
  - 8 GB+ RAM
  - Ubuntu-based OS with CUDA-capable GPU (for upstream perception stacks)

`config/jetson_optimized.yaml` contains reduced update rates and smaller Map-Lite submaps suitable for resource-constrained deployments.

## Dependencies and Installation

### Ubuntu / Debian example

Install dependencies:

```bash
sudo apt-get install libeigen3-dev libyaml-cpp-dev
# Optional: PCL
sudo apt-get install libpcl-dev
```

Then build as shown above.

Axis SDK is currently structured as a CMake project producing:

- `axis_core` (library)
- `axis_node` (example executable)

You can link `axis_core` into your own applications via CMake:

```cmake
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)

add_subdirectory(axis_sdk)

target_link_libraries(your_app PRIVATE axis_core Eigen3::Eigen yaml-cpp)
```
