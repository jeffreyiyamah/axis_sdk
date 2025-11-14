#include "axis/axis_localizer.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <random>
#include <iomanip>

#ifdef WITH_PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/random_sample.h>
#endif

// Simulated sensor data generator
class SensorSimulator {
public:
    SensorSimulator() : rng_(rd_()), dist_(-1.0, 1.0) {}
    
    struct IMUData {
        Eigen::Vector3d accel;
        Eigen::Vector3d gyro;
        double timestamp;
    };
    
    struct GPSData {
        double lat, lon, alt;
        Eigen::Matrix3d covariance;
        double timestamp;
        bool valid;
    };
    
    struct LidarOdomData {
        Eigen::Isometry3d delta_pose;
        axis::Matrix6d covariance;
        double timestamp;
        bool valid;
    };
    
    struct WheelOdomData {
        double v_left, v_right;
        double timestamp;
        bool valid;
    };
    
    struct LidarScanData {
#ifdef WITH_PCL
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud;
#else
        std::vector<Eigen::Vector3d> points;
#endif
        double timestamp;
    };
    
    IMUData generateIMU(double timestamp) {
        IMUData data;
        // Simulate constant acceleration with small noise
        data.accel = Eigen::Vector3d(0.1, 0.0, 9.81) + Eigen::Vector3d(dist_(rng_), dist_(rng_), dist_(rng_)) * 0.01;
        data.gyro = Eigen::Vector3d(0.0, 0.0, 0.1) + Eigen::Vector3d(dist_(rng_), dist_(rng_), dist_(rng_)) * 0.001;
        data.timestamp = timestamp;
        return data;
    }
    
    GPSData generateGPS(double timestamp, bool valid = true) {
        GPSData data;
        data.lat = 37.7749 + dist_(rng_) * 0.0001; // San Francisco area
        data.lon = -122.4194 + dist_(rng_) * 0.0001;
        data.alt = 100.0 + dist_(rng_) * 1.0;
        data.covariance = Eigen::Matrix3d::Identity() * 4.0; // 2m std deviation
        data.timestamp = timestamp;
        data.valid = valid;
        return data;
    }
    
    LidarOdomData generateLidarOdom(double timestamp, bool valid = true) {
        LidarOdomData data;
        data.delta_pose = Eigen::Isometry3d::Identity();
        // Simulate small forward motion
        data.delta_pose.translation() = Eigen::Vector3d(0.1, 0.0, 0.0);
        data.delta_pose.linear() = Eigen::Matrix3d::Identity();
        data.covariance = axis::Matrix6d::Identity() * 0.01;
        data.timestamp = timestamp;
        data.valid = valid;
        return data;
    }
    
    WheelOdomData generateWheelOdom(double timestamp, bool valid = true) {
        WheelOdomData data;
        data.v_left = 1.0 + dist_(rng_) * 0.1;
        data.v_right = 1.0 + dist_(rng_) * 0.1;
        data.timestamp = timestamp;
        data.valid = valid;
        return data;
    }
    
    LidarScanData generateLidarScan(double timestamp) {
        LidarScanData data;
        data.timestamp = timestamp;
        
#ifdef WITH_PCL
        data.cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
        
        // Generate random points in a 10m x 10m x 5m box
        for (int i = 0; i < 1000; ++i) {
            pcl::PointXYZ point;
            point.x = dist_(rng_) * 5.0;
            point.y = dist_(rng_) * 5.0;
            point.z = dist_(rng_) * 2.5;
            data.cloud->push_back(point);
        }
#else
        // Generate random points
        for (int i = 0; i < 1000; ++i) {
            Eigen::Vector3d point(dist_(rng_) * 5.0, dist_(rng_) * 5.0, dist_(rng_) * 2.5);
            data.points.push_back(point);
        }
#endif
        
        return data;
    }

private:
    std::random_device rd_;
    std::mt19937 rng_;
    std::uniform_real_distribution<double> dist_;
};

void logPose(const axis::PoseMessage& pose) {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "POSE: pos=[" << pose.position.x() << ", " << pose.position.y() << ", " << pose.position.z() << "] ";
    std::cout << "quat=[" << pose.orientation.w() << ", " << pose.orientation.x() << ", " 
              << pose.orientation.y() << ", " << pose.orientation.z() << "] ";
    std::cout << "conf=" << std::setprecision(3) << pose.confidence << " ";
    std::cout << "mode=" << static_cast<int>(pose.mode) << std::endl;
}

int main() {
    std::cout << "=== Axis SDK Example ===" << std::endl;
    
    // Create localizer with configuration
    axis::AxisLocalizer axis("config/default_config.yaml");
    
    // Initialize and start
    if (!axis.initialize()) {
        std::cerr << "Failed to initialize AxisLocalizer" << std::endl;
        return -1;
    }
    
    if (!axis.start()) {
        std::cerr << "Failed to start AxisLocalizer" << std::endl;
        return -1;
    }
    
    std::cout << "AxisLocalizer started successfully" << std::endl;
    
    // Create sensor simulator
    SensorSimulator simulator;
    
    // Simulation parameters
    const double imu_rate = 100.0;      // Hz
    const double gps_rate = 10.0;       // Hz
    const double lidar_rate = 10.0;     // Hz
    const double wheel_rate = 50.0;     // Hz
    const double simulation_time = 30.0; // seconds
    
    std::cout << "Running simulation for " << simulation_time << " seconds..." << std::endl;
    std::cout << "Press Ctrl+C to stop early" << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    auto last_imu = start_time;
    auto last_gps = start_time;
    auto last_lidar = start_time;
    auto last_wheel = start_time;
    auto last_status = start_time;
    
    bool running = true;
    int imu_count = 0, gps_count = 0, lidar_count = 0, wheel_count = 0;
    
    try {
        while (running) {
            auto now = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - start_time);
            
            // Check simulation end
            if (elapsed.count() >= simulation_time) {
                std::cout << "Simulation completed" << std::endl;
                break;
            }
            
            double current_time = elapsed.count();
            
            // Generate IMU data at 100Hz
            auto imu_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - last_imu);
            if (imu_elapsed.count() >= 1.0 / imu_rate) {
                auto imu_data = simulator.generateIMU(current_time);
                axis.feedIMU(imu_data.accel, imu_data.gyro, imu_data.timestamp);
                last_imu = now;
                imu_count++;
            }
            
            // Generate GPS data at 10Hz
            auto gps_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - last_gps);
            if (gps_elapsed.count() >= 1.0 / gps_rate) {
                auto gps_data = simulator.generateGPS(current_time, true);
                if (gps_data.valid) {
                    axis.feedGPS(gps_data.lat, gps_data.lon, gps_data.alt, 
                                gps_data.covariance, gps_data.timestamp);
                }
                last_gps = now;
                gps_count++;
            }
            
            // Generate LiDAR odometry at 10Hz
            auto lidar_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - last_lidar);
            if (lidar_elapsed.count() >= 1.0 / lidar_rate) {
                auto lidar_data = simulator.generateLidarOdom(current_time, true);
                if (lidar_data.valid) {
                    axis.feedLidarOdom(lidar_data.delta_pose, lidar_data.covariance, lidar_data.timestamp);
                }
                
                // Also generate LiDAR scans for Map-Lite
                auto scan_data = simulator.generateLidarScan(current_time);
#ifdef WITH_PCL
                axis.feedLidarScan(scan_data.cloud, scan_data.timestamp);
#else
                axis.feedLidarScan(scan_data.points, scan_data.timestamp);
#endif
                
                last_lidar = now;
                lidar_count++;
            }
            
            // Generate wheel odometry at 50Hz
            auto wheel_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - last_wheel);
            if (wheel_elapsed.count() >= 1.0 / wheel_rate) {
                auto wheel_data = simulator.generateWheelOdom(current_time, true);
                if (wheel_data.valid) {
                    axis.feedWheelOdom(wheel_data.v_left, wheel_data.v_right, wheel_data.timestamp);
                }
                last_wheel = now;
                wheel_count++;
            }
            
            // Print status every 5 seconds
            auto status_elapsed = std::chrono::duration_cast<std::chrono::duration<double>>(now - last_status);
            if (status_elapsed.count() >= 5.0) {
                auto pose = axis.getPose();
                auto mode = axis.getMode();
                auto health = axis.getSensorHealth();
                auto weights = axis.getMLWeights();
                
                std::cout << "\n--- Status Update (t=" << std::setprecision(1) << elapsed.count() << "s) ---" << std::endl;
                logPose(pose);
                
                std::cout << "Mode: ";
                switch (mode) {
                    case axis::OperatingMode::NOMINAL: std::cout << "NOMINAL"; break;
                    case axis::OperatingMode::DEAD_RECKONING: std::cout << "DEAD_RECKONING"; break;
                    case axis::OperatingMode::LIDAR_ASSIST: std::cout << "LIDAR_ASSIST"; break;
                    case axis::OperatingMode::RELOCALIZED: std::cout << "RELOCALIZED"; break;
                    case axis::OperatingMode::FAIL_SAFE: std::cout << "FAIL_SAFE"; break;
                    default: std::cout << "UNKNOWN"; break;
                }
                std::cout << std::endl;
                
                std::cout << "Sensor Health: ";
                for (const auto& [sensor, status] : health) {
                    std::cout << sensor << "=";
                    switch (status) {
                        case axis::SensorHealth::ONLINE: std::cout << "OK "; break;
                        case axis::SensorHealth::DEGRADED: std::cout << "DEG "; break;
                        case axis::SensorHealth::OFFLINE: std::cout << "OFF "; break;
                        case axis::SensorHealth::UNAVAILABLE: std::cout << "UNAV "; break;
                        default: std::cout << "UNK "; break;
                    }
                }
                std::cout << std::endl;
                
                if (weights.size() >= 5) {
                    std::cout << "ML Weights: IMU=" << std::setprecision(3) << weights[0]
                              << " WHEEL=" << weights[1] << " LIDAR=" << weights[2]
                              << " VO=" << weights[3] << " GPS=" << weights[4] << std::endl;
                }
                
                auto relocal_result = axis.getLastRelocalizationResult();
                if (relocal_result && relocal_result->success) {
                    std::cout << "Relocalization: SUCCESS (score=" << relocal_result->match_score << ")" << std::endl;
                }
                
                std::cout << "Message counts: IMU=" << imu_count << " GPS=" << gps_count 
                          << " LiDAR=" << lidar_count << " Wheel=" << wheel_count << std::endl;
                
                last_status = now;
            }
            
            // Small sleep to prevent CPU spinning
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during simulation: " << e.what() << std::endl;
        running = false;
    }
    
    // Stop the localizer
    std::cout << "\nStopping AxisLocalizer..." << std::endl;
    axis.stop();
    
    std::cout << "=== Simulation Complete ===" << std::endl;
    std::cout << "Total messages processed:" << std::endl;
    std::cout << "  IMU: " << imu_count << std::endl;
    std::cout << "  GPS: " << gps_count << std::endl;
    std::cout << "  LiDAR: " << lidar_count << std::endl;
    std::cout << "  Wheel: " << wheel_count << std::endl;
    
    return 0;
}
