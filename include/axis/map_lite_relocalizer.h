#pragma once
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <atomic>
#include <condition_variable>
#include <queue>
#include <chrono>
#include <optional>
#include <functional>
#include "axis/types.h"
#include <unordered_set>
#ifdef WITH_PCL
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/registration/icp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/octree/octree_pointcloud.h>
#endif

namespace axis {

/**
 * @brief LiDAR scan data structure
 */
struct LidarScan {
    double timestamp{0.0};
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
    
#ifdef WITH_PCL
    pcl::PointCloud<pcl::PointXYZ>::Ptr pointcloud;
#else
    std::vector<Eigen::Vector3d> points;
#endif
    
    LidarScan() {
#ifdef WITH_PCL
        pointcloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(new pcl::PointCloud<pcl::PointXYZ>);
#endif
    }
};

/**
 * @brief Relocalization result
 */
struct RelocalizationResult {
    bool success{false};
    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    double match_score{0.0};
    double timestamp{0.0};
    int inlier_count{0};
    
    RelocalizationResult() = default;
    RelocalizationResult(bool s, const Eigen::Isometry3d& t, double score, double ts, int inliers)
        : success(s), transform(t), match_score(score), timestamp(ts), inlier_count(inliers) {}
};

/**
 * @brief Configuration for MapLite relocalizer
 */
struct MapLiteConfig {
    double submap_radius{50.0};          // meters
    int max_scans_in_submap{100};        // maximum number of scans to keep
    double icp_max_correspondence_distance{0.5}; // meters
    int icp_max_iterations{50};
    double icp_transformation_epsilon{1e-6};
    double icp_fitness_epsilon{1e-6};
    double match_score_threshold{0.7};   // minimum score for valid relocalization
    double min_time_between_attempts{1.0}; // seconds
    double loop_closure_distance_threshold{10.0}; // meters
    double voxel_leaf_size{0.1};         // meters for downsampling
    bool enable_voxel_filter{true};
    bool enable_loop_closure{true};
    
    MapLiteConfig() = default;
};

/**
 * @brief Map-based LiDAR relocalization engine
 * 
 * Maintains a rolling local submap of recent LiDAR scans and attempts
 * to relocalize the vehicle when GPS is unavailable by matching current
 * scans against the accumulated submap using ICP or NDT.
 */
class MapLiteRelocalizer {
public:
    /**
     * @brief Callback type for relocalization results
     */
    using RelocalizationCallback = std::function<void(const RelocalizationResult&)>;
    
    MapLiteRelocalizer();
    explicit MapLiteRelocalizer(const MapLiteConfig& config);
    ~MapLiteRelocalizer();
    
    /**
     * @brief Initialize the relocalizer
     * @return true if initialization successful
     */
    bool initialize();
    
    /**
     * @brief Shutdown the relocalizer and stop background thread
     */
    void shutdown();
    
    /**
     * @brief Add a new LiDAR scan to the submap
     * @param scan LiDAR scan data
     */
    void addScan(const LidarScan& scan);
    
    /**
     * @brief Attempt relocalization with current scan
     * @return Optional relocalization result
     */
    std::optional<RelocalizationResult> attemptRelocalization();
    
    /**
     * @brief Check if relocalizer is active (GPS unavailable)
     */
    bool isActive() const { return is_active_.load(); }
    
    /**
     * @brief Set active status (true when GPS unavailable)
     */
    void setActive(bool active) { is_active_.store(active); }
    
    /**
     * @brief Get current submap size
     */
    size_t getSubmapSize() const;
    
    /**
     * @brief Get configuration
     */
    const MapLiteConfig& getConfig() const { return config_; }
    
    /**
     * @brief Set configuration
     */
    void setConfig(const MapLiteConfig& config);
    
    /**
     * @brief Set callback for relocalization results
     */
    void setRelocalizationCallback(RelocalizationCallback callback) {
        callback_ = callback;
    }
    
    /**
     * @brief Get last relocalization result
     */
    std::optional<RelocalizationResult> getLastResult() const;

private:
    MapLiteConfig config_;
    bool initialized_{false};
    std::atomic<bool> is_active_{false};
    std::atomic<bool> should_shutdown_{false};
    
    // Submap storage
    mutable std::mutex submap_mutex_;
    std::vector<LidarScan> submap_scans_;
    Eigen::Vector3d submap_center_ = Eigen::Vector3d::Zero();
    
    // Background processing
    std::unique_ptr<std::thread> worker_thread_;
    std::mutex queue_mutex_;
    std::queue<LidarScan> scan_queue_;
    std::condition_variable queue_cv_;
    std::chrono::steady_clock::time_point last_attempt_time_;
    
    // Results
    std::optional<RelocalizationResult> last_result_;
    mutable std::mutex result_mutex_;
    RelocalizationCallback callback_;
    
    // PCL-specific members
#ifdef WITH_PCL
    pcl::VoxelGrid<pcl::PointXYZ> voxel_filter_;
    pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp_;
#endif
    
    // Private methods
    void workerLoop();
    void processScan(const LidarScan& scan);
    void updateSubmap(const LidarScan& scan);
    std::optional<RelocalizationResult> performICP(const LidarScan& current_scan);
    bool shouldAttemptRelocalization() const;
    bool isLoopClosureCandidate(const LidarScan& scan) const;
    
    // Utility methods
    Eigen::Isometry3d poseToTransform(const Eigen::Vector3d& position, 
                                     const Eigen::Quaterniond& orientation) const;
    void transformPointCloud(const LidarScan& scan, 
                            const Eigen::Isometry3d& transform,
                            LidarScan& transformed_scan) const;
    double computeDistance(const Eigen::Vector3d& p1, const Eigen::Vector3d& p2) const;
    
#ifdef WITH_PCL
    void downsamplePointCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) const;
    pcl::PointCloud<pcl::PointXYZ>::Ptr createSubmapCloud() const;
#else
    std::vector<Eigen::Vector3d> downsamplePoints(const std::vector<Eigen::Vector3d>& points) const;
    std::vector<Eigen::Vector3d> createSubmapPoints() const;
#endif
};

} // namespace axis
