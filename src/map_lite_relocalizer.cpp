#include "axis/map_lite_relocalizer.h"
#include <iostream>
#include <algorithm>
#include <cmath>
#include <unordered_set>


namespace axis {

MapLiteRelocalizer::MapLiteRelocalizer() 
    : last_attempt_time_(std::chrono::steady_clock::now()) {
}

MapLiteRelocalizer::MapLiteRelocalizer(const MapLiteConfig& config)
    : config_(config), last_attempt_time_(std::chrono::steady_clock::now()) {
}

MapLiteRelocalizer::~MapLiteRelocalizer() {
    shutdown();
}

bool MapLiteRelocalizer::initialize() {
#ifdef WITH_PCL
    // Configure ICP
    icp_.setMaxCorrespondenceDistance(config_.icp_max_correspondence_distance);
    icp_.setMaximumIterations(config_.icp_max_iterations);
    icp_.setTransformationEpsilon(config_.icp_transformation_epsilon);
    icp_.setEuclideanFitnessEpsilon(config_.icp_fitness_epsilon);
    
    // Configure voxel filter
    if (config_.enable_voxel_filter) {
        voxel_filter_.setLeafSize(config_.voxel_leaf_size, 
                                 config_.voxel_leaf_size, 
                                 config_.voxel_leaf_size);
    }
#endif
    
    // Start background worker thread
    should_shutdown_.store(false);
    worker_thread_ = std::make_unique<std::thread>(&MapLiteRelocalizer::workerLoop, this);
    
    initialized_ = true;
    return true;
}

void MapLiteRelocalizer::shutdown() {
    if (initialized_) {
        should_shutdown_.store(true);
        queue_cv_.notify_all();
        
        if (worker_thread_ && worker_thread_->joinable()) {
            worker_thread_->join();
        }
        
        initialized_ = false;
    }
}

void MapLiteRelocalizer::addScan(const LidarScan& scan) {
    if (!initialized_) {
        std::cerr << "MapLiteRelocalizer not initialized" << std::endl;
        return;
    }
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        scan_queue_.push(scan);
    }
    queue_cv_.notify_one();
}

std::optional<RelocalizationResult> MapLiteRelocalizer::attemptRelocalization() {
    if (!initialized_ || !is_active_.load()) {
        return std::nullopt;
    }
    
    // Get the most recent scan for immediate relocalization attempt
    LidarScan current_scan;
    bool has_scan = false;
    
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        if (!scan_queue_.empty()) {
            current_scan = scan_queue_.back();
            has_scan = true;
        }
    }
    
    if (!has_scan) {
        return std::nullopt;
    }
    
    return performICP(current_scan);
}

size_t MapLiteRelocalizer::getSubmapSize() const {
    std::lock_guard<std::mutex> lock(submap_mutex_);
    return submap_scans_.size();
}

void MapLiteRelocalizer::setConfig(const MapLiteConfig& config) {
    config_ = config;
    
#ifdef WITH_PCL
    // Update ICP parameters
    icp_.setMaxCorrespondenceDistance(config_.icp_max_correspondence_distance);
    icp_.setMaximumIterations(config_.icp_max_iterations);
    icp_.setTransformationEpsilon(config_.icp_transformation_epsilon);
    icp_.setEuclideanFitnessEpsilon(config_.icp_fitness_epsilon);
    
    // Update voxel filter
    if (config_.enable_voxel_filter) {
        voxel_filter_.setLeafSize(config_.voxel_leaf_size, 
                                 config_.voxel_leaf_size, 
                                 config_.voxel_leaf_size);
    }
#endif
}

std::optional<RelocalizationResult> MapLiteRelocalizer::getLastResult() const {
    std::lock_guard<std::mutex> lock(result_mutex_);
    return last_result_;
}

void MapLiteRelocalizer::workerLoop() {
    while (!should_shutdown_.load()) {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        queue_cv_.wait(lock, [this] { return !scan_queue_.empty() || should_shutdown_.load(); });
        
        if (should_shutdown_.load()) {
            break;
        }
        
        if (!scan_queue_.empty()) {
            LidarScan scan = scan_queue_.front();
            scan_queue_.pop();
            lock.unlock();
            
            processScan(scan);
        } else {
            lock.unlock();
        }
    }
}

void MapLiteRelocalizer::processScan(const LidarScan& scan) {
    // Update submap with new scan
    updateSubmap(scan);
    
    // Attempt relocalization if conditions are met
    if (is_active_.load() && shouldAttemptRelocalization()) {
        auto result = performICP(scan);
        
        if (result && result->success) {
            {
                std::lock_guard<std::mutex> lock(result_mutex_);
                last_result_ = result;
            }
            
            last_attempt_time_ = std::chrono::steady_clock::now();
            
            // Call callback if set
            if (callback_) {
                callback_(*result);
            }
        }
    }
}

void MapLiteRelocalizer::updateSubmap(const LidarScan& scan) {
    std::lock_guard<std::mutex> lock(submap_mutex_);
    
    // Add scan to submap
    submap_scans_.push_back(scan);
    
    // Update submap center (exponential moving average)
    double alpha = 0.1; // smoothing factor
    submap_center_ = (1.0 - alpha) * submap_center_ + alpha * scan.position;
    
    // Remove old scans beyond radius or exceeding max count
    auto it = submap_scans_.begin();
    while (it != submap_scans_.end()) {
        double distance = computeDistance(it->position, submap_center_);
        if (distance > config_.submap_radius || 
            submap_scans_.size() > config_.max_scans_in_submap) {
            it = submap_scans_.erase(it);
        } else {
            ++it;
        }
    }
}

std::optional<RelocalizationResult> MapLiteRelocalizer::performICP(const LidarScan& current_scan) {
    std::lock_guard<std::mutex> lock(submap_mutex_);
    
    if (submap_scans_.empty()) {
        return std::nullopt;
    }
    
#ifdef WITH_PCL
    // Create submap point cloud
    auto submap_cloud = createSubmapCloud();
    if (submap_cloud->empty()) {
        return std::nullopt;
    }
    
    // Downsample current scan
    auto current_cloud = current_scan.pointcloud;
    if (config_.enable_voxel_filter) {
        downsamplePointCloud(current_cloud);
    }
    
    if (current_cloud->empty()) {
        return std::nullopt;
    }
    
    // Perform ICP
    icp_.setInputSource(current_cloud);
    icp_.setInputTarget(submap_cloud);
    
    pcl::PointCloud<pcl::PointXYZ> aligned_cloud;
    icp_.align(aligned_cloud);
    
    if (!icp_.hasConverged()) {
        return std::nullopt;
    }
    
    // Get transformation matrix
    Eigen::Matrix4f transformation = icp_.getFinalTransformation();
    double fitness_score = icp_.getFitnessScore();
    
    // Check if match is good enough
    if (fitness_score > config_.match_score_threshold) {
        return std::nullopt;
    }
    
    // Convert to Isometry3d
    Eigen::Isometry3d transform;
    transform.matrix() = transformation.cast<double>();
    
    // Count inliers (points within correspondence distance)
    int inlier_count = 0;
    for (size_t i = 0; i < aligned_cloud.size(); ++i) {
        double dist = std::sqrt(
            std::pow(aligned_cloud[i].x - current_cloud->at(i).x, 2) +
            std::pow(aligned_cloud[i].y - current_cloud->at(i).y, 2) +
            std::pow(aligned_cloud[i].z - current_cloud->at(i).z, 2)
        );
        if (dist < config_.icp_max_correspondence_distance) {
            ++inlier_count;
        }
    }
    
    return RelocalizationResult(true, transform, fitness_score, 
                               current_scan.timestamp, inlier_count);
#else
    // Fallback implementation without PCL
    auto submap_points = createSubmapPoints();
    if (submap_points.empty()) {
        return std::nullopt;
    }
    
    auto current_points = current_scan.points;
    if (config_.enable_voxel_filter) {
        current_points = downsamplePoints(current_points);
    }
    
    if (current_points.empty()) {
        return std::nullopt;
    }
    
    // Simple point-to-point ICP implementation
    // This is a basic version - in production, use PCL or a proper ICP library
    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    const int max_iterations = config_.icp_max_iterations;
    double prev_error = std::numeric_limits<double>::max();
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Transform current points
        std::vector<Eigen::Vector3d> transformed_points;
        for (const auto& point : current_points) {
            transformed_points.push_back(transform * point);
        }
        
        // Find correspondences
        double total_error = 0.0;
        int correspondences = 0;
        
        for (const auto& point : transformed_points) {
            double min_dist = std::numeric_limits<double>::max();
            for (const auto& submap_point : submap_points) {
                double dist = computeDistance(point, submap_point);
                if (dist < min_dist) {
                    min_dist = dist;
                }
            }
            if (min_dist < config_.icp_max_correspondence_distance) {
                total_error += min_dist;
                ++correspondences;
            }
        }
        
        double avg_error = correspondences > 0 ? total_error / correspondences : total_error;
        
        // Check convergence
        if (std::abs(prev_error - avg_error) < config_.icp_transformation_epsilon) {
            break;
        }
        prev_error = avg_error;
        
        // Simple update (in real implementation, compute proper transformation)
        // For now, just use identity as placeholder
        break;
    }
    
    // Simple scoring based on average error
    double match_score = prev_error;
    
    if (match_score > config_.match_score_threshold) {
        return std::nullopt;
    }
    
    return RelocalizationResult(true, transform, match_score, 
                               current_scan.timestamp, current_points.size());
#endif
}

bool MapLiteRelocalizer::shouldAttemptRelocalization() const {
    auto now = std::chrono::steady_clock::now();
    auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - last_attempt_time_);
    return elapsed.count() >= config_.min_time_between_attempts;
}

bool MapLiteRelocalizer::isLoopClosureCandidate(const LidarScan& scan) const {
    if (!config_.enable_loop_closure || submap_scans_.empty()) {
        return false;
    }
    
    // Check if we're revisiting an area within the loop closure threshold
    for (const auto& submap_scan : submap_scans_) {
        double distance = computeDistance(scan.position, submap_scan.position);
        if (distance < config_.loop_closure_distance_threshold) {
            return true;
        }
    }
    
    return false;
}

Eigen::Isometry3d MapLiteRelocalizer::poseToTransform(
    const Eigen::Vector3d& position, 
    const Eigen::Quaterniond& orientation) const {
    Eigen::Isometry3d transform = Eigen::Isometry3d::Identity();
    transform.translation() = position;
    transform.linear() = orientation.toRotationMatrix();
    return transform;
}

void MapLiteRelocalizer::transformPointCloud(
    const LidarScan& scan, 
    const Eigen::Isometry3d& transform,
    LidarScan& transformed_scan) const {
    transformed_scan = scan;
    transformed_scan.position = transform * scan.position;
    transformed_scan.orientation = transform.linear() * scan.orientation;
    
#ifdef WITH_PCL
    if (scan.pointcloud && !scan.pointcloud->empty()) {
        transformed_scan.pointcloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(
            new pcl::PointCloud<pcl::PointXYZ>);
        transformed_scan.pointcloud->resize(scan.pointcloud->size());
        
        Eigen::Matrix4f transform_matrix = transform.matrix().cast<float>();
        for (size_t i = 0; i < scan.pointcloud->size(); ++i) {
            const auto& point = scan.pointcloud->at(i);
            Eigen::Vector4f homogeneous(point.x, point.y, point.z, 1.0f);
            Eigen::Vector4f transformed = transform_matrix * homogeneous;
            transformed_scan.pointcloud->at(i).x = transformed[0];
            transformed_scan.pointcloud->at(i).y = transformed[1];
            transformed_scan.pointcloud->at(i).z = transformed[2];
        }
    }
#else
    transformed_scan.points.clear();
    transformed_scan.points.reserve(scan.points.size());
    for (const auto& point : scan.points) {
        transformed_scan.points.push_back(transform * point);
    }
#endif
}

double MapLiteRelocalizer::computeDistance(
    const Eigen::Vector3d& p1, 
    const Eigen::Vector3d& p2) const {
    return (p1 - p2).norm();
}

#ifdef WITH_PCL
void MapLiteRelocalizer::downsamplePointCloud(
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) const {
    if (!config_.enable_voxel_filter || cloud->empty()) {
        return;
    }
    
    voxel_filter_.setInputCloud(cloud);
    voxel_filter_.filter(*cloud);
}

pcl::PointCloud<pcl::PointXYZ>::Ptr MapLiteRelocalizer::createSubmapCloud() const {
    auto submap_cloud = pcl::PointCloud<pcl::PointXYZ>::Ptr(
        new pcl::PointCloud<pcl::PointXYZ>);
    
    for (const auto& scan : submap_scans_) {
        if (scan.pointcloud && !scan.pointcloud->empty()) {
            // Transform scan to submap frame and add to cloud
            Eigen::Isometry3d transform = poseToTransform(scan.position, scan.orientation);
            
            for (const auto& point : *scan.pointcloud) {
                Eigen::Vector4f homogeneous(point.x, point.y, point.z, 1.0f);
                Eigen::Vector4f transformed = transform.matrix().cast<float>() * homogeneous;
                
                pcl::PointXYZ transformed_point;
                transformed_point.x = transformed[0];
                transformed_point.y = transformed[1];
                transformed_point.z = transformed[2];
                
                submap_cloud->push_back(transformed_point);
            }
        }
    }
    
    if (config_.enable_voxel_filter) {
        downsamplePointCloud(submap_cloud);
    }
    
    return submap_cloud;
}
#else
std::vector<Eigen::Vector3d> MapLiteRelocalizer::downsamplePoints(
    const std::vector<Eigen::Vector3d>& points) const {
    if (!config_.enable_voxel_filter || points.empty()) {
        return points;
    }
    
    // Simple voxel grid downsampling
    std::unordered_set<uint64_t> voxel_set;
    std::vector<Eigen::Vector3d> downsampled;
    
    double voxel_size = config_.voxel_leaf_size;
    
    for (const auto& point : points) {
        int voxel_x = static_cast<int>(std::floor(point.x() / voxel_size));
        int voxel_y = static_cast<int>(std::floor(point.y() / voxel_size));
        int voxel_z = static_cast<int>(std::floor(point.z() / voxel_size));
        
        uint64_t voxel_key = (static_cast<uint64_t>(voxel_x) << 42) |
                            (static_cast<uint64_t>(voxel_y) << 21) |
                            static_cast<uint64_t>(voxel_z);
        
        if (voxel_set.find(voxel_key) == voxel_set.end()) {
            voxel_set.insert(voxel_key);
            downsampled.push_back(point);
        }
    }
    
    return downsampled;
}

std::vector<Eigen::Vector3d> MapLiteRelocalizer::createSubmapPoints() const {
    std::vector<Eigen::Vector3d> submap_points;
    
    for (const auto& scan : submap_scans_) {
        if (!scan.points.empty()) {
            // Transform scan to submap frame and add to points
            Eigen::Isometry3d transform = poseToTransform(scan.position, scan.orientation);
            
            for (const auto& point : scan.points) {
                submap_points.push_back(transform * point);
            }
        }
    }
    
    if (config_.enable_voxel_filter) {
        submap_points = downsamplePoints(submap_points);
    }
    
    return submap_points;
}
#endif

} // namespace axis
