#ifndef GAP_FOLLOWER_HPP
#define GAP_FOLLOWER_HPP

#include <vector>
#include <utility>
#include <iostream>
#include <cmath>
#include <unordered_map>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <chrono>

struct GapFollowerParams {
    // Parameters for BEV simulation mapping and Gap Follower
    double bev_resolution = 0.05; // meters per pixel (20 px/m)
    int bev_ego_x = 400;          // pixel x coordinate of ego (middle of 40m)
    int bev_ego_y = 2000;         // pixel y coordinate of ego (bottom of 100m)
    double raycast_angle_min = -M_PI;
    double raycast_angle_max = M_PI;
    double raycast_angle_increment = M_PI / 180.0; // 1 degree
    double raycast_max_range = 30.0; // in meters
    double min_gap_size = 5.0;
    double range_thresh = 15.0;
    double goal_angle = 0.0;
};

struct SimulatedScan {
    std::vector<float> ranges;
    float angle_min;
    float angle_max;
    float angle_increment;
    float range_max;
    float range_min;
};

class GapFollower {
public:
    GapFollower(const GapFollowerParams& params);
    ~GapFollower() = default;

    // Process a BEV contour image (drivable area = 255/white, non-drivable = 0/black)
    void processBEVContour(const cv::Mat& bev_image);

    // Get the indices of the target gap for visualization
    std::pair<int, int> getTargetGapIndices() const { return target_gap_indices_; }
    const std::vector<std::pair<int, int>>& getAllGaps() const { return gap_indices_; }
    const SimulatedScan& getLastScan() const { return last_scan_; }
    const GapFollowerParams& getParams() const { return params_; }

private:
    SimulatedScan simulateLaserScan(const cv::Mat& bev_image);
    
    // Find all gaps in the scan and pick the best one towards the goal
    void findBestGap(const SimulatedScan& scan);
    
    void normalizeGapAngle(float& gap_angle);

    GapFollowerParams params_;
    SimulatedScan last_scan_;

    float last_gap_size_;
    bool deadend_;
    
    std::vector<std::pair<int, int>> gap_indices_;
    std::pair<int, int> target_gap_indices_;

    std::chrono::steady_clock::time_point velocitytimestamp_;
    std::chrono::steady_clock::time_point plannertimestamp_;
};

extern "C" {
    void* create_gap_follower(const GapFollowerParams* params);
    void destroy_gap_follower(void* handle);
    void process_bev_contour(void* handle, unsigned char* data, int rows, int cols, 
                            int* gap_idx1, int* gap_idx2,
                            int* all_gaps, int* num_gaps, int max_gaps);
}

#endif // GAP_FOLLOWER_HPP
