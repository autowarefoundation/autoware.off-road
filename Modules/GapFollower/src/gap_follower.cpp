#include "gap_follower.hpp"

GapFollower::GapFollower(const GapFollowerParams& params) 
    : params_(params),
      last_gap_size_(0.0f),
      deadend_(false) 
{
    velocitytimestamp_ = std::chrono::steady_clock::now();
    plannertimestamp_ = std::chrono::steady_clock::now();
}

void GapFollower::processBEVContour(const cv::Mat& bev_image) {
    plannertimestamp_ = std::chrono::steady_clock::now();

    last_scan_ = simulateLaserScan(bev_image);
    
    findBestGap(last_scan_);

    auto duration = std::chrono::steady_clock::now() - plannertimestamp_;
    double time_lapse = std::chrono::duration<double, std::milli>(duration).count();
    // std::cout << "planner process time: " << time_lapse << " ms\n";
}

SimulatedScan GapFollower::simulateLaserScan(const cv::Mat& bev_image) {
    SimulatedScan scan;
    scan.angle_min = params_.raycast_angle_min;
    scan.angle_max = params_.raycast_angle_max;
    scan.angle_increment = params_.raycast_angle_increment;
    scan.range_min = 0.0;
    scan.range_max = params_.raycast_max_range;

    int num_rays = std::ceil((scan.angle_max - scan.angle_min) / scan.angle_increment);
    scan.ranges.resize(num_rays, scan.range_max);

    int width = bev_image.cols;
    int height = bev_image.rows;

    for (int i = 0; i < num_rays; ++i) {
        float angle = scan.angle_min + i * scan.angle_increment;
        // In image coordinates, Y is usually down, X is right.
        // Assuming angle 0 is straight ahead (negative Y in image)
        // In standard ROS LiDAR: +angle is Left, -angle is Right.
        // Image X: 0 is left, width is right.
        // So +angle (Left) means decreasing X (dx < 0)
        // So -angle (Right) means increasing X (dx > 0)
        float dx = -std::sin(angle);
        float dy = -std::cos(angle);

        float max_range_pixels = scan.range_max / params_.bev_resolution;
        float ray_range = scan.range_max;

        for (int r = 1; r <= max_range_pixels; ++r) {
            int px = params_.bev_ego_x + static_cast<int>(std::round(r * dx));
            int py = params_.bev_ego_y + static_cast<int>(std::round(r * dy));

            if (px < 0 || px >= width || py < 0 || py >= height) {
                ray_range = scan.range_max;
                break;
            }

            // Assume 1 channel grayscale where 0 is non-drivable, >0 is drivable
            // Or if it's 3 channel, check intensity. Handle 1-channel for simplicity here.
            uint8_t pixel_val = 0;
            if (bev_image.channels() == 1) {
                pixel_val = bev_image.at<uint8_t>(py, px);
            } else if (bev_image.channels() == 3) {
                cv::Vec3b col = bev_image.at<cv::Vec3b>(py, px);
                pixel_val = col[0]; // Just check one channel
            }

            if (pixel_val >= 128) { // hit the drivable contour (white/bright pixel)
                ray_range = r * params_.bev_resolution;
                break;
            }
        }
        scan.ranges[i] = ray_range;
    }

    return scan;
}

void GapFollower::findBestGap(const SimulatedScan& scan) {
    const float min_gap_size_val = params_.min_gap_size;
    const int scan_size = scan.ranges.size();
    const float angle_increment = scan.angle_increment;
    const float angle_min = scan.angle_min;
    const float angle_max = scan.angle_max;
    
    std::vector<float> scan_ranges = scan.ranges;
    std::vector<int> scan_ids(scan_size, 0);

    if (scan_ranges.empty()) return;

    scan_ranges[0] = std::min(scan_ranges[0], 2.0f * min_gap_size_val);
    scan_ranges[scan_size - 1] = std::min(scan_ranges[scan_size - 1], 2.0f * min_gap_size_val);

    int new_id = 1;
    for (int i = 0; i < (int)scan_ranges.size(); i++) {
        if (scan_ranges[i] > params_.range_thresh) continue;

        if (scan_ids[i] == 0) {
            scan_ids[i] = new_id;
            new_id++;
        }

        const float check_limit = min_gap_size_val < scan_ranges[i] ? 
            std::min(static_cast<float>(angle_min + i * angle_increment + std::asin(min_gap_size_val / scan_ranges[i])), angle_max) : angle_max;

        for (int j = i + 1; angle_min + j * angle_increment <= check_limit; j++) {
            if (scan_ranges[j] > params_.range_thresh || scan_ids[i] == scan_ids[j]) continue;

            const float dist = std::sqrt(std::pow(scan_ranges[i], 2) + std::pow(scan_ranges[j], 2) -
                                         2 * scan_ranges[i] * scan_ranges[j] * std::cos(std::abs(i - j) * angle_increment));

            if (dist <= min_gap_size_val) {
                if (scan_ids[j] == 0) {
                    scan_ids[j] = scan_ids[i];
                } else {
                    int old_id = scan_ids[j];
                    for (int k = 0; k <= j; k++) {
                        if (scan_ids[k] == old_id) {
                            scan_ids[k] = scan_ids[i];
                        }
                    }
                }
            }
        }
    }

    // Renumber IDs to be 1..N
    std::unordered_map<int, int> id_remap;
    int next_id = 1;
    for (int &id : scan_ids) {
        if (id == 0) continue;
        if (id_remap.find(id) == id_remap.end()) {
            id_remap[id] = next_id++;
        }
        id = id_remap[id];
    }
    int max_id = next_id - 1;

    std::unordered_map<int, std::pair<int, int>> group_map;
    std::unordered_map<int, float> group_min_dist_map;

    for (int i = 0; i < (int)scan_ranges.size(); i++) {
        if (scan_ids[i] == 0) continue;
        int id = scan_ids[i];
        if (group_map.find(id) == group_map.end()) {
            group_map[id] = {i, i};
            group_min_dist_map[id] = scan_ranges[i];
        } else {
            group_map[id].first = std::min(group_map[id].first, i);
            group_map[id].second = std::max(group_map[id].second, i);
            group_min_dist_map[id] = std::min(group_min_dist_map[id], scan_ranges[i]);
        }
    }

    // Remove occluded groups
    std::vector<int> ids_to_remove;
    for (int i = 1; i <= max_id; i++) {
        if (group_map.find(i) == group_map.end()) continue;
        for (int j = i + 1; j <= max_id; j++) {
            if (group_map.find(j) == group_map.end()) continue;
            // Check for overlap in indices
            if (std::max(group_map[i].first, group_map[j].first) <= std::min(group_map[i].second, group_map[j].second)) {
                if (group_min_dist_map[i] < group_min_dist_map[j]) {
                    ids_to_remove.push_back(j);
                } else {
                    ids_to_remove.push_back(i);
                }
            }
        }
    }

    for (int id : ids_to_remove) {
        group_map.erase(id);
    }

    std::vector<std::pair<int, int>> groups;
    for (auto const& [id, range] : group_map) {
        groups.push_back(range);
    }
    std::sort(groups.begin(), groups.end());

    if (!groups.empty()) {
        deadend_ = (groups[0].first == 0 && groups[0].second == scan_size - 1);
    } else {
        deadend_ = false;
    }

    gap_indices_.clear();
    if (groups.size() > 1) {
        for (size_t i = 0; i < groups.size() - 1; i++) {
            gap_indices_.push_back({groups[i].second, groups[i + 1].first});
        }
    } else if (groups.empty()) {
        int mid = scan_size / 2;
        gap_indices_.push_back({std::max(0, mid - 30), std::min(scan_size - 1, mid + 30)});
    }

    if (gap_indices_.size() > 2) {
        gap_indices_.erase(gap_indices_.begin());
        gap_indices_.pop_back();
    }

    // Pick target gap closest to goal angle
    const float goal_angle = params_.goal_angle;
    float min_angle_diff = std::numeric_limits<float>::max();
    for (auto const& it : gap_indices_) {
        float angle_first = angle_min + it.first * angle_increment;
        float angle_second = angle_min + it.second * angle_increment;
        float gap_center_angle = (angle_first + angle_second) / 2.0f;
        float min_diff = std::abs(gap_center_angle - goal_angle);
        if (min_diff < min_angle_diff) {
            min_angle_diff = min_diff;
            target_gap_indices_ = it;
        }
    }

    float target_angle = angle_min + (target_gap_indices_.first + target_gap_indices_.second) / 2.0f * angle_increment;
    float avg_range = (scan.ranges[target_gap_indices_.first] + scan.ranges[target_gap_indices_.second]) / 2.0f;

    std::cout << "[GapFollower] Found " << gap_indices_.size() << " gaps. "
              << "Target indices: [" << target_gap_indices_.first << ", " << target_gap_indices_.second << "], "
              << "Angle: " << target_angle * 180.0 / M_PI << " deg, "
              << "Avg Range: " << avg_range << "m\n";
}

void GapFollower::normalizeGapAngle(float& gap_angle) {
    gap_angle = std::fmod(gap_angle + 3.0f * M_PI / 2.0f, 2.0f * M_PI);
    if (gap_angle < 0) gap_angle += 2.0f * M_PI;
    gap_angle -= M_PI;
}

extern "C" {
    void* create_gap_follower(const GapFollowerParams* params) {
        if (!params) return nullptr;
        return new GapFollower(*params);
    }

    void destroy_gap_follower(void* handle) {
        if (handle) {
            delete static_cast<GapFollower*>(handle);
        }
    }

    void process_bev_contour(void* handle, unsigned char* data, int rows, int cols, 
                            int* gap_idx1, int* gap_idx2,
                            int* all_gaps, int* num_gaps, int max_gaps) {
        if (!handle || !data) return;
        
        GapFollower* gf = static_cast<GapFollower*>(handle);
        
        // Create OpenCV Mat from the raw pixel data (assuming 1-channel grayscale)
        cv::Mat bev_image(rows, cols, CV_8UC1, data);
        
        gf->processBEVContour(bev_image);
        
        std::pair<int, int> indices = gf->getTargetGapIndices();
        if (gap_idx1) *gap_idx1 = indices.first;
        if (gap_idx2) *gap_idx2 = indices.second;

        if (all_gaps && num_gaps) {
            const auto& gaps = gf->getAllGaps();
            const auto& scan = gf->getLastScan();
            const auto& params = gf->getParams();
            int count = std::min((int)gaps.size(), max_gaps);
            *num_gaps = count;
            for (int i = 0; i < count; ++i) {
                int g1 = gaps[i].first;
                int g2 = gaps[i].second;
                
                float angle1 = scan.angle_min + g1 * scan.angle_increment;
                float r1 = scan.ranges[g1] / params.bev_resolution;
                int px1 = params.bev_ego_x + std::round(r1 * -std::sin(angle1));
                int py1 = params.bev_ego_y + std::round(r1 * -std::cos(angle1));
                
                float angle2 = scan.angle_min + g2 * scan.angle_increment;
                float r2 = scan.ranges[g2] / params.bev_resolution;
                int px2 = params.bev_ego_x + std::round(r2 * -std::sin(angle2));
                int py2 = params.bev_ego_y + std::round(r2 * -std::cos(angle2));
                
                all_gaps[i * 6 + 0] = g1;
                all_gaps[i * 6 + 1] = g2;
                all_gaps[i * 6 + 2] = px1;
                all_gaps[i * 6 + 3] = py1;
                all_gaps[i * 6 + 4] = px2;
                all_gaps[i * 6 + 5] = py2;
            }
        }
    }
}
