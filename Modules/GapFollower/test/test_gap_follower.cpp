#include <iostream>
#include <opencv2/opencv.hpp>
#include "gap_follower.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <topdown_image_path>" << std::endl;
        return 1;
    }

    std::string image_path = argv[1];
    cv::Mat bev_image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
    if (bev_image.empty()) {
        std::cerr << "Error: Could not read image " << image_path << std::endl;
        return 1;
    }

    // Initialize parameters
    GapFollowerParams params;
    params.bev_resolution = 0.05;
    params.bev_ego_x = 400;
    params.bev_ego_y = 2000;
    params.raycast_max_range = 30.0;
    params.min_gap_size = 5.0;
    params.range_thresh = 15.0;
    params.goal_angle = 0.0;

    // Create GapFollower instance
    GapFollower gap_follower(params);

    // Process the image
    std::cout << "Processing image: " << image_path << std::endl;
    gap_follower.processBEVContour(bev_image);

    // Print the result
    std::cout << "Results:" << std::endl;
    std::pair<int, int> gap_indices = gap_follower.getTargetGapIndices();
    std::cout << "  Target Gap Indices: [" << gap_indices.first << ", " << gap_indices.second << "]" << std::endl;

    return 0;
}
