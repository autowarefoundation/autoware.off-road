#!/bin/bash
# Remove Isaac Sim from LD_LIBRARY_PATH to avoid spdlog conflicts with ROS2 Humble
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | sed -e 's|/home/autoware/isaacsim6/[^:]*:||g' -e 's|:/home/autoware/isaacsim6/[^:]*||g')
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
python3 CARLA_mining_demo.py "$@"
