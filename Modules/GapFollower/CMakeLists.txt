cmake_minimum_required(VERSION 3.10)
project(GapFollower)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include OpenCV and your own headers
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)

# Add source files
# In a real context, you might want to create a library out of this
# and link it to an executable that tests or runs it.
add_library(${PROJECT_NAME} SHARED
    src/gap_follower.cpp
)

target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# Example test executable
add_executable(test_gap_follower test/test_gap_follower.cpp)
target_link_libraries(test_gap_follower ${PROJECT_NAME} ${OpenCV_LIBS})
