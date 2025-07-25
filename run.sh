#!/bin/bash

# Track background PIDs
PIDS=()

# Cleanup function to kill all background jobs
cleanup() {
    echo "Stopping all processes..."
    for pid in "${PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
    wait
    exit 0
}

# Trap Ctrl+C
trap cleanup SIGINT

# Start PX4 in background
(
  cd ~/PX4-Autopilot || exit
  PX4_GZ_WORLD=line_following_track make px4_sitl gz_x500_mono_cam_down
) &
PIDS+=($!)

# Wait for PX4
sleep 2

# Start Micro XRCE Agent in background
MicroXRCEAgent udp4 -p 8888 &
PIDS+=($!)

# Wait for agent
sleep 2

# Start ROS 2 bridge in background
(
  cd ~
  source /opt/ros/jazzy/setup.bash
  ros2 run ros_gz_bridge parameter_bridge /world/line_following_track/model/x500_mono_cam_down_0/link/camera_link/sensor/imager/image@sensor_msgs/msg/Image@gz.msgs.Image
) &
PIDS+=($!)

# Wait for all background processes
wait
