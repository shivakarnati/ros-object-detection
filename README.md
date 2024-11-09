# ROS-Humble Object Detection

This repository contains a **ROS-Humble** based object detection project. It allows you to record videos, replay them, and run object detection on the recorded data using ROS 2 and an object detection node.

## Table of Contents

- [About the Project](#about-the-project)
- [Usage Instructions](#usage-instructions)
  - [1. Record Video in .mcap Format](#1-record-video-in-mcap-format)
  - [2. Play Recorded .mcap File](#2-play-recorded-mcap-file)
  - [3. Run Object Detection Node](#3-run-object-detection-node)
- [Setup and Installation](#setup-and-installation)
- [ROS Commands](#ros-commands)
- [Contributing](#contributing)
- [License](#license)

---

## About the Project

This project is a **ROS-Humble** based object detection application that uses a pre-trained object detection model. The system supports recording videos in `.mcap` format, replaying them, and running an object detection node that processes the frames and identifies objects in real-time.

## Usage Instructions

### 1. Record Video in .mcap Format

To record a video in `.mcap` format, use a ROS 2-compatible recording tool. Ensure that your ROS setup is running and configured to capture the desired video stream.

### 2. Play Recorded .mcap File

Once you have recorded a video, you can replay it using the following command:

```bash
ros2 bag play -s mcap rec1_0.mcap
```
### 3. Run Object Detection Node
```bash
colcon build
```
### 4. Source the environment to make ROS packages available:
```bash
source ./install/setup.bash
```
### 5. Run the object detection node with the following command:
```bash
ros2 run object_det object_det_node --ros-args -p image_subscription:="/oakd/rgb/preview/image_raw"
```
