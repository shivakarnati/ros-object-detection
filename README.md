This repository contains ROS-Humble based object detection.\
-> Record the video in .mcap file\
-> Run your .mcap file using ```ros2 bag play -s mcap rec1_0.mcap``` command\
-> Run the following commands to run ros program

```
colcon build
source ./install/setup.bash
ros2 run object_det object_det_node --ros-args -p image_subscription:="/oakd/rgb/preview/image_raw"
```
