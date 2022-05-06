# import cv_bridge in ROS
## Installation of dependencies
``` bash
sudo apt-get install ros-melodic-cv-bridge
```
## Creat a new workspace for cv_bridge
```bash
cd ~
mkdir -p cv_bridge_ws/src && cd cv_bridge_ws/src
catkin_init_workspace
git clone https://github.com/ros-perception/vision_opencv.git
cd ../
catkin_make install -DPYTHON_EXECUTABLE=/usr/bin/python3
```
## Add the compiled file to the environment
```bash
source /home/xxx/cv_bridge_ws/install/setup.bash --extend
```
Don't forget to change "xxx" to your own file path!
If you don't want to source it every time, you can add it to .bashrc file.