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


# Mediapipe installation
$ sudo apt-get install python3.8-minimal python3.8-venv python3-pip

$ sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1

$ sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.6 2

$ sudo update-alternatives --install /usr/bin/python python /usr/bin/python2 3

$ sudo update-alternatives --config python

There are 2 choices for the alternative python (providing /usr/bin/python).

```
  Selection    Path                Priority   Status
------------------------------------------------------------
  0            /usr/bin/python2     1         auto mode
* 1            /usr/bin/python2     1         manual mode
  2            /usr/bin/python3.6   2         manual mode
  3            /usr/bin/python3.8   3         manual mode

Press <enter> to keep the current choice[*], or type selection number: 3
```

$ python -m pip install --upgrade pip

$ sudo apt-get remove python3-matplotlib
$ python -m pip install matplotlib

$ python -m pip install mediapipe

$ sudo update-alternatives --config python

```
There are 2 choices for the alternative python (providing /usr/bin/python).

  Selection    Path                Priority   Status
------------------------------------------------------------
  0            /usr/bin/python2     1         auto mode
  1            /usr/bin/python2     1         manual mode
  2            /usr/bin/python3.6   2         manual mode
* 3            /usr/bin/python3.8   3         manual mode

Press <enter> to keep the current choice[*], or type selection number: 1
```
