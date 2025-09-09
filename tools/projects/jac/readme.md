# data extraction

## brief introduce

Extraction pcd data from rosbag's .bag files. Needing pypcd(or open3d) to save point cloud data and ros packages to extract the data from our .bag files. For every bag file, there are msg structure to save the data and timestmaps. So we can use the above packages to extract the data.

## installation

we need to install the rosbag and pypcd packages to run the tool.

### how to install the pypcd packages

 git clone https://github.com/dimatura/pypcd
 cd pypcd
 git fetch origin pull/9/head:python3
 git checkout python3
 python3 setup.py install --user

 ### how to install the open3d packages

  pip install open3d

### how to install the rosbag packages

 we install the ros python interface here.
 the https://rospypi.github.io/simple/ may have some problem to access, we can download the .whl file form this site and install it instead.

 conda install -c conda-forge ros-roslz4
 pip install --extra-index-url https://rospypi.github.io/simple/ rospy rosbag
 pip install cvbridge3
 pip install rosbag roslz4 ros_numpy std_msgs sensor_msgs geometry-msgs visualization-msgs actionlib-msgs nav_msgs tf2_msgs cv_bridge --extra-index-url https://rospypi.github.io/simple/

 I'm not sure if the numpy version had confilct, I just list my numpy version
 pip install numpy==1.23.3


### example to use the tool

 cd the path of tool and use the tool 
 python rosbagParsing.py -l path/to/input_file path/to/output_file

### ps

the enviroment installation are complex, so we also create a docker to store the tools, you can down the image from the above link
https://softwaremotion.feishu.cn/wiki/GUyMwDIuuitbirkLvOocBiEonbf
the document of this docker image, you can get the detail usage of these code:
https://softwaremotion.feishu.cn/wiki/DEkawLhEWiTl0FkLKnucJ7DRnkd?fromScene=spaceOverview