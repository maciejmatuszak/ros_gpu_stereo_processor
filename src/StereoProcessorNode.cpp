#include "gpuimageproc/StereoProcessor.h"
#include <ros/ros.h>

int main(int argc, char **argv)
{
    /**
     * The ros::init() function needs to see argc and argv so that it can perform
     * any ROS arguments and name remapping that were provided at the command line.
     * For programmatic remappings you can use a different version of init() which takes
     * remappings directly, but for most command-line programs, passing argc and argv is
     * the easiest way to do it.  The third argument to init() is the name of the node.
     *
     * You must call one of the versions of ros::init() before using any other
     * part of the ROS system.
     */
    ros::init(argc, argv, "gpuimageproc");

    /**
     * NodeHandle is the main access point to communications with the ROS system.
     * The first NodeHandle constructed will fully initialize this node, and the last
     * NodeHandle destructed will close down the node.
     */
    ros::NodeHandle nh;
    ros::NodeHandle private_nh("~");

    if (true)
    {
        private_nh.setParam("camera_info_file_left", "/data/git/temp_ws/src/gpuimageproc/test/stereobm/test_data/left.yaml");
        private_nh.setParam("camera_info_file_right", "/data/git/temp_ws/src/gpuimageproc/test/stereobm/test_data/right.yaml");
    }

    gpuimageproc::StereoProcessor processor(nh, private_nh);
    ros::spin();
}
