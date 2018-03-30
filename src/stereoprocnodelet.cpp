#include "gpuimageproc/stereoprocnodelet.h"

namespace gpuimageproc
{

void StereoProcNodelet::onInit()
{
    ros::NodeHandle nh         = getNodeHandle();
    ros::NodeHandle private_nh = getPrivateNodeHandle();
    stereoProcessorPtr         = boost::make_shared<StereoProcessor>(nh, private_nh);
}

} // namespace

// Register nodelet
#include <pluginlib/class_list_macros.h>

PLUGINLIB_EXPORT_CLASS(gpuimageproc::StereoProcNodelet, nodelet::Nodelet)
