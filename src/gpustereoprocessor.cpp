#include "gpuimageproc/gpustereoprocessor.h"

namespace gpuimageproc
{

GpuStereoProcessor::GpuStereoProcessor() { cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice()); }

void GpuStereoProcessor::initStereoModel(const sensor_msgs::CameraInfoConstPtr &l_info_msg, const sensor_msgs::CameraInfoConstPtr &r_info_msg)
{
    // Update the camera model
    model_.fromCameraInfo(l_info_msg, r_info_msg);
}

void GpuStereoProcessor::uploadMat(GpuMatSource mat_source, const cv::Mat &cv_mat) { getGpuMat(mat_source).upload(cv_mat, getStream(mat_source)); }

cv::cuda::GpuMat &GpuStereoProcessor::getGpuMat(GpuMatSource source)
{
    switch (source)
    {
    case GpuMatSource::GPU_MAT_SRC_L_MONO:
        return l_mono;
    case GpuMatSource::GPU_MAT_SRC_R_MONO:
        return r_mono;
    default:
        ROS_ERROR("invalid source in getGpuMat: %d", source);
    }
}

cv::cuda::Stream &GpuStereoProcessor::getStream(GpuMatSource source)
{
    if (source & GPU_MAT_SIDE_L == GPU_MAT_SIDE_L)
    {
        return l_strm;
    }
    else// if(source & GPU_MAT_SRC_MASK_R == GPU_MAT_SRC_MASK_R)
    {
        return r_strm;
    }
}

void GpuStereoProcessor::enqueueSendImage(GpuMatSource source, const sensor_msgs::ImageConstPtr &imagePattern, std::string encoding, ros::Publisher &pub)
{
    GPUSender::Ptr t = boost::make_shared<GPUSender>(imagePattern, encoding, pub);
    senders.push_back(t);
    t->enqueueSend(getGpuMat(source), getStream(source));
}

void GpuStereoProcessor::colorConvertImage(GpuMatSource source, GpuMatSource dest, int colorConversion, int dcn)
{
    cv::cuda::demosaicing(getGpuMat(source), getGpuMat(dest), colorConversion, dcn, getStream(source));
}

void GpuStereoProcessor::rectifyImage(GpuMatSource source, GpuMatSource dest, cv::InterpolationFlags interpolation)
{
    model_.left().rectifyImageGPU(getGpuMat(source), getGpuMat(dest), interpolation, getStream(source));
}

} // namespace
