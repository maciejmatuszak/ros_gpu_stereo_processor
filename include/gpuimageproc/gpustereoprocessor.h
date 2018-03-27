#pragma once
#include "gpuimageproc/gpu_sender.h"
#include <image_geometry/stereo_camera_model.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudastereo.hpp>
#include <opencv2/cudawarping.hpp>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <stereo_msgs/DisparityImage.h>

namespace gpuimageproc
{

enum GpuMatSource
{
    GPU_MAT_SIDE_L = 0,
    GPU_MAT_SIDE_R = 1,
    GPU_MAT_SIDE_MASK = GPU_MAT_SIDE_L | GPU_MAT_SIDE_R,

    GPU_MAT_SRC_RAW = 1 << 1,
    GPU_MAT_SRC_MONO = 1 << 2,
    GPU_MAT_SRC_COLOR = 1 << 3,
    GPU_MAT_SRC_RECT_MONO = 1 << 4,
    GPU_MAT_SRC_RECT_COLOR = 1 << 5,
    GPU_MAT_SRC_MASK = GPU_MAT_SRC_RAW | GPU_MAT_SRC_MONO | GPU_MAT_SRC_COLOR | GPU_MAT_SRC_RECT_MONO | GPU_MAT_SRC_RECT_COLOR,

    GPU_MAT_SRC_L_RAW = GPU_MAT_SRC_RAW | GPU_MAT_SIDE_L,
    GPU_MAT_SRC_R_RAW = GPU_MAT_SRC_RAW | GPU_MAT_SIDE_R,
    GPU_MAT_SRC_L_MONO = GPU_MAT_SRC_MONO | GPU_MAT_SIDE_L,
    GPU_MAT_SRC_R_MONO = GPU_MAT_SRC_MONO | GPU_MAT_SIDE_R,
    GPU_MAT_SRC_L_COLOR = GPU_MAT_SRC_COLOR | GPU_MAT_SIDE_L,
    GPU_MAT_SRC_R_COLOR = GPU_MAT_SRC_COLOR | GPU_MAT_SIDE_R,
    GPU_MAT_SRC_L_RECT_MONO = GPU_MAT_SRC_RECT_MONO | GPU_MAT_SIDE_L,
    GPU_MAT_SRC_R_RECT_MONO = GPU_MAT_SRC_RECT_MONO | GPU_MAT_SIDE_R,
    GPU_MAT_SRC_L_RECT_COLOR = GPU_MAT_SRC_RECT_COLOR | GPU_MAT_SIDE_L,
    GPU_MAT_SRC_R_RECT_COLOR = GPU_MAT_SRC_RECT_COLOR | GPU_MAT_SIDE_R
};

class GpuStereoProcessor
{

    GpuStereoProcessor();

  public:
    void initStereoModel(const sensor_msgs::CameraInfoConstPtr &l_info_msg, const sensor_msgs::CameraInfoConstPtr &r_info_msg);
    void uploadMat(GpuMatSource mat_source, const cv::Mat &cv_mat);
    void enqueueSendImage(GpuMatSource source, const sensor_msgs::ImageConstPtr &imagePattern, std::string encoding, ros::Publisher &pub);
    void enqueueSendDisparity(const sensor_msgs::ImageConstPtr &imagePattern, ros::Publisher &pub);
    void colorConvertImage(GpuMatSource source, GpuMatSource dest, int colorConversion, int dcn);
    void rectifyImage(GpuMatSource source, GpuMatSource dest, cv::InterpolationFlags interpolation);
    void computeDisparity();

  protected:
    cv::cuda::GpuMat &getGpuMat(GpuMatSource source);
    cv::cuda::Stream &getStream(GpuMatSource source);

    std::vector<GPUSender::Ptr> senders;
    cv::cuda::Stream l_strm, r_strm;
    cv::cuda::GpuMat l_raw, r_raw;
    cv::cuda::GpuMat l_mono, r_mono;
    cv::cuda::GpuMat l_color, r_color;
    cv::cuda::GpuMat l_rect_mono, r_rect_mono;
    cv::cuda::GpuMat l_rect_color, r_rect_color;
    cv::cuda::GpuMat disparity, disparity_32F;

    stereo_msgs::DisparityImagePtr disp_msg_;
    ros::Publisher &pub_disparity_;
    void sendDisparity(void) { pub_disparity_.publish(disp_msg_); }

    cv::cuda::HostMem filter_buf_;

    image_geometry::StereoCameraModel model_;

    cv::Ptr<cv::cuda::StereoBM> block_matcher_;
    cv::Ptr<cv::cuda::DisparityBilateralFilter> bilateral_filter_;
    const int block_matcher_min_disparity_ = 0;

    bool bilateral_filter_enabled_;
    int maxSpeckleSize_;
    double maxDiff_;
};
}
