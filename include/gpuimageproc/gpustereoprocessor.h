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
#include <unordered_map>

namespace gpuimageproc
{

enum GpuMatSource
{
    GPU_MAT_SIDE_L    = 1 << 0,
    GPU_MAT_SIDE_R    = 1 << 1,
    GPU_MAT_SIDE_MASK = GPU_MAT_SIDE_L | GPU_MAT_SIDE_R,

    GPU_MAT_SRC_RAW           = 1 << 2,
    GPU_MAT_SRC_MONO          = 1 << 3,
    GPU_MAT_SRC_COLOR         = 1 << 4,
    GPU_MAT_SRC_RECT_MONO     = 1 << 5,
    GPU_MAT_SRC_RECT_COLOR    = 1 << 6,
    GPU_MAT_SRC_DISPARITY     = 1 << 7,
    GPU_MAT_SRC_DISPARITY_32F = 1 << 8,
    GPU_MAT_SRC_MASK = GPU_MAT_SRC_RAW | GPU_MAT_SRC_MONO | GPU_MAT_SRC_COLOR | GPU_MAT_SRC_RECT_MONO | GPU_MAT_SRC_RECT_COLOR | GPU_MAT_SRC_DISPARITY | GPU_MAT_SRC_DISPARITY_32F,

    GPU_MAT_SRC_L_RAW           = GPU_MAT_SRC_RAW | GPU_MAT_SIDE_L,
    GPU_MAT_SRC_R_RAW           = GPU_MAT_SRC_RAW | GPU_MAT_SIDE_R,
    GPU_MAT_SRC_L_MONO          = GPU_MAT_SRC_MONO | GPU_MAT_SIDE_L,
    GPU_MAT_SRC_R_MONO          = GPU_MAT_SRC_MONO | GPU_MAT_SIDE_R,
    GPU_MAT_SRC_L_COLOR         = GPU_MAT_SRC_COLOR | GPU_MAT_SIDE_L,
    GPU_MAT_SRC_R_COLOR         = GPU_MAT_SRC_COLOR | GPU_MAT_SIDE_R,
    GPU_MAT_SRC_L_RECT_MONO     = GPU_MAT_SRC_RECT_MONO | GPU_MAT_SIDE_L,
    GPU_MAT_SRC_R_RECT_MONO     = GPU_MAT_SRC_RECT_MONO | GPU_MAT_SIDE_R,
    GPU_MAT_SRC_L_RECT_COLOR    = GPU_MAT_SRC_RECT_COLOR | GPU_MAT_SIDE_L,
    GPU_MAT_SRC_R_RECT_COLOR    = GPU_MAT_SRC_RECT_COLOR | GPU_MAT_SIDE_R,
    GPU_MAT_SRC_L_DISPARITY     = GPU_MAT_SRC_DISPARITY | GPU_MAT_SIDE_L,
    GPU_MAT_SRC_R_DISPARITY     = GPU_MAT_SRC_DISPARITY | GPU_MAT_SIDE_R,
    GPU_MAT_SRC_L_DISPARITY_32F = GPU_MAT_SRC_DISPARITY_32F | GPU_MAT_SIDE_L,
    GPU_MAT_SRC_R_DISPARITY_32F = GPU_MAT_SRC_DISPARITY_32F | GPU_MAT_SIDE_R
};

class GpuStereoProcessor
{

  public:
    GpuStereoProcessor();

    void initStereoModel(const sensor_msgs::CameraInfoConstPtr &l_info_msg, const sensor_msgs::CameraInfoConstPtr &r_info_msg);
    void initStereoModel(const std::string &left_cal_file, const std::string &right_cal_file);
    bool isStereoModelInitialised();
    void uploadMat(GpuMatSource mat_source, const cv::Mat &cv_mat);
    void downloadMat(GpuMatSource mat_source, const cv::Mat &cv_mat);
    GPUSender::Ptr enqueueSendImage(GpuMatSource source, const sensor_msgs::ImageConstPtr &imagePattern, std::string encoding, ros::Publisher *pub);
    GPUSender::Ptr enqueueSendDisparity(GpuMatSource source, const sensor_msgs::ImageConstPtr &imagePattern, ros::Publisher *pub);
    void colorConvertImage(GpuMatSource source, GpuMatSource dest, int colorConversion, int dcn);
    void rectifyImage(GpuMatSource source, GpuMatSource dest, cv::InterpolationFlags interpolation);
    void rectifyImageLeft(const cv::Mat &source, cv::Mat &dest, cv::InterpolationFlags interpolation);
    void rectifyImageRight(const cv::Mat &source, cv::Mat &dest, cv::InterpolationFlags interpolation);
    void computeDisparity(GpuMatSource left, GpuMatSource right, GpuMatSource disparity);
    void computeDisparity(cv::Mat& left, cv::Mat& right, cv::Mat& disparity);
    void waitForStream(GpuMatSource stream_source);
    void waitForAllStreams();
    void cleanSenders();
    void setPreFilterType(int filter_type);
    void setRefineDisparity(bool ref_disp);
    void setBlockSize(int block_size);
    void setNumDisparities(int numDisp);
    void setTextureThreshold(int threshold);

  protected:
    boost::shared_ptr<cv::cuda::GpuMat> getGpuMat(GpuMatSource source);
    cv::cuda::Stream &getStream(GpuMatSource source);

    cv::cuda::Stream l_strm, r_strm;
    std::vector<GPUSender::Ptr> senders;
    std::unordered_map<std::string, boost::shared_ptr<cv::cuda::GpuMat> > gpu_mats;

    cv::cuda::HostMem filter_buf_;

    image_geometry::StereoCameraModel model_;

    cv::Ptr<cv::cuda::StereoBM> block_matcher_gpu_;
    cv::Ptr<cv::StereoBM> block_matcher_cpu_;
    cv::Ptr<cv::cuda::DisparityBilateralFilter> bilateral_filter_;
    std::string l_cam_name_, r_cam_name;

    bool bilateral_filter_enabled_;
    int maxSpeckleSize_;
    double maxDiff_;
};
}
