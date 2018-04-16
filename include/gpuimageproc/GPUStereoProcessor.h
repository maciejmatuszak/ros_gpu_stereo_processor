#pragma once
#include "gpuimageproc/GpuSenderIfc.h"
#include <ros/ros.h>
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
    GPU_MAT_SRC_DISPARITY_IMG = 1 << 9,
    GPU_MAT_SRC_POINTS2       = 1 << 10,
    GPU_MAT_SRC_MASK = GPU_MAT_SRC_RAW | GPU_MAT_SRC_MONO | GPU_MAT_SRC_COLOR | GPU_MAT_SRC_RECT_MONO | GPU_MAT_SRC_RECT_COLOR | GPU_MAT_SRC_DISPARITY | GPU_MAT_SRC_DISPARITY_32F |
                       GPU_MAT_SRC_DISPARITY_IMG | GPU_MAT_SRC_POINTS2,

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
    GPU_MAT_SRC_R_DISPARITY_32F = GPU_MAT_SRC_DISPARITY_32F | GPU_MAT_SIDE_R,
    GPU_MAT_SRC_L_DISPARITY_IMG = GPU_MAT_SRC_DISPARITY_IMG | GPU_MAT_SIDE_L,
    GPU_MAT_SRC_R_DISPARITY_IMG = GPU_MAT_SRC_DISPARITY_IMG | GPU_MAT_SIDE_R,
    GPU_MAT_SRC_L_POINTS2       = GPU_MAT_SRC_POINTS2 | GPU_MAT_SIDE_L,
    GPU_MAT_SRC_R_POINTS2       = GPU_MAT_SRC_POINTS2 | GPU_MAT_SIDE_R
};

inline GpuMatSource operator&(GpuMatSource a, GpuMatSource b) { return static_cast<GpuMatSource>(static_cast<int>(a) & static_cast<int>(b)); }

inline GpuMatSource operator|(GpuMatSource a, GpuMatSource b) { return static_cast<GpuMatSource>(static_cast<int>(a) | static_cast<int>(b)); }

class GpuStereoProcessor
{
  public:
    GpuStereoProcessor();

    void initStereoModel(const sensor_msgs::CameraInfoConstPtr &l_info_msg, const sensor_msgs::CameraInfoConstPtr &r_info_msg);
    void initStereoModel(const std::string &left_cal_file, const std::string &right_cal_file);
    bool isStereoModelInitialised();
    void convertRawToColor(GpuMatSource side);
    void convertRawToMono(GpuMatSource side);
    void uploadMat(GpuMatSource mat_source, const cv::Mat &cv_mat, std::string encoding = "");
    void downloadMat(GpuMatSource mat_source, const cv::Mat &cv_mat);
    void convertColor(GpuMatSource mat_source, GpuMatSource mat_dst, const std::string &src_encoding, const std::string &dst_encoding);
    GPUSenderIfcPtr enqueueSendImage(GpuMatSource source, const sensor_msgs::ImageConstPtr &imagePattern, std::string encoding, ros::Publisher *pub);
    GPUSenderIfcPtr enqueueSendDisparity(GpuMatSource source, const sensor_msgs::ImageConstPtr &imagePattern, ros::Publisher *pub);
    GPUSenderIfcPtr enqueueSendPoints(GpuMatSource points_source, GpuMatSource color_source, const sensor_msgs::ImageConstPtr &imagePattern, ros::Publisher *pub);
    void rectifyImage(GpuMatSource source, GpuMatSource dest, cv::InterpolationFlags interpolation);
    void rectifyImageLeft(const cv::Mat &source, cv::Mat &dest, cv::InterpolationFlags interpolation);
    void rectifyImageRight(const cv::Mat &source, cv::Mat &dest, cv::InterpolationFlags interpolation);
    void computeDisparity(GpuMatSource left, GpuMatSource right, GpuMatSource disparity);
    void computeDisparityImage(GpuMatSource disparity_src, GpuMatSource disp_image_dest);
    void computeDisparity(cv::Mat &left, cv::Mat &right, cv::Mat &disparity);
    void projectDisparityTo3DPoints(GpuMatSource disparity_src, GpuMatSource points_src);
    void waitForStream(GpuMatSource stream_source);
    void waitForAllStreams();
    void cleanSenders();
    void setPreFilterType(int filter_type);
    void setRefineDisparity(bool ref_disp);
    void setBlockSize(int block_size);
    void setNumDisparities(int numDisp);
    void setMinDisparity(int minDisp);
    void setTextureThreshold(int threshold);
    void printStats(std::string name, cv::Mat &mat);
    void filterSpeckles(cv::Mat &disparity);

    int getMaxSpeckleSize() const;
    void setMaxSpeckleSize(int maxSpeckleSize);

    double getMaxSpeckleDiff() const;
    void setMaxSpeckleDiff(double maxSpeckleDiff);

protected:
    boost::shared_ptr<cv::cuda::HostMem> getHostMem(GpuMatSource source);
    cv::cuda::Stream &getStream(GpuMatSource source);

    cv::cuda::Stream l_strm, r_strm;
    std::vector<GPUSenderIfcPtr> senders;
    std::unordered_map<std::string, boost::shared_ptr<cv::cuda::HostMem> > gpu_mats;

    image_geometry::StereoCameraModel model_;

    cv::Ptr<cv::cuda::StereoBM> block_matcher_gpu_;
    cv::Ptr<cv::StereoBM> block_matcher_cpu_;
    cv::Ptr<cv::cuda::DisparityBilateralFilter> bilateral_filter_;
    std::string l_cam_name_, r_cam_name;
    std::string l_raw_encoding_, r_raw_encoding_;

    bool bilateral_filter_enabled_;
    int maxSpeckleSize_;
    double maxSpeckleDiff_;
    cv::Mat _speclesBuf;
};
}
