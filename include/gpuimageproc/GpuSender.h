#pragma once
#include "gpuimageproc/GPUStereoProcessor.h"
#include <image_geometry/stereo_camera_model.h>
#include <opencv2/cudawarping.hpp>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <stereo_msgs/DisparityImage.h>

namespace gpuimageproc
{

class GpuStereoProcessor;

class GPUSender
{
  public:
    // Create for points2
    GPUSender(GpuStereoProcessor *gpuProcessor, const std_msgs::Header &header, ros::Publisher *pub);

    // Create for image
    GPUSender(GpuStereoProcessor *gpuProcessor, sensor_msgs::ImageConstPtr example, std::string encoding, ros::Publisher *pub);

    // create for Disparity message
    GPUSender(GpuStereoProcessor *gpuProcessor, sensor_msgs::ImageConstPtr example, int blockSize, int numDisparities, int minDisparity, double fx, double baseline,
              ros::Publisher *pub);

    ~GPUSender();

    inline bool isValidPoint(const cv::Vec3f &pt);

    void fillInPointMessage();

    void send(void);

    void enqueueSend(cv::cuda::GpuMat &m, cv::cuda::GpuMat &col, cv::cuda::Stream &strm);

    void enqueueSend(cv::cuda::GpuMat &m, cv::cuda::Stream &strm);

    cv::Mat &getCpuData();
    sensor_msgs::PointCloud2Ptr &getPointsMessage();

    bool wasDataSent();

  private:
    static const int STEP = 16;
    bool data_sent;
    sensor_msgs::ImagePtr image_msg_;
    stereo_msgs::DisparityImagePtr disp_msg_;
    sensor_msgs::PointCloud2Ptr points2_msg_;
    std_msgs::Header header_;
    cv::Mat cpu_data_;
    cv::Mat cpu_color_data_;
    ros::Publisher *publisher_;
    gpuimageproc::GpuStereoProcessor *gpuProcessor_;
};

typedef boost::shared_ptr<GPUSender> GPUSenderPtr;

} // namespace
