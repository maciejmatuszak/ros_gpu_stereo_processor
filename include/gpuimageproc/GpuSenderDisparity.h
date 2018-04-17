#pragma once
#include "gpuimageproc/GpuSenderIfc.h"
#include <ros/ros.h>
#include <sensor_msgs/image_encodings.h>
#include <stereo_msgs/DisparityImage.h>

namespace gpuimageproc
{
class GPUSenderDisparity : public GPUSenderIfc
{
  public:
    GPUSenderDisparity(const std_msgs::Header *header, const ros::Publisher *pub, boost::shared_ptr<cv::cuda::HostMem> imageHMem, int blockSize, int numDisparities,
                       int minDisparity, double fx, double baseline);

    // GPUSenderIfc interface
  public:
    void fillInData() override;
    void publish() override;
    boost::shared_ptr<cv::cuda::HostMem> getDisparityHostMem();

  private:
    boost::shared_ptr<cv::cuda::HostMem> disparityHMem_;
    boost::shared_ptr<stereo_msgs::DisparityImage> disparity_msg_;
    int blockSize_;
    int numDisparities_;
    int minDisparity_;
    double fx_;
    double baseline_;
};

typedef boost::shared_ptr<GPUSenderDisparity> GPUSenderDisparityPtr;

} // namespace
