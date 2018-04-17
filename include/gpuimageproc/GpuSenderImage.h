#pragma once
#include "gpuimageproc/GpuSenderIfc.h"
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>

namespace gpuimageproc
{
class GPUSenderImage : public GPUSenderIfc
{
  public:
    GPUSenderImage(const std_msgs::Header *header, const ros::Publisher *pub, boost::shared_ptr<cv::cuda::HostMem> imageHMem, std::string encoding);

    // GPUSenderIfc interface
  public:
    void fillInData() override;
    void publish() override;

  private:
    boost::shared_ptr<cv::cuda::HostMem> imageHMem_;
    boost::shared_ptr<sensor_msgs::Image> image_msg_;
};

typedef boost::shared_ptr<GPUSenderImage> GPUSenderImagePtr;

} // namespace
