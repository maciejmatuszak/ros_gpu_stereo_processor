#pragma once
#include "gpuimageproc/GpuSenderIfc.h"
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>

namespace gpuimageproc
{
class GPUSenderPc2 : public GPUSenderIfc
{
  public:
    GPUSenderPc2(const std_msgs::Header *header, const ros::Publisher *pub, cv::cuda::HostMem *imageHMem, cv::cuda::HostMem *colorHMem);

    // GPUSenderIfc interface
public:
    void fillInData() override;
    void publish() override;
private:
    bool isValidPoint(const cv::Vec3f &pt);
    boost::shared_ptr<cv::cuda::HostMem> pc2HMem_;
    boost::shared_ptr<cv::cuda::HostMem> colorHMem_;
    boost::shared_ptr<sensor_msgs::PointCloud2> points2_msg_;
};

typedef boost::shared_ptr<GPUSenderPc2> GPUSenderPc2Ptr;

} // namespace
