#pragma once
#include <opencv2/cudawarping.hpp>
#include <ros/ros.h>

namespace gpuimageproc
{

class GPUSenderIfc
{
  public:
    GPUSenderIfc(const std_msgs::Header *header, const ros::Publisher *pub);
    virtual void fillInData() = 0;
    virtual void publish()    = 0;

    void enqueueSend(cv::cuda::Stream &strm);

    bool wasDataSent();

  protected:
    bool data_sent_;
    const ros::Publisher *publisher_;
    const std_msgs::Header *header_;
};

typedef boost::shared_ptr<GPUSenderIfc> GPUSenderIfcPtr;

} // namespace
