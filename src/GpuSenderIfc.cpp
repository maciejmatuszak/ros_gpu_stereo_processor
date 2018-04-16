#include "gpuimageproc/GpuSenderIfc.h"

namespace gpuimageproc
{

GPUSenderIfc::GPUSenderIfc(const std_msgs::Header *header, const ros::Publisher *pub)
    : data_sent_(false)
    , header_(header)
    , publisher_(pub)
{
}

void GPUSenderIfc::enqueueSend(cv::cuda::Stream &strm)
{

    strm.enqueueHostCallback(
        [](int status, void *userData) {
            (void)status;
            GPUSenderIfc *sender = static_cast<GPUSenderIfc *>(userData);
            sender->fillInData();
            sender->publish();
            sender->data_sent_ = true;

        },
        (void *)this);
}

bool GPUSenderIfc::wasDataSent() { return data_sent_; }

} // namespace
