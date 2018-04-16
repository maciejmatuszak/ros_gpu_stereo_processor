#include "gpuimageproc/GpuSenderImage.h"

namespace gpuimageproc
{

GPUSenderImage::GPUSenderImage(const std_msgs::Header *header, const ros::Publisher *pub, cv::cuda::HostMem *imageHMem, std::string encoding)
    : GPUSenderIfc(header, pub)
    , imageHMem_(imageHMem)
{
    image_msg_           = boost::make_shared<sensor_msgs::Image>();
    image_msg_->header   = *header;
    image_msg_->encoding = encoding;
}

void GPUSenderImage::fillInData()
{

    image_msg_->height = imageHMem_->rows;
    image_msg_->width  = imageHMem_->cols;
    image_msg_->step   = imageHMem_->step;
    image_msg_->data.resize(image_msg_->height * image_msg_->step);
    cv::Mat image_msg_Mat(image_msg_->height, image_msg_->width, CV_MAKETYPE(CV_8U, channels), &image_msg_->data[0], image_msg_->step);
    imageHMem_->createMatHeader().copyTo(image_msg_Mat);
}

void GPUSenderImage::publish() { publisher_->publish(image_msg_); }

} // namespace
