#include "gpuimageproc/GpuSenderDisparity.h"

namespace gpuimageproc
{

GPUSenderDisparity::GPUSenderDisparity(const std_msgs::Header *header, const ros::Publisher *pub, cv::cuda::HostMem *imageHMem, int blockSize, int numDisparities, int minDisparity,
                                       double fx, double baseline)
    : GPUSenderIfc(header, pub)
    , disparityHMem_(imageHMem)
    , blockSize_(blockSize)
    , numDisparities_(numDisparities)
    , minDisparity_(minDisparity)
    , fx_(fx)
    , baseline_(baseline)
{
}

void GPUSenderDisparity::fillInData()
{

    disparity_msg_         = boost::make_shared<stereo_msgs::DisparityImage>();
    disparity_msg_->header = disparity_msg_->image.header = *header_;
    disparity_msg_->image.encoding        = sensor_msgs::image_encodings::TYPE_32FC1;

    disparity_msg_->image.height          = disparityHMem_->rows;
    disparity_msg_->image.width           = disparityHMem_->cols;
    disparity_msg_->image.step            = disparityHMem_->step;
    // Compute window of (potentially) valid disparities
    int border                        = blockSize_ / 2;
    int left                          = numDisparities_ + minDisparity_ + border - 1;
    int wtf                           = (minDisparity_ >= 0) ? border + minDisparity_ : std::max(border, -minDisparity_);
    int right                         = disparity_msg_->image.width - 1 - wtf;
    int top                           = border;
    int bottom                        = disparity_msg_->image.height - 1 - border;
    disparity_msg_->valid_window.x_offset = left;
    disparity_msg_->valid_window.y_offset = top;
    disparity_msg_->valid_window.width    = right - left;
    disparity_msg_->valid_window.height   = bottom - top;
    disparity_msg_->min_disparity         = minDisparity_ + 1;
    disparity_msg_->max_disparity         = minDisparity_ + numDisparities_ - 1;
    disparity_msg_->f                     = fx_;
    disparity_msg_->T                     = baseline_;
    disparity_msg_->image.data.resize(disparity_msg_->image.height * disparity_msg_->image.step);

    cv::Mat disparity_msg_Mat(disparityHMem_->rows, disparityHMem_->cols, disparityHMem_->type(), &disparity_msg_->image.data[0], disparityHMem_->step);
    disparityHMem_->createMatHeader().copyTo(disparity_msg_Mat);
}

void GPUSenderDisparity::publish() { publisher_->publish(disparity_msg_); }

} // namespace
