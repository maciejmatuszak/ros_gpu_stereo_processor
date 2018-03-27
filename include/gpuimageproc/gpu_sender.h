#pragma once
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <opencv2/cudawarping.hpp>
#include <ros/ros.h>

namespace gpuimageproc {


class GPUSender
{
    public:
        GPUSender(sensor_msgs::ImageConstPtr example, std::string encoding, ros::Publisher& pub) :
            image_msg_(new sensor_msgs::Image()),
            publisher_(pub)
        {
            int bitdepth, channels;
            bitdepth = sensor_msgs::image_encodings::bitDepth(encoding);
            channels = sensor_msgs::image_encodings::numChannels(encoding);
            image_msg_->header = example->header;
            image_msg_->height = example->height;
            image_msg_->width = example->width;
            image_msg_->encoding = encoding;
            image_msg_->step = example->width*bitdepth*channels/8;
            image_msg_->data.resize( image_msg_->height * image_msg_->step );
            image_data_ = cv::Mat(image_msg_->height, image_msg_->width,
                    CV_MAKETYPE(CV_8U, channels),
                    &image_msg_->data[0], image_msg_->step);
            cv::cuda::registerPageLocked(image_data_);
        }
        ~GPUSender()
        {
            cv::cuda::unregisterPageLocked(image_data_);
        }
        void send(void)
        {
            publisher_.publish( image_msg_ );
        }
        void enqueueSend(cv::cuda::GpuMat& m, cv::cuda::Stream& strm)
        {
            m.download(image_data_, strm);
            strm.enqueueHostCallback(
                [](int status, void *userData)
                {
                    (void)status;
                   static_cast<GPUSender *>(userData)->send();
                },
                (void *)this);
        }
        typedef boost::shared_ptr<GPUSender> Ptr;
    private:
        sensor_msgs::ImagePtr image_msg_;
        cv::Mat image_data_;
        ros::Publisher& publisher_;
};
} // namespace
