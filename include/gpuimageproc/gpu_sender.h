#pragma once
#include <opencv2/cudawarping.hpp>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/image_encodings.h>
#include <stereo_msgs/DisparityImage.h>

namespace gpuimageproc
{

class GPUSender
{
  public:
    GPUSender(sensor_msgs::ImageConstPtr example, std::string encoding, ros::Publisher *pub)
        : publisher_(pub)
    {
        image_msg_ = boost::make_shared<sensor_msgs::Image>();
        int bitdepth, channels;
        bitdepth             = sensor_msgs::image_encodings::bitDepth(encoding);
        channels             = sensor_msgs::image_encodings::numChannels(encoding);
        image_msg_->header   = example->header;
        image_msg_->height   = example->height;
        image_msg_->width    = example->width;
        image_msg_->encoding = encoding;
        image_msg_->step     = example->width * bitdepth * channels / 8;
        image_msg_->data.resize(image_msg_->height * image_msg_->step);
        cpu_data_ = cv::Mat(image_msg_->height, image_msg_->width, CV_MAKETYPE(CV_8U, channels), &image_msg_->data[0], image_msg_->step);
        cv::cuda::registerPageLocked(cpu_data_);
    }

    GPUSender(sensor_msgs::ImageConstPtr example, int blockSize, int numDisparities, int minDisparity, ros::Publisher *pub)
        : publisher_(pub)
    {
        disp_msg_               = boost::make_shared<stereo_msgs::DisparityImage>();
        disp_msg_->header       = example->header;
        disp_msg_->image.header = example->header;

        // Compute window of (potentially) valid disparities
        int border                       = blockSize / 2;
        int left                         = numDisparities + minDisparity + border - 1;
        int wtf                          = (minDisparity >= 0) ? border + minDisparity : std::max(border, -minDisparity);
        int right                        = disp_msg_->image.width - 1 - wtf;
        int top                          = border;
        int bottom                       = disp_msg_->image.height - 1 - border;
        disp_msg_->valid_window.x_offset = left;
        disp_msg_->valid_window.y_offset = top;
        disp_msg_->valid_window.width    = right - left;
        disp_msg_->valid_window.height   = bottom - top;
        disp_msg_->min_disparity         = minDisparity + 1;
        disp_msg_->max_disparity         = minDisparity + numDisparities - 1;

        disp_msg_->image.height   = example->height;
        disp_msg_->image.width    = example->width;
        disp_msg_->image.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
        disp_msg_->image.step     = example->width * sizeof(float);
        disp_msg_->image.data.resize(disp_msg_->image.step * disp_msg_->image.height);
        cpu_data_ = cv::Mat(disp_msg_->image.height, disp_msg_->image.width, CV_32FC1, (void *)&disp_msg_->image.data[0], disp_msg_->image.step);
        cv::cuda::registerPageLocked(cpu_data_);
    }

    ~GPUSender() { cv::cuda::unregisterPageLocked(cpu_data_); }

    void send(void)
    {
        if (image_msg_ && publisher_)
        {
            publisher_->publish(image_msg_);
        }
        if (disp_msg_ && publisher_)
        {
            publisher_->publish(disp_msg_);
        }
    }

    void enqueueSend(cv::cuda::GpuMat &m, cv::cuda::Stream &strm)
    {
        auto vptr = (void*)&cpu_data_.data[0];
        m.download(cpu_data_, strm);
        //cpu_data_ should be constructed in such a way that it does not require resize
        assert(vptr == (void*)&cpu_data_.data[0]);
        strm.enqueueHostCallback(
            [](int status, void *userData) {
                (void)status;
                static_cast<GPUSender *>(userData)->send();
            },
            (void *)this);
    }
    cv::Mat& getCpuData()
    {
        return cpu_data_;
    }

    typedef boost::shared_ptr<GPUSender> Ptr;

  private:
    sensor_msgs::ImagePtr image_msg_;
    stereo_msgs::DisparityImagePtr disp_msg_;
    cv::Mat cpu_data_;
    ros::Publisher *publisher_;
};
} // namespace
