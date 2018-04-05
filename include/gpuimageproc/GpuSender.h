#pragma once
#include <opencv2/cudawarping.hpp>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>
#include <stereo_msgs/DisparityImage.h>
#include <image_geometry/stereo_camera_model.h>

namespace gpuimageproc
{

class GPUSender
{
  public:
    GPUSender(sensor_msgs::ImageConstPtr example, ros::Publisher *pub)
        : publisher_(pub)
    {
        points_msg_         = boost::make_shared<sensor_msgs::PointCloud2>();
        points_msg_->header = example->header;
        points_msg_->height = example->height;
        points_msg_->width  = example->width;
        points_msg_->fields.resize(4);
        points_msg_->fields[0].name     = "x";
        points_msg_->fields[0].offset   = 0;
        points_msg_->fields[0].count    = 1;
        points_msg_->fields[0].datatype = sensor_msgs::PointField::FLOAT32;
        points_msg_->fields[1].name     = "y";
        points_msg_->fields[1].offset   = 4;
        points_msg_->fields[1].count    = 1;
        points_msg_->fields[1].datatype = sensor_msgs::PointField::FLOAT32;
        points_msg_->fields[2].name     = "z";
        points_msg_->fields[2].offset   = 8;
        points_msg_->fields[2].count    = 1;
        points_msg_->fields[2].datatype = sensor_msgs::PointField::FLOAT32;
        points_msg_->fields[3].name     = "rgb";
        points_msg_->fields[3].offset   = 12;
        points_msg_->fields[3].count    = 1;
        points_msg_->fields[3].datatype = sensor_msgs::PointField::FLOAT32;
        // points_msg_->is_bigendian = false; ???

        points_msg_->point_step = STEP;
        points_msg_->row_step   = points_msg_->point_step * points_msg_->width;
        points_msg_->data.resize(points_msg_->row_step * points_msg_->height);
        points_msg_->is_dense = false; // there may be invalid points
        cpu_data_             = cv::Mat(image_msg_->height, image_msg_->width, CV_32FC3, &(points_msg_->data[0]), points_msg_->row_step);
        cpu_color_data_.create(image_msg_->height, image_msg_->width, CV_32FC3);

        cv::cuda::registerPageLocked(cpu_data_);
    }

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

    inline bool isValidPoint(const cv::Vec3f& pt)
    {
      // Check both for disparities explicitly marked as invalid (where OpenCV maps pt.z to MISSING_Z)
      // and zero disparities (point mapped to infinity).
      return pt[2] != image_geometry::StereoCameraModel::MISSING_Z && !std::isinf(pt[2]);
    }

    void fillInPointMessage()
    {
        // fill in points
        const cv::Mat_<cv::Vec3b> cpu_xyz(cpu_data_);
        float bad_point = std::numeric_limits<float>::quiet_NaN();
        int offset      = 0;
        for (int v = 0; v < cpu_xyz.rows; ++v)
        {
            for (int u = 0; u < cpu_xyz.cols; ++u, offset += STEP)
            {
                if (isValidPoint(cpu_xyz(v, u)))
                {
                    // x,y,z,rgba
                    memcpy(&points_msg_->data[offset + 0], &cpu_xyz(v, u)[0], sizeof(float));
                    memcpy(&points_msg_->data[offset + 4], &cpu_xyz(v, u)[1], sizeof(float));
                    memcpy(&points_msg_->data[offset + 8], &cpu_xyz(v, u)[2], sizeof(float));
                }
                else
                {
                    memcpy(&points_msg_->data[offset + 0], &bad_point, sizeof(float));
                    memcpy(&points_msg_->data[offset + 4], &bad_point, sizeof(float));
                    memcpy(&points_msg_->data[offset + 8], &bad_point, sizeof(float));
                }
            }
        }

        // Fill in color
        offset = 0;
        const cv::Mat_<cv::Vec3b> color(cpu_color_data_);
        for (int v = 0; v < cpu_xyz.rows; ++v)
        {
            for (int u = 0; u < cpu_xyz.cols; ++u, offset += STEP)
            {
                if (isValidPoint(cpu_xyz(v,u)))
                {
                    const cv::Vec3b& bgr = color(v,u);
                    int32_t rgb_packed = (bgr[2] << 16) | (bgr[1] << 8) | bgr[0];
                    memcpy (&points_msg_->data[offset + 12], &rgb_packed, sizeof (int32_t));
                }
                else
                {
                    memcpy (&points_msg_->data[offset + 12], &bad_point, sizeof (float));
                }
            }
        }
    }

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
        if (points_msg_ && publisher_)
        {
            fillInPointMessage();
            publisher_->publish(points_msg_);
        }
    }

    void enqueueSend(cv::cuda::GpuMat &m, cv::cuda::GpuMat &col, cv::cuda::Stream &strm)
    {
        // cpu_data_ should be constructed in such a way that it does not require resize
        auto vptr = (void *)&cpu_data_.data[0];
        m.download(cpu_data_, strm);
        assert(vptr == (void *)&cpu_data_.data[0]);

        vptr = (void *)&cpu_color_data_.data[0];
        col.download(cpu_color_data_, strm);
        assert(vptr == (void *)&cpu_color_data_.data[0]);

        strm.enqueueHostCallback(
            [](int status, void *userData) {
                (void)status;
                static_cast<GPUSender *>(userData)->send();
            },
            (void *)this);
    }

    void enqueueSend(cv::cuda::GpuMat &m, cv::cuda::Stream &strm)
    {
        auto vptr = (void *)&cpu_data_.data[0];
        m.download(cpu_data_, strm);
        // cpu_data_ should be constructed in such a way that it does not require resize
        assert(vptr == (void *)&cpu_data_.data[0]);
        strm.enqueueHostCallback(
            [](int status, void *userData) {
                (void)status;
                static_cast<GPUSender *>(userData)->send();
            },
            (void *)this);
    }

    cv::Mat &getCpuData() { return cpu_data_; }

    typedef boost::shared_ptr<GPUSender> Ptr;

  private:
    static const int STEP   = 16;
    sensor_msgs::ImagePtr image_msg_;
    stereo_msgs::DisparityImagePtr disp_msg_;
    sensor_msgs::PointCloud2Ptr points_msg_;
    cv::Mat cpu_data_;
    cv::Mat cpu_color_data_;
    ros::Publisher *publisher_;
};
} // namespace
