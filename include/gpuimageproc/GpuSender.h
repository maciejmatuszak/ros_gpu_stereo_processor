#pragma once
#include <image_geometry/stereo_camera_model.h>
#include <opencv2/cudawarping.hpp>
#include <ros/ros.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <sensor_msgs/image_encodings.h>
#include <stereo_msgs/DisparityImage.h>

namespace gpuimageproc
{

class GPUSender
{
  public:
    GPUSender(const std_msgs::Header& header, ros::Publisher *pub)
        : publisher_(pub)
        , data_sent(false)
        , header_(header)
    {
        points2_msg_ = boost::make_shared<sensor_msgs::PointCloud2>();
    }

    GPUSender(sensor_msgs::ImageConstPtr example, std::string encoding, ros::Publisher *pub)
        : publisher_(pub)
        , data_sent(false)
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
        , data_sent(false)
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

    ~GPUSender()
    {
        cv::cuda::unregisterPageLocked(cpu_data_);
        if (!cpu_color_data_.empty())
        {
            cv::cuda::unregisterPageLocked(cpu_color_data_);
        }
    }

    inline bool isValidPoint(const cv::Vec3f &pt)
    {
        // Check both for disparities explicitly marked as invalid (where OpenCV maps pt.z to MISSING_Z)
        // and zero disparities (point mapped to infinity).
        return pt[2] != image_geometry::StereoCameraModel::MISSING_Z && !std::isinf(pt[2]);
    }

    void fillInPointMessage()
    {
        // fill in points
        const cv::Mat_<cv::Vec3f> xyz(cpu_data_.rows, cpu_data_.cols, (cv::Vec3f *)&cpu_data_.data[0], cpu_data_.step);
        const cv::Mat_<cv::Vec3b> color(cpu_color_data_.rows, cpu_color_data_.cols, (cv::Vec3b *)&cpu_color_data_.data[0], cpu_color_data_.step);

        // Fill in new PointCloud2 message (2D image-like layout)

        points2_msg_->header        = header_;
        points2_msg_->height        = xyz.rows;
        points2_msg_->width         = xyz.cols;
        points2_msg_->is_bigendian  = false;
        points2_msg_->is_dense      = false; // there may be invalid points

        sensor_msgs::PointCloud2Modifier pcd_modifier(*points2_msg_);
        pcd_modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");

        sensor_msgs::PointCloud2Iterator<float> iter_x(*points2_msg_, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(*points2_msg_, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(*points2_msg_, "z");
        sensor_msgs::PointCloud2Iterator<uint8_t> iter_r(*points2_msg_, "r");
        sensor_msgs::PointCloud2Iterator<uint8_t> iter_g(*points2_msg_, "g");
        sensor_msgs::PointCloud2Iterator<uint8_t> iter_b(*points2_msg_, "b");

        float bad_point = std::numeric_limits<float>::quiet_NaN();
        //Fill in xyz
        for (int v = 0; v < xyz.rows; ++v)
        {
            for (int u = 0; u < xyz.cols; ++u, ++iter_x, ++iter_y, ++iter_z)
            {
                if (isValidPoint(xyz(v, u)))
                {
                    // x,y,z
                    *iter_x = xyz(v, u)[0];
                    *iter_y = xyz(v, u)[1];
                    *iter_z = xyz(v, u)[2];
                }
                else
                {
                    *iter_x = *iter_y = *iter_z = bad_point;
                }
            }
        }

        //Fill in color
        for (int v = 0; v < xyz.rows; ++v)
        {
            for (int u = 0; u < xyz.cols; ++u, ++iter_r, ++iter_g, ++iter_b)
            {
                const cv::Vec3b &bgr = color(v, u);
                *iter_r              = bgr[2];
                *iter_g              = bgr[1];
                *iter_b              = bgr[0];
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
        if (points2_msg_)
        {
            fillInPointMessage();
            if (publisher_)
            {
                publisher_->publish(points2_msg_);
            }
        }

        data_sent = true;
    }

    void enqueueSend(cv::cuda::GpuMat &m, cv::cuda::GpuMat &col, cv::cuda::Stream &strm)
    {
        cpu_data_.create(m.size(),m.type());
        cv::cuda::registerPageLocked(cpu_data_);
        cpu_color_data_.create(col.size(),col.type());
        cv::cuda::registerPageLocked(cpu_color_data_);

        m.download(cpu_data_, strm);
        col.download(cpu_color_data_, strm);

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
    sensor_msgs::PointCloud2Ptr &getPointsMessage() { return points2_msg_; }

    bool wasDataSent() { return data_sent; }
    typedef boost::shared_ptr<GPUSender> Ptr;

  private:
    static const int STEP = 16;
    bool data_sent;
    sensor_msgs::ImagePtr image_msg_;
    stereo_msgs::DisparityImagePtr disp_msg_;
    sensor_msgs::PointCloud2Ptr points2_msg_;
    std_msgs::Header header_;
    cv::Mat cpu_data_;
    cv::Mat cpu_color_data_;
    ros::Publisher *publisher_;
};
} // namespace
