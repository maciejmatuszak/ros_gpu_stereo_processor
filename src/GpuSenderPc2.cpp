#include "gpuimageproc/GpuSenderPc2.h"
#include <sensor_msgs/point_cloud2_iterator.h>
#include <image_geometry/stereo_camera_model.h>

namespace gpuimageproc
{

GPUSenderPc2::GPUSenderPc2(const std_msgs::Header *header, const ros::Publisher *pub, cv::cuda::HostMem *pc2HMem, cv::cuda::HostMem *colorHMem)
    : GPUSenderIfc(header, pub)
    , pc2HMem_(pc2HMem)
    , colorHMem_(colorHMem)
{
}

void GPUSenderPc2::fillInData()
{
    points2_msg_         = boost::make_shared<sensor_msgs::PointCloud2>();

    // fill in points
    const cv::Mat_<cv::Vec3f> xyz(pc2HMem_->rows, pc2HMem_->cols, (cv::Vec3f *)&pc2HMem_->data[0], pc2HMem_->step);
    const cv::Mat_<cv::Vec3b> color(colorHMem_->rows, colorHMem_->cols, (cv::Vec3b *)&colorHMem_->data[0], colorHMem_->step);

    // Fill in new PointCloud2 message (2D image-like layout)

    points2_msg_->header       = *header_;
    points2_msg_->height       = xyz.rows;
    points2_msg_->width        = xyz.cols;
    points2_msg_->is_bigendian = false;
    points2_msg_->is_dense     = false; // there may be invalid points

    sensor_msgs::PointCloud2Modifier pcd_modifier(*points2_msg_);
    pcd_modifier.setPointCloud2FieldsByString(2, "xyz", "rgb");

    sensor_msgs::PointCloud2Iterator<float> iter_x(*points2_msg_, "x");
    sensor_msgs::PointCloud2Iterator<float> iter_y(*points2_msg_, "y");
    sensor_msgs::PointCloud2Iterator<float> iter_z(*points2_msg_, "z");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_r(*points2_msg_, "r");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_g(*points2_msg_, "g");
    sensor_msgs::PointCloud2Iterator<uint8_t> iter_b(*points2_msg_, "b");

    float bad_point = std::numeric_limits<float>::quiet_NaN();
    // Fill in xyz
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

    // Fill in color
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

void GPUSenderPc2::publish() { publisher_->publish(points2_msg_); }


inline bool GPUSenderPc2::isValidPoint(const cv::Vec3f &pt)
{
    // Check both for disparities explicitly marked as invalid (where OpenCV maps pt.z to MISSING_Z)
    // and zero disparities (point mapped to infinity).
    return pt[2] != image_geometry::StereoCameraModel::MISSING_Z && !std::isinf(pt[2]);
}

} // namespace
