#pragma once
#include <boost/thread/lock_guard.hpp>

#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>

#include <dynamic_reconfigure/server.h>
#include <image_geometry/stereo_camera_model.h>
#include <image_transport/subscriber_filter.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/sync_policies/exact_time.h>
#include <message_filters/synchronizer.h>
#include <opencv2/cudastereo.hpp>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <stereo_msgs/DisparityImage.h>

#include "gpuimageproc/ConnectedTopics.h"
#include "gpuimageproc/GPUConfig.h"
#include "gpuimageproc/GPUStereoProcessor.h"

namespace gpuimageproc
{
class StereoProcessor
{

  public:
    const static std::string CAMERA_TOPIC_LEFT;
    const static std::string CAMERA_TOPIC_RIGHT;
    const static std::string CAMERA_TOPIC_IMAGE;
    const static std::string CAMERA_TOPIC_INFO;

    StereoProcessor(ros::NodeHandle &nh, ros::NodeHandle &private_nh);

  protected:
    ros::NodeHandle &nh;
    ros::NodeHandle &private_nh;
    boost::shared_ptr<image_transport::ImageTransport> it_;

    // Subscriptions
    image_transport::SubscriberFilter sub_l_raw_image_, sub_r_raw_image_;
    message_filters::Subscriber<sensor_msgs::CameraInfo> sub_l_info_, sub_r_info_;

    typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::Image> ExactPolicyImages;
    typedef message_filters::sync_policies::ExactTime<sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::Image, sensor_msgs::CameraInfo> ExactPolicyImagesAndInfo;

    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::Image> ApproximatePolicyImages;
    typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::CameraInfo, sensor_msgs::Image, sensor_msgs::CameraInfo>
        ApproximatePolicyImagesAndInfo;

    typedef message_filters::Synchronizer<ExactPolicyImages> ExactSyncImages;
    typedef message_filters::Synchronizer<ExactPolicyImagesAndInfo> ExactSyncImagesAndInfo;

    typedef message_filters::Synchronizer<ApproximatePolicyImages> ApproximateSyncImages;
    typedef message_filters::Synchronizer<ApproximatePolicyImagesAndInfo> ApproximateSyncImagesAndInfo;

    boost::shared_ptr<ExactSyncImages> exact_sync_images_;
    boost::shared_ptr<ExactSyncImagesAndInfo> exact_sync_images_and_info_;

    boost::shared_ptr<ApproximateSyncImages> approximate_sync_images_;
    boost::shared_ptr<ApproximateSyncImagesAndInfo> approximate_sync_images_and_info_;

    image_transport::ImageTransport it;
    // Publications
    boost::mutex connect_mutex_;
    ros::Publisher pub_mono_left_;
    ros::Publisher pub_mono_right_;
    ros::Publisher pub_color_left_;
    ros::Publisher pub_color_right_;
    image_transport::CameraPublisher pub_mono_rect_left_;
    ros::Publisher pub_mono_rect_right_;
    ros::Publisher pub_color_rect_left_;
    ros::Publisher pub_color_rect_right_;
    ros::Publisher pub_disparity_;
    ros::Publisher pub_disparity_vis_;
    ros::Publisher pub_pointcloud_;
    struct ConnectedTopics connected_;
    std::string camera_info_file_left_, camera_info_file_right_;
    bool camera_info_from_files_;

    stereo_msgs::DisparityImagePtr disp_msg_;
    cv::Mat_<float> disp_msg_data_;
    cv::cuda::HostMem filter_buf_;
    boost::shared_ptr<GpuStereoProcessor> stereoProcessor_;

    // Dynamic reconfigure
    boost::recursive_mutex config_mutex_;
    typedef gpuimageproc::GPUConfig Config;
    typedef dynamic_reconfigure::Server<Config> ReconfigureServer;
    boost::shared_ptr<ReconfigureServer> reconfigure_server_;

    // Processing state (note: only safe because we're single-threaded!)
    image_geometry::StereoCameraModel model_;

    cv::Ptr<cv::cuda::DisparityBilateralFilter> bilateral_filter_;
    const int block_matcher_min_disparity_ = 0;

    cv::Ptr<cv::cuda::StereoConstantSpaceBP> csbp_;

    void connectCb();

    void imageAndInfoCb(const sensor_msgs::ImageConstPtr &l_raw_msg, const sensor_msgs::CameraInfoConstPtr &l_info_msg, const sensor_msgs::ImageConstPtr &r_raw_msg,
                        const sensor_msgs::CameraInfoConstPtr &r_info_msg);

    void imageCb(const sensor_msgs::ImageConstPtr &l_raw_msg, const sensor_msgs::ImageConstPtr &r_raw_msg);

    void configCb(Config &config, uint32_t level);

    bool bilateral_filter_enabled_;
};
}
