#include <cv_bridge/cv_bridge.h>
#include <image_transport/image_transport.h>
#include <pluginlib/class_list_macros.h>

#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/image_encodings.h>
#include <stereo_msgs/DisparityImage.h>
#include <boost/timer.hpp>
#include "gpuimageproc/stereoproc.h"

namespace gpuimageproc
{

StereoProcessor::StereoProcessor(ros::NodeHandle &nh, ros::NodeHandle &private_nh)
    : nh(nh)
    , private_nh(private_nh)
{
    stereoProcessor_        = boost::make_shared<GpuStereoProcessor>();
    camera_info_file_left_  = "/data/git/temp_ws/src/gpuimageproc/test/stereobm/test_data/left.yaml";
    camera_info_file_right_ = "/data/git/temp_ws/src/gpuimageproc/test/stereobm/test_data/right.yaml";

    it_.reset(new image_transport::ImageTransport(nh));

    // Synchronize inputs. Topic subscriptions happen on demand in the connection
    // callback. Optionally do approximate synchronization.
    int queue_size;
    private_nh.param("queue_size", queue_size, 5);
    bool approx;
    private_nh.param("approximate_sync", approx, false);

    private_nh.param<std::string>("camera_info_file_left", camera_info_file_left_, camera_info_file_left_);
    private_nh.param<std::string>("camera_info_file_right", camera_info_file_right_, camera_info_file_right_);

    ROS_INFO("PARAM: camera_info_file_left:%s", camera_info_file_left_.c_str());
    ROS_INFO("PARAM: camera_info_file_right:%s", camera_info_file_right_.c_str());
    ROS_INFO("PARAM: queue_size:%d", queue_size);
    ROS_INFO("PARAM: approximate_sync:%s", approx ? "true" : "false");

    if (!camera_info_file_left_.empty() || !camera_info_file_right_.empty())
    {
        camera_info_from_files_ = true;
    }

    if (approx)
    {
        if (camera_info_from_files_)
        {
            approximate_sync_images_.reset(new ApproximateSyncImages(ApproximatePolicyImages(queue_size), sub_l_raw_image_, sub_r_raw_image_));
            approximate_sync_images_->registerCallback(boost::bind(&StereoProcessor::imageCb, this, _1, _2));
        }
        else
        {
            approximate_sync_images_and_info_.reset(
                new ApproximateSyncImagesAndInfo(ApproximatePolicyImagesAndInfo(queue_size), sub_l_raw_image_, sub_l_info_, sub_r_raw_image_, sub_r_info_));
            approximate_sync_images_and_info_->registerCallback(boost::bind(&StereoProcessor::imageAndInfoCb, this, _1, _2, _3, _4));
        }
    }
    else
    {
        if (camera_info_from_files_)
        {
            exact_sync_images_.reset(new ExactSyncImages(ExactPolicyImages(queue_size), sub_l_raw_image_, sub_r_raw_image_));
            exact_sync_images_->registerCallback(boost::bind(&StereoProcessor::imageCb, this, _1, _2));
        }
        else
        {
            exact_sync_images_and_info_.reset(new ExactSyncImagesAndInfo(ExactPolicyImagesAndInfo(queue_size), sub_l_raw_image_, sub_l_info_, sub_r_raw_image_, sub_r_info_));
            exact_sync_images_and_info_->registerCallback(boost::bind(&StereoProcessor::imageAndInfoCb, this, _1, _2, _3, _4));
        }
    }

    // Set up dynamic reconfiguration
    ReconfigureServer::CallbackType f = boost::bind(&StereoProcessor::configCb, this, _1, _2);
    reconfigure_server_.reset(new ReconfigureServer(config_mutex_, private_nh));
    reconfigure_server_->setCallback(f);

    // Monitor whether anyone is subscribed to the output
    ros::SubscriberStatusCallback connect_cb = boost::bind(&StereoProcessor::connectCb, this);
    // Make sure we don't enter connectCb() between advertising and assigning to pub_disparity_
    boost::lock_guard<boost::mutex> lock(connect_mutex_);
    int publisher_queue_size;
    private_nh.param("publisher_queue_size", publisher_queue_size, 1);
    pub_mono_left_        = private_nh.advertise<sensor_msgs::Image>("left/image_mono", publisher_queue_size, connect_cb, connect_cb);
    pub_mono_right_       = private_nh.advertise<sensor_msgs::Image>("right/image_mono", publisher_queue_size, connect_cb, connect_cb);
    pub_color_left_       = private_nh.advertise<sensor_msgs::Image>("left/image_color", publisher_queue_size, connect_cb, connect_cb);
    pub_color_right_      = private_nh.advertise<sensor_msgs::Image>("right/image_color", publisher_queue_size, connect_cb, connect_cb);
    pub_mono_rect_left_   = private_nh.advertise<sensor_msgs::Image>("left/rect_mono", publisher_queue_size, connect_cb, connect_cb);
    pub_color_rect_left_  = private_nh.advertise<sensor_msgs::Image>("left/rect_color", publisher_queue_size, connect_cb, connect_cb);
    pub_mono_rect_right_  = private_nh.advertise<sensor_msgs::Image>("right/rect_mono", publisher_queue_size, connect_cb, connect_cb);
    pub_color_rect_right_ = private_nh.advertise<sensor_msgs::Image>("right/rect_color", publisher_queue_size, connect_cb, connect_cb);
    pub_disparity_        = private_nh.advertise<stereo_msgs::DisparityImage>("disparity", publisher_queue_size, connect_cb, connect_cb);
    pub_disparity_vis_    = private_nh.advertise<sensor_msgs::Image>("disparity_vis", publisher_queue_size, connect_cb, connect_cb);
    pub_pointcloud_       = private_nh.advertise<sensor_msgs::PointCloud2>("pointcloud", publisher_queue_size, connect_cb, connect_cb);
}

// Handles (un)subscribing when clients (un)subscribe
void StereoProcessor::connectCb()
{
    boost::lock_guard<boost::mutex> connect_lock(connect_mutex_);
    connected_.DebayerMonoLeft   = (pub_mono_left_.getNumSubscribers() > 0) ? 1 : 0;
    connected_.DebayerMonoRight  = (pub_mono_right_.getNumSubscribers() > 0) ? 1 : 0;
    connected_.DebayerColorLeft  = (pub_color_left_.getNumSubscribers() > 0) ? 1 : 0;
    connected_.DebayerColorRight = (pub_color_right_.getNumSubscribers() > 0) ? 1 : 0;
    connected_.RectifyMonoLeft   = (pub_mono_rect_left_.getNumSubscribers() > 0) ? 1 : 0;
    connected_.RectifyMonoRight  = (pub_mono_rect_right_.getNumSubscribers() > 0) ? 1 : 0;
    connected_.RectifyColorLeft  = (pub_color_rect_left_.getNumSubscribers() > 0) ? 1 : 0;
    connected_.RectifyColorRight = (pub_color_rect_right_.getNumSubscribers() > 0) ? 1 : 0;
    connected_.Disparity         = (pub_disparity_.getNumSubscribers() > 0) ? 1 : 0;
    connected_.DisparityVis      = (pub_disparity_vis_.getNumSubscribers() > 0) ? 1 : 0;
    connected_.Pointcloud        = (pub_pointcloud_.getNumSubscribers() > 0) ? 1 : 0;
    int level                    = connected_.level();
    if (level == 0)
    {
        ROS_INFO("Un-subscribing from images and camera infos");
        sub_l_raw_image_.unsubscribe();
        sub_l_info_.unsubscribe();
        sub_r_raw_image_.unsubscribe();
        sub_r_info_.unsubscribe();
    }
    else if (!sub_l_raw_image_.getSubscriber())
    {
        // Queue size 1 should be OK; the one that matters is the synchronizer queue size.
        /// @todo Allow remapping left, right?
        image_transport::TransportHints hints("raw", ros::TransportHints(), private_nh);
        ROS_INFO("Subscribing to raw images");
        sub_l_raw_image_.subscribe(*it_, "left/image_raw", 1, hints);
        sub_r_raw_image_.subscribe(*it_, "right/image_raw", 1, hints);
        if (!camera_info_from_files_)
        {
            ROS_INFO("Subscribing to camera infos");
            sub_l_info_.subscribe(nh, "left/camera_info", 1);
            sub_r_info_.subscribe(nh, "right/camera_info", 1);
        }
    }
}

void StereoProcessor::imageAndInfoCb(const sensor_msgs::ImageConstPtr &l_raw_msg, const sensor_msgs::CameraInfoConstPtr &l_info_msg, const sensor_msgs::ImageConstPtr &r_raw_msg,
                                     const sensor_msgs::CameraInfoConstPtr &r_info_msg)
{
    if (!stereoProcessor_->isStereoModelInitialised())
    {
        if (!camera_info_from_files_)
        {
            stereoProcessor_->initStereoModel(l_info_msg, r_info_msg);
        }
    }
    imageCb(l_raw_msg, r_raw_msg);
}

void StereoProcessor::imageCb(const sensor_msgs::ImageConstPtr &l_raw_msg, const sensor_msgs::ImageConstPtr &r_raw_msg)
{
    boost::lock_guard<boost::recursive_mutex> config_lock(config_mutex_);
    boost::lock_guard<boost::mutex> connect_lock(connect_mutex_);
    boost::timer perf_timer;
    int level = connected_.level();
    ROS_DEBUG("got images, level %d", level);

    // TODO depend on timing there are potentially senders that did not completed yet...
    stereoProcessor_->cleanSenders();
    if (!stereoProcessor_->isStereoModelInitialised())
    {
        if (camera_info_from_files_)
        {
            stereoProcessor_->initStereoModel(camera_info_file_left_, camera_info_file_right_);
        }
    }

    // Create cv::Mat views onto all buffers
    auto l_cpu_raw = cv_bridge::toCvShare(l_raw_msg, l_raw_msg->encoding);
    auto r_cpu_raw = cv_bridge::toCvShare(r_raw_msg, r_raw_msg->encoding);
    cv::cuda::registerPageLocked(const_cast<cv::Mat &>(l_cpu_raw->image));
    cv::cuda::registerPageLocked(const_cast<cv::Mat &>(r_cpu_raw->image));

    stereoProcessor_->uploadMat(GpuMatSource::GPU_MAT_SRC_L_RAW, l_cpu_raw->image, l_cpu_raw->encoding);
    stereoProcessor_->uploadMat(GpuMatSource::GPU_MAT_SRC_R_RAW, r_cpu_raw->image, r_cpu_raw->encoding);

    if (connected_.DebayerMonoLeft || connected_.RectifyMonoLeft || connected_.Disparity || connected_.DisparityVis || connected_.Pointcloud)
    {
        stereoProcessor_->convertRawToMono(GPU_MAT_SIDE_L);
    }
    if (connected_.DebayerMonoLeft)
    {
        stereoProcessor_->enqueueSendImage(GPU_MAT_SRC_L_MONO, l_raw_msg, sensor_msgs::image_encodings::MONO8, &pub_mono_left_);
    }

    if (connected_.DebayerMonoRight || connected_.RectifyMonoRight || connected_.Disparity || connected_.DisparityVis || connected_.Pointcloud)
    {
        stereoProcessor_->convertRawToMono(GPU_MAT_SIDE_R);
    }
    if (connected_.DebayerMonoRight)
    {
        stereoProcessor_->enqueueSendImage(GPU_MAT_SRC_R_MONO, r_raw_msg, sensor_msgs::image_encodings::MONO8, &pub_mono_right_);
    }

    if (connected_.DebayerColorLeft || connected_.RectifyColorLeft || connected_.Pointcloud)
    {
        stereoProcessor_->convertRawToColor(GPU_MAT_SIDE_L);
    }
    if (connected_.DebayerColorLeft)
    {
        stereoProcessor_->enqueueSendImage(GPU_MAT_SRC_L_COLOR, l_raw_msg, sensor_msgs::image_encodings::BGR8, &pub_color_left_);
    }

    if (connected_.DebayerColorRight || connected_.RectifyColorRight)
    {
        stereoProcessor_->convertRawToColor(GPU_MAT_SIDE_R);
    }
    if (connected_.DebayerColorRight)
    {
        stereoProcessor_->enqueueSendImage(GPU_MAT_SRC_R_COLOR, r_raw_msg, sensor_msgs::image_encodings::BGR8, &pub_color_right_);
    }

    if (connected_.RectifyMonoLeft || connected_.Disparity || connected_.DisparityVis || connected_.Pointcloud)
    {
        stereoProcessor_->rectifyImage(GPU_MAT_SRC_L_MONO, GPU_MAT_SRC_L_RECT_MONO, cv::InterpolationFlags::INTER_LINEAR);
    }
    if (connected_.RectifyMonoRight || connected_.Disparity || connected_.DisparityVis || connected_.Pointcloud)
    {
        stereoProcessor_->rectifyImage(GPU_MAT_SRC_R_MONO, GPU_MAT_SRC_R_RECT_MONO, cv::InterpolationFlags::INTER_LINEAR);
    }

    if (connected_.RectifyMonoLeft)
    {
        stereoProcessor_->enqueueSendImage(GPU_MAT_SRC_L_RECT_MONO, l_raw_msg, sensor_msgs::image_encodings::MONO8, &pub_mono_rect_left_);
    }

    if (connected_.RectifyMonoRight)
    {
        stereoProcessor_->enqueueSendImage(GPU_MAT_SRC_R_RECT_MONO, r_raw_msg, sensor_msgs::image_encodings::MONO8, &pub_mono_rect_right_);
    }

    if (connected_.RectifyColorLeft || connected_.Pointcloud)
    {
        stereoProcessor_->rectifyImage(GPU_MAT_SRC_L_COLOR, GPU_MAT_SRC_L_RECT_COLOR, cv::INTER_LINEAR);
    }
    if (connected_.RectifyColorRight)
    {
        stereoProcessor_->rectifyImage(GPU_MAT_SRC_R_COLOR, GPU_MAT_SRC_R_RECT_COLOR, cv::INTER_LINEAR);
    }

    if (connected_.RectifyColorLeft)
    {
        stereoProcessor_->enqueueSendImage(GPU_MAT_SRC_L_RECT_COLOR, l_raw_msg, sensor_msgs::image_encodings::BGR8, &pub_color_rect_left_);
    }

    if (connected_.RectifyColorRight)
    {
        stereoProcessor_->enqueueSendImage(GPU_MAT_SRC_R_RECT_COLOR, r_raw_msg, sensor_msgs::image_encodings::BGR8, &pub_color_rect_right_);
    }

    if (connected_.Disparity || connected_.DisparityVis || connected_.Pointcloud)
    {
        stereoProcessor_->computeDisparity(GPU_MAT_SRC_L_RECT_MONO, GPU_MAT_SRC_R_RECT_MONO, GPU_MAT_SRC_L_DISPARITY);
    }

    if (connected_.Disparity)
    {
        stereoProcessor_->enqueueSendDisparity(GPU_MAT_SRC_L_DISPARITY_32F, l_raw_msg, &pub_disparity_);
    }

    if (connected_.DisparityVis)
    {
        // TODO
    }
    stereoProcessor_->waitForAllStreams();
    double duration = perf_timer.elapsed();
    ROS_INFO("Image Callback took: %.2f [ms]", duration * 1000.0);
    cv::cuda::unregisterPageLocked(const_cast<cv::Mat &>(l_cpu_raw->image));
    cv::cuda::unregisterPageLocked(const_cast<cv::Mat &>(r_cpu_raw->image));
}

inline bool isValidPoint(const cv::Vec3f &pt)
{
    // Check both for disparities explicitly marked as invalid (where OpenCV maps pt.z to MISSING_Z)
    // and zero disparities (point mapped to infinity).
    return pt[2] != image_geometry::StereoCameraModel::MISSING_Z && !std::isinf(pt[2]);
}

void StereoProcessor::filterSpeckles(void)
{
    cv::Mat disp = filter_buf_.createMatHeader();
    cv::filterSpeckles(disp, 0, maxSpeckleSize_, maxDiff_ * 256);
}

void StereoProcessor::configCb(Config &config, uint32_t level)
{
    // Tweak all settings to be valid
    config.correlation_window_size |= 0x1;                       // must be odd
    config.disparity_range = (config.disparity_range / 16) * 16; // must be multiple of 16

    stereoProcessor_->setPreFilterType(config.xsobel ? cv::cuda::StereoBM::PREFILTER_XSOBEL : cv::cuda::StereoBM::PREFILTER_NORMALIZED_RESPONSE);
    stereoProcessor_->setRefineDisparity(config.refine_disparity);
    stereoProcessor_->setBlockSize(config.correlation_window_size);
    stereoProcessor_->setNumDisparities(config.disparity_range);
    stereoProcessor_->setTextureThreshold(config.texture_threshold);
    ROS_INFO("Reconfigure winsz:%d ndisp:%d tex:%3.1f", config.correlation_window_size, config.disparity_range, config.texture_threshold);

    //    if (bilateral_filter_.empty())
    //    {
    //        bilateral_filter_ = cv::cuda::createDisparityBilateralFilter(config.filter_ndisp, config.filter_radius, config.filter_iters);
    //    }
    //    bilateral_filter_->setNumDisparities(config.filter_ndisp);
    //    bilateral_filter_->setRadius(config.filter_radius);
    //    bilateral_filter_->setNumIters(config.filter_iters);

    //    bilateral_filter_->setEdgeThreshold(config.filter_edge_threshold);
    //    bilateral_filter_->setMaxDiscThreshold(config.filter_max_disc_threshold);
    //    bilateral_filter_->setSigmaRange(config.filter_sigma_range);
    //    bilateral_filter_enabled_ = config.bilateral_filter;
    //    maxDiff_                  = config.max_diff;
    //    maxSpeckleSize_           = config.max_speckle_size;
}

} // namespace stereo_image_proc
