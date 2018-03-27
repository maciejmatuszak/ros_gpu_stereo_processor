#include "gpuimageproc/gpustereoprocessor.h"

namespace gpuimageproc
{

GpuStereoProcessor::GpuStereoProcessor() { cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice()); }

void GpuStereoProcessor::initStereoModel(const sensor_msgs::CameraInfoConstPtr &l_info_msg, const sensor_msgs::CameraInfoConstPtr &r_info_msg)
{
    // Update the camera model
    model_.fromCameraInfo(l_info_msg, r_info_msg);
}

void GpuStereoProcessor::uploadMat(GpuMatSource mat_source, const cv::Mat &cv_mat) { getGpuMat(mat_source).upload(cv_mat, getStream(mat_source)); }

cv::cuda::GpuMat &GpuStereoProcessor::getGpuMat(GpuMatSource source)
{
    switch (source)
    {
    case GpuMatSource::GPU_MAT_SRC_L_MONO:
        return l_mono;
    case GpuMatSource::GPU_MAT_SRC_R_MONO:
        return r_mono;
    default:
        ROS_ERROR("invalid source in getGpuMat: %d", source);
    }
}

cv::cuda::Stream &GpuStereoProcessor::getStream(GpuMatSource source)
{
    if (source & GPU_MAT_SIDE_L == GPU_MAT_SIDE_L)
    {
        return l_strm;
    }
    else // if(source & GPU_MAT_SRC_MASK_R == GPU_MAT_SRC_MASK_R)
    {
        return r_strm;
    }
}

void GpuStereoProcessor::enqueueSendImage(GpuMatSource source, const sensor_msgs::ImageConstPtr &imagePattern, std::string encoding, ros::Publisher &pub)
{
    GPUSender::Ptr t = boost::make_shared<GPUSender>(imagePattern, encoding, pub);
    senders.push_back(t);
    t->enqueueSend(getGpuMat(source), getStream(source));
}

void GpuStereoProcessor::enqueueSendDisparity(const sensor_msgs::ImageConstPtr &imagePattern, ros::Publisher &pub)
{
    pub_disparity_ = pub;
    // Allocate new disparity image message
    disp_msg_.reset(new stereo_msgs::DisparityImage());
    disp_msg_->header = imagePattern->header;
    disp_msg_->image.header = imagePattern->header;

    // Compute window of (potentially) valid disparities
    int border = block_matcher_->getBlockSize() / 2;
    int left = block_matcher_->getNumDisparities() + block_matcher_->getMinDisparity() + border - 1;
    int wtf = (block_matcher_->getMinDisparity() >= 0) ? border + block_matcher_->getMinDisparity() : std::max(border, -block_matcher_->getMinDisparity());
    int right = disp_msg_->image.width - 1 - wtf;
    int top = border;
    int bottom = disp_msg_->image.height - 1 - border;
    disp_msg_->valid_window.x_offset = left;
    disp_msg_->valid_window.y_offset = top;
    disp_msg_->valid_window.width = right - left;
    disp_msg_->valid_window.height = bottom - top;
    disp_msg_->min_disparity = block_matcher_->getMinDisparity() + 1;
    disp_msg_->max_disparity = block_matcher_->getMinDisparity() + block_matcher_->getNumDisparities() - 1;

    disp_msg_->image.height = l_rect_mono.rows;
    disp_msg_->image.width = l_rect_mono.cols;
    disp_msg_->image.encoding = sensor_msgs::image_encodings::TYPE_32FC1;
    disp_msg_->image.step = l_rect_mono.cols * sizeof(float);
    disp_msg_->image.data.resize(disp_msg_->image.step * disp_msg_->image.height);
    cv::Mat_<float> disp_msg_data_ = cv::Mat_<float>(disp_msg_->image.height, disp_msg_->image.width, (float *)&disp_msg_->image.data[0], disp_msg_->image.step);
    cv::cuda::registerPageLocked(disp_msg_data_);

    disparity_32F.download(disp_msg_data_, l_strm);

    l_strm.enqueueHostCallback(
        [](int status, void *userData) {
            (void)status;
            pub_disparity_.publish(disp_msg_);
            static_cast<GpuStereoProcessor *>(userData)->sendDisparity();
        },
        (void *)this);
}

void GpuStereoProcessor::colorConvertImage(GpuMatSource source, GpuMatSource dest, int colorConversion, int dcn)
{
    cv::cuda::demosaicing(getGpuMat(source), getGpuMat(dest), colorConversion, dcn, getStream(source));
}

void GpuStereoProcessor::rectifyImage(GpuMatSource source, GpuMatSource dest, cv::InterpolationFlags interpolation)
{
    if (source & GPU_MAT_SIDE_L == GPU_MAT_SIDE_L)
    {
        model_.left().rectifyImageGPU(getGpuMat(source), getGpuMat(dest), interpolation, getStream(source));
    }
    else
    {
        model_.right().rectifyImageGPU(getGpuMat(source), getGpuMat(dest), interpolation, getStream(source));
    }
}

void GpuStereoProcessor::computeDisparity()
{
    // Fixed-point disparity is 16 times the true value: d = d_fp / 16.0 = x_l - x_r.
    static const int DPP = 16; // disparities per pixel
    static const double inv_dpp = 1.0 / DPP;

    r_strm.waitForCompletion();

    // Block matcher produces 16-bit signed (fixed point) disparity image
    block_matcher_->compute(l_rect_mono, r_rect_mono, disparity, l_strm);
    disparity.convertTo(disparity_32F, CV_32FC1, inv_dpp, -(model_.left().cx() - model_.right().cx()));
}

} // namespace
