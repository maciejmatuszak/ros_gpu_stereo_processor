#include "gpuimageproc/gpustereoprocessor.h"
#include <camera_calibration_parsers/parse.h>

namespace gpuimageproc
{

GpuStereoProcessor::GpuStereoProcessor()
{
    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
    block_matcher_ = cv::cuda::createStereoBM(128, 19);
}

void GpuStereoProcessor::initStereoModel(const sensor_msgs::CameraInfoConstPtr &l_info_msg, const sensor_msgs::CameraInfoConstPtr &r_info_msg)
{
    // Update the camera model
    l_cam_name_ = "left";
    l_cam_name_ = "right";
    model_.fromCameraInfo(l_info_msg, r_info_msg);
}

void GpuStereoProcessor::initStereoModel(const std::string& left_cal_file, const std::string& right_cal_file)
{
    sensor_msgs::CameraInfo l_info, r_info;
    camera_calibration_parsers::readCalibration(left_cal_file, l_cam_name_, l_info);
    camera_calibration_parsers::readCalibration(right_cal_file, r_cam_name, r_info);
    model_.fromCameraInfo(l_info, r_info);
}

void GpuStereoProcessor::uploadMat(GpuMatSource mat_source, const cv::Mat &cv_mat)
{
    auto gmat = getGpuMat(mat_source);
    gmat->upload(cv_mat, getStream(mat_source));
}

void GpuStereoProcessor::downloadMat(GpuMatSource mat_source, const cv::Mat &cv_mat)
{
    auto gmat = getGpuMat(mat_source);
    cv::cuda::createContinuous(gmat->size(), gmat->type(), cv_mat);
    gmat->download(cv_mat, getStream(mat_source));
    getStream(mat_source).waitForCompletion();
}

boost::shared_ptr<cv::cuda::GpuMat> GpuStereoProcessor::getGpuMat(GpuMatSource source)
{
    boost::shared_ptr<cv::cuda::GpuMat> gpuMat;
    auto str_source = std::to_string(source);
    if (gpu_mats.find(str_source) == gpu_mats.end())
    {
        gpuMat = boost::make_shared<cv::cuda::GpuMat>();
        gpu_mats[str_source] = gpuMat;
        return gpu_mats[str_source];
    }
    else
    {
        return gpu_mats[str_source];
    }
}

cv::cuda::Stream &GpuStereoProcessor::getStream(GpuMatSource source)
{
    if ((source & GPU_MAT_SIDE_L) == GPU_MAT_SIDE_L)
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
    t->enqueueSend(*getGpuMat(source), getStream(source));
}

void GpuStereoProcessor::enqueueSendDisparity(GpuMatSource source, const sensor_msgs::ImageConstPtr &imagePattern, ros::Publisher &pub)
{
    GPUSender::Ptr t = boost::make_shared<GPUSender>(imagePattern, block_matcher_->getBlockSize(), block_matcher_->getNumDisparities(), block_matcher_->getMinDisparity(), pub);
    senders.push_back(t);
    t->enqueueSend(*getGpuMat(source), getStream(source));
}

void GpuStereoProcessor::colorConvertImage(GpuMatSource source, GpuMatSource dest, int colorConversion, int dcn)
{
    cv::cuda::demosaicing(*getGpuMat(source), *getGpuMat(dest), colorConversion, dcn, getStream(source));
}

void GpuStereoProcessor::rectifyImage(GpuMatSource source, GpuMatSource dest, cv::InterpolationFlags interpolation)
{
    if ((source & GPU_MAT_SIDE_L) == GPU_MAT_SIDE_L)
    {
        model_.left().rectifyImageGPU(*getGpuMat(source), *getGpuMat(dest), interpolation, getStream(source));
    }
    else
    {
        model_.right().rectifyImageGPU(*getGpuMat(source), *getGpuMat(dest), interpolation, getStream(source));
    }
}

void GpuStereoProcessor::computeDisparity(GpuMatSource left, GpuMatSource right, GpuMatSource disparity)
{
    // Fixed-point disparity is 16 times the true value: d = d_fp / 16.0 = x_l - x_r.
    static const int DPP = 16; // disparities per pixel
    static const double inv_dpp = 1.0 / DPP;

    getStream(static_cast<GpuMatSource>(disparity ^ GPU_MAT_SIDE_MASK)).waitForCompletion();

    // Block matcher produces 16-bit signed (fixed point) disparity image
    auto lgpu = getGpuMat(left);
    auto rgpu = getGpuMat(right);
    auto dgpu = getGpuMat(disparity);
    block_matcher_->compute(*lgpu, *rgpu, *dgpu, getStream(disparity));

    GpuMatSource disparity_f32 = static_cast<GpuMatSource>((disparity & GPU_MAT_SIDE_MASK) | GPU_MAT_SRC_DISPARITY_32F);
    //         side left or right \___________________________/    ^---plus disparity 32F selector
    // TODO:the x offset will be different for right side for now just use assert to prevent the use of Right disparity
    assert((disparity & GPU_MAT_SIDE_MASK) == GPU_MAT_SIDE_L);

    getGpuMat(disparity)->convertTo(*getGpuMat(disparity_f32), CV_32FC1, inv_dpp, -(model_.left().cx() - model_.right().cx()));
}

void GpuStereoProcessor::waitForStream(GpuMatSource stream_source) { getStream(stream_source).waitForCompletion(); }

void GpuStereoProcessor::waitForAllStreams()
{
    l_strm.waitForCompletion();
    r_strm.waitForCompletion();
}

void GpuStereoProcessor::cleanSenders() { senders.clear(); }

void GpuStereoProcessor::setPreFilterType(int filter_type) { block_matcher_->setPreFilterType(filter_type); }

void GpuStereoProcessor::setRefineDisparity(bool ref_disp) { block_matcher_->setRefineDisparity(ref_disp); }

void GpuStereoProcessor::setBlockSize(int block_size) { block_matcher_->setBlockSize(block_size); }

void GpuStereoProcessor::setNumDisparities(int numDisp) { block_matcher_->setNumDisparities(numDisp); }

void GpuStereoProcessor::setTextureThreshold(int threshold) { block_matcher_->setTextureThreshold(threshold); }

} // namespace
