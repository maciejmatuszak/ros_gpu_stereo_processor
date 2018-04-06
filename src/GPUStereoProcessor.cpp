#include "gpuimageproc/GPUStereoProcessor.h"
#include <camera_calibration_parsers/parse.h>
#include <cv_bridge/cv_bridge.h>

namespace gpuimageproc
{

GpuStereoProcessor::GpuStereoProcessor()
    : l_raw_encoding_("")
    , r_raw_encoding_("")

{
    // cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
    block_matcher_gpu_ = cv::cuda::createStereoBM(128, 19);
    block_matcher_gpu_->setRefineDisparity(false);
    block_matcher_cpu_ = cv::StereoBM::create(128, 19);

    block_matcher_cpu_->setBlockSize(block_matcher_gpu_->getBlockSize());
    block_matcher_cpu_->setDisp12MaxDiff(block_matcher_gpu_->getDisp12MaxDiff());
    block_matcher_cpu_->setMinDisparity(block_matcher_gpu_->getMinDisparity());
    block_matcher_cpu_->setNumDisparities(block_matcher_gpu_->getNumDisparities());
    block_matcher_cpu_->setPreFilterCap(block_matcher_gpu_->getPreFilterCap());
    block_matcher_gpu_->setPreFilterSize(5);
    block_matcher_cpu_->setPreFilterSize(5);
    block_matcher_gpu_->setPreFilterType(cv::StereoBM::PREFILTER_XSOBEL);
    block_matcher_cpu_->setPreFilterType(block_matcher_gpu_->getPreFilterType());
    block_matcher_cpu_->setROI1(block_matcher_gpu_->getROI1());
    block_matcher_cpu_->setROI2(block_matcher_gpu_->getROI2());
    block_matcher_cpu_->setSmallerBlockSize(block_matcher_gpu_->getSmallerBlockSize());
    block_matcher_cpu_->setSpeckleRange(block_matcher_gpu_->getSpeckleRange());
    block_matcher_gpu_->setSpeckleWindowSize(0);
    block_matcher_cpu_->setSpeckleWindowSize(0);
    block_matcher_cpu_->setTextureThreshold(block_matcher_gpu_->getTextureThreshold());
    block_matcher_cpu_->setUniquenessRatio(block_matcher_gpu_->getUniquenessRatio());
}

void GpuStereoProcessor::initStereoModel(const sensor_msgs::CameraInfoConstPtr &l_info_msg, const sensor_msgs::CameraInfoConstPtr &r_info_msg)
{
    // Update the camera model
    l_cam_name_ = "left";
    l_cam_name_ = "right";
    model_.fromCameraInfo(l_info_msg, r_info_msg);
    ROS_INFO("camera model initialised from messages");
}

void GpuStereoProcessor::initStereoModel(const std::string &left_cal_file, const std::string &right_cal_file)
{
    sensor_msgs::CameraInfo l_info, r_info;
    camera_calibration_parsers::readCalibration(left_cal_file, l_cam_name_, l_info);
    camera_calibration_parsers::readCalibration(right_cal_file, r_cam_name, r_info);
    model_.fromCameraInfo(l_info, r_info);
    ROS_INFO("camera model initialised from files");
}

bool GpuStereoProcessor::isStereoModelInitialised() { return model_.initialized(); }

void GpuStereoProcessor::convertRawToColor(GpuMatSource side)
{
    if ((side & GPU_MAT_SIDE_MASK) == GPU_MAT_SIDE_L)
    {
        convertColor(GPU_MAT_SRC_L_RAW, GPU_MAT_SRC_L_COLOR, l_raw_encoding_, sensor_msgs::image_encodings::BGR8);
    }
    else
    {
        convertColor(GPU_MAT_SRC_R_RAW, GPU_MAT_SRC_R_COLOR, r_raw_encoding_, sensor_msgs::image_encodings::BGR8);
    }
}

void GpuStereoProcessor::convertRawToMono(GpuMatSource side)
{

    if ((side & GPU_MAT_SIDE_MASK) == GPU_MAT_SIDE_L)
    {
        convertColor(GPU_MAT_SRC_L_RAW, GPU_MAT_SRC_L_MONO, l_raw_encoding_, sensor_msgs::image_encodings::MONO8);
    }
    else
    {
        convertColor(GPU_MAT_SRC_R_RAW, GPU_MAT_SRC_R_MONO, r_raw_encoding_, sensor_msgs::image_encodings::MONO8);
    }
}
void GpuStereoProcessor::uploadMat(GpuMatSource mat_source, const cv::Mat &cv_mat, std::string encoding)
{
    auto gmat = getGpuMat(mat_source);
    gmat->upload(cv_mat, getStream(mat_source));
    if (encoding.empty())
    {
        return;
    }

    // store the encoding
    if ((mat_source & GPU_MAT_SIDE_MASK) == GPU_MAT_SIDE_L)
    {
        l_raw_encoding_ = encoding;
    }
    else
    {
        r_raw_encoding_ = encoding;
    }
}

void GpuStereoProcessor::downloadMat(GpuMatSource mat_source, const cv::Mat &cv_mat)
{
    auto gmat = getGpuMat(mat_source);
    cv::cuda::createContinuous(gmat->size(), gmat->type(), cv_mat);
    gmat->download(cv_mat, getStream(mat_source));
    getStream(mat_source).waitForCompletion();
}

void GpuStereoProcessor::convertColor(GpuMatSource mat_source, GpuMatSource mat_dst, const std::string &src_encoding, const std::string &dst_encoding)
{
    assert(!src_encoding.empty());
    assert(!dst_encoding.empty());
    // Copy metadata
    auto srcMat    = getGpuMat(mat_source);
    auto dstMat    = getGpuMat(mat_dst);
    auto srcStream = getStream(mat_source);

    // Copy to new buffer if same encoding requested
    if (dst_encoding.empty() || dst_encoding == src_encoding)
    {
        srcMat->copyTo(*dstMat);
    }
    else
    {
        // Convert the source data to the desired encoding
        const std::vector<int> conversion_codes = cv_bridge::getConversionCode(src_encoding, dst_encoding);
        assert(conversion_codes.size() == 1);
        // for (size_t i = 0; i < conversion_codes.size(); ++i)
        //{
        int conversion_code = conversion_codes[0];
        if (conversion_code == cv_bridge::SAME_FORMAT)
        {
            // Same number of channels, but different bit depth
            int src_depth = sensor_msgs::image_encodings::bitDepth(src_encoding);
            int dst_depth = sensor_msgs::image_encodings::bitDepth(dst_encoding);
            // Keep the number of channels for now but changed to the final depth
            // int image2_type = CV_MAKETYPE(CV_MAT_DEPTH(getCvType(dst_encoding)), image1.channels());
            int image2_type = cv_bridge::getCvType(dst_encoding);

            // Do scaling between CV_8U [0,255] and CV_16U [0,65535] images.
            if (src_depth == 8 && dst_depth == 16)
                srcMat->convertTo(*dstMat, image2_type, 65535. / 255., srcStream);
            else if (src_depth == 16 && dst_depth == 8)
                srcMat->convertTo(*dstMat, image2_type, 255. / 65535., srcStream);
            else
                srcMat->convertTo(*dstMat, image2_type, srcStream);
        }
        else
        {
            // Perform color conversion
            int dcn = sensor_msgs::image_encodings::numChannels(dst_encoding);
            cv::cuda::cvtColor(*srcMat, *dstMat, conversion_code, dcn, srcStream);
        }
        // image1 = image2;
        //}
    }
}

boost::shared_ptr<cv::cuda::GpuMat> GpuStereoProcessor::getGpuMat(GpuMatSource source)
{
    boost::shared_ptr<cv::cuda::GpuMat> gpuMat;
    auto str_source = std::to_string(source);
    if (gpu_mats.find(str_source) == gpu_mats.end())
    {
        gpuMat               = boost::make_shared<cv::cuda::GpuMat>();
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

GPUSender::Ptr GpuStereoProcessor::enqueueSendImage(GpuMatSource source, const sensor_msgs::ImageConstPtr &imagePattern, std::string encoding, ros::Publisher *pub)
{
    GPUSender::Ptr t = boost::make_shared<GPUSender>(imagePattern, encoding, pub);
    senders.push_back(t);
    t->enqueueSend(*getGpuMat(source), getStream(source));
    return t;
}

GPUSender::Ptr GpuStereoProcessor::enqueueSendDisparity(GpuMatSource source, const sensor_msgs::ImageConstPtr &imagePattern, ros::Publisher *pub)
{
    GPUSender::Ptr t = boost::make_shared<GPUSender>(imagePattern, block_matcher_gpu_->getBlockSize(), block_matcher_gpu_->getNumDisparities(), model_.right().fx(),
                                                     model_.baseline(), block_matcher_gpu_->getMinDisparity(), pub);
    senders.push_back(t);
    t->enqueueSend(*getGpuMat(source), getStream(source));
    return t;
}

GPUSender::Ptr GpuStereoProcessor::enqueueSendPoints(GpuMatSource points_source, GpuMatSource color_source, const sensor_msgs::ImageConstPtr &imagePattern, ros::Publisher *pub)
{
    GPUSender::Ptr t = boost::make_shared<GPUSender>(imagePattern->header, pub);
    senders.push_back(t);
    t->enqueueSend(*getGpuMat(points_source), *getGpuMat(color_source), getStream(points_source));
    return t;
}

void GpuStereoProcessor::rectifyImage(GpuMatSource source, GpuMatSource dest, cv::InterpolationFlags interpolation)
{
    assert(model_.initialized());
    if ((source & GPU_MAT_SIDE_L) == GPU_MAT_SIDE_L)
    {
        model_.left().rectifyImageGPU(*getGpuMat(source), *getGpuMat(dest), interpolation, getStream(source));
    }
    else
    {
        model_.right().rectifyImageGPU(*getGpuMat(source), *getGpuMat(dest), interpolation, getStream(source));
    }
}

void GpuStereoProcessor::rectifyImageLeft(const cv::Mat &source, cv::Mat &dest, cv::InterpolationFlags interpolation)
{
    assert(model_.initialized());
    model_.left().rectifyImage(source, dest, interpolation);
}

void GpuStereoProcessor::rectifyImageRight(const cv::Mat &source, cv::Mat &dest, cv::InterpolationFlags interpolation)
{
    assert(model_.initialized());
    model_.right().rectifyImage(source, dest, interpolation);
}

void GpuStereoProcessor::computeDisparity(GpuMatSource left, GpuMatSource right, GpuMatSource disparity)
{
    assert(model_.initialized());
    // Fixed-point disparity is 16 times the true value: d = d_fp / 16.0 = x_l - x_r.
    static const int DPP        = 16; // disparities per pixel
    static const double inv_dpp = 1.0 / DPP;

    // Block matcher produces 16-bit signed (fixed point) disparity image
    auto lgpu    = getGpuMat(left);
    auto rgpu    = getGpuMat(right);
    auto dgpu    = getGpuMat(disparity);
    double shift = -(model_.left().cx() - model_.right().cx());
    ROS_INFO("model Left Cx:%f; Right Cx:%f; baseline:%f", model_.left().cx(), model_.right().cx(), model_.baseline());
    GpuMatSource disparity_f32 = static_cast<GpuMatSource>((disparity & GPU_MAT_SIDE_MASK) | GPU_MAT_SRC_DISPARITY_32F);
    auto dgpu32                = getGpuMat(disparity_f32);

    block_matcher_gpu_->compute(*lgpu, *rgpu, *dgpu, getStream(disparity));

    //         side left or right \___________________________/    ^---plus disparity 32F selector
    // TODO:the x offset will be different for right side for now just use assert to prevent the use of Right disparity
    assert((disparity & GPU_MAT_SIDE_MASK) == GPU_MAT_SIDE_L);

    cv::Mat disp;
    dgpu->download(disp);
    printStats("Disparity clean", disp);

    dgpu->convertTo(*dgpu32, CV_32FC1, inv_dpp, shift);
    // dgpu->convertTo(*dgpu32, CV_32FC1, 1      , shift);

    dgpu32->download(disp);
    printStats("Disparity scaled", disp);
}

void GpuStereoProcessor::computeDisparityImage(GpuMatSource disparity_src, GpuMatSource disp_image_dest)
{
    assert(model_.initialized());
    auto disparity = getGpuMat(disparity_src);
    auto image     = getGpuMat(disp_image_dest);
    auto ndisp     = block_matcher_gpu_->getNumDisparities();
    cv::cuda::drawColorDisp(*disparity, *image, ndisp, getStream(disparity_src));
}
void GpuStereoProcessor::projectDisparityTo3DPoints(GpuMatSource disparity_src, GpuMatSource points_src)
{
    assert(model_.initialized());
    model_.projectDisparityImageTo3dGPU(*getGpuMat(disparity_src), /* disparity gpu mat */
                                        *getGpuMat(points_src),    /* points gpu mat */
                                        true,                      /* handle missing points */
                                        getStream(disparity_src)   /* gpu stream */
                                        );
}

void GpuStereoProcessor::computeDisparity(cv::Mat &left, cv::Mat &right, cv::Mat &disparity)
{
    assert(model_.initialized());
    // Fixed-point disparity is 16 times the true value: d = d_fp / 16.0 = x_l - x_r.
    static const int DPP        = 16; // disparities per pixel
    static const double inv_dpp = 1.0 / DPP;

    block_matcher_cpu_->compute(left, right, disparity);
    disparity.convertTo(disparity, CV_32FC1, inv_dpp, -(model_.left().cx() - model_.right().cx()));
}

void GpuStereoProcessor::waitForStream(GpuMatSource stream_source) { getStream(stream_source).waitForCompletion(); }

void GpuStereoProcessor::waitForAllStreams()
{
    l_strm.waitForCompletion();
    r_strm.waitForCompletion();
}

void GpuStereoProcessor::cleanSenders() { senders.clear(); }

void GpuStereoProcessor::setPreFilterType(int filter_type)
{
    block_matcher_gpu_->setPreFilterType(filter_type);
    block_matcher_cpu_->setPreFilterType(filter_type);
}

void GpuStereoProcessor::setRefineDisparity(bool ref_disp) { block_matcher_gpu_->setRefineDisparity(ref_disp); }

void GpuStereoProcessor::setBlockSize(int block_size)
{
    block_matcher_gpu_->setBlockSize(block_size);
    block_matcher_cpu_->setBlockSize(block_size);
}

void GpuStereoProcessor::setNumDisparities(int numDisp)
{
    block_matcher_gpu_->setNumDisparities(numDisp);
    block_matcher_cpu_->setNumDisparities(numDisp);
}

void GpuStereoProcessor::setMinDisparity(int minDisp)
{
    block_matcher_gpu_->setMinDisparity(minDisp);
    block_matcher_cpu_->setMinDisparity(minDisp);
}

void GpuStereoProcessor::setTextureThreshold(int threshold)
{
    block_matcher_gpu_->setTextureThreshold(threshold);
    block_matcher_cpu_->setTextureThreshold(threshold);
}

void GpuStereoProcessor::printStats(std::string name, cv::Mat &mat)
{
    double min, max;

    std::vector<cv::Mat> channels;
    channels.resize(mat.channels());
    cv::split(mat, channels);

    for (int i = 0; i < channels.size(); ++i)
    {
        cv::minMaxLoc(channels[i], &min, &max);
        auto mean_Val = cv::mean(channels[i])[0];
        ROS_INFO("ARRAY STATS:%s; channel:%d; min:%f; max:%f; mean:%f;", name.c_str(), i, min, max, mean_Val);
    }
}

} // namespace
