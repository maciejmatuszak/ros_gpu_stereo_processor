#include "gpuimageproc/GPUStereoProcessor.h"
#include "gpuimageproc/GpuSenderDisparity.h"
#include "gpuimageproc/GpuSenderImage.h"
#include "gpuimageproc/GpuSenderPc2.h"
#include <boost/timer.hpp>
#include <camera_calibration_parsers/parse.h>
#include <cv_bridge/cv_bridge.h>

namespace gpuimageproc
{

GpuStereoProcessor::GpuStereoProcessor()
    : l_raw_encoding_("")
    , r_raw_encoding_("")

{
    cv::cuda::printShortCudaDeviceInfo(cv::cuda::getDevice());
    block_matcher_gpu_ = cv::cuda::createStereoBM(48, 19);
    block_matcher_gpu_->setRefineDisparity(false);
    block_matcher_cpu_ = cv::StereoBM::create(48, 19);

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
    auto gmat = getHostMem(mat_source);
    gmat->create(cv_mat.rows, cv_mat.cols, cv_mat.type());

    cv_mat.copyTo(gmat->createMatHeader());
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
    auto hMem = getHostMem(mat_source);
    cv::cuda::createContinuous(hMem->size(), hMem->type(), cv_mat);
    getStream(mat_source).waitForCompletion();
    hMem->createMatHeader().copyTo(cv_mat);
}

void GpuStereoProcessor::convertColor(GpuMatSource mat_source, GpuMatSource mat_dst, const std::string &src_encoding, const std::string &dst_encoding)
{
    assert(!src_encoding.empty());
    assert(!dst_encoding.empty());
    // Copy metadata
    auto srcMat    = getHostMem(mat_source);
    auto dstMat    = getHostMem(mat_dst);
    auto srcStream = getStream(mat_source);

    // Copy to new buffer if same encoding requested
    if (dst_encoding.empty() || dst_encoding == src_encoding)
    {
        dstMat->create(srcMat->rows, srcMat->cols, srcMat->type());
        srcMat->createMatHeader().copyTo(dstMat->createMatHeader());
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

            dstMat->create(srcMat->rows, srcMat->cols, image2_type);
            // Do scaling between CV_8U [0,255] and CV_16U [0,65535] images.
            if (src_depth == 8 && dst_depth == 16)
                srcMat->createGpuMatHeader().convertTo(dstMat->createGpuMatHeader(), image2_type, 65535. / 255., srcStream);
            else if (src_depth == 16 && dst_depth == 8)
                srcMat->createGpuMatHeader().convertTo(dstMat->createGpuMatHeader(), image2_type, 255. / 65535., srcStream);
            else
                srcMat->createGpuMatHeader().convertTo(dstMat->createGpuMatHeader(), image2_type, srcStream);
        }
        else
        {
            // Perform color conversion
            int dcn         = sensor_msgs::image_encodings::numChannels(dst_encoding);
            int image2_type = cv_bridge::getCvType(dst_encoding);

            dstMat->create(srcMat->rows, srcMat->cols, image2_type);
            cv::cuda::cvtColor(*srcMat, *dstMat, conversion_code, dcn, srcStream);
        }
        // image1 = image2;
        //}
    }
}

boost::shared_ptr<cv::cuda::HostMem> GpuStereoProcessor::getHostMem(GpuMatSource source)
{
    boost::shared_ptr<cv::cuda::HostMem> hMem;
    auto str_source = std::to_string(source);
    if (gpu_mats.find(str_source) == gpu_mats.end())
    {
        hMem                 = boost::make_shared<cv::cuda::HostMem>(cv::cuda::HostMem::SHARED);
        gpu_mats[str_source] = hMem;
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

double GpuStereoProcessor::getMaxSpeckleDiff() const { return maxSpeckleDiff_; }

void GpuStereoProcessor::setMaxSpeckleDiff(double maxSpeckleDiff) { maxSpeckleDiff_ = maxSpeckleDiff; }

int GpuStereoProcessor::getMaxSpeckleSize() const { return maxSpeckleSize_; }

void GpuStereoProcessor::setMaxSpeckleSize(int maxSpeckleSize) { maxSpeckleSize_ = maxSpeckleSize; }

GPUSenderImagePtr GpuStereoProcessor::enqueueSendImage(GpuMatSource source, const sensor_msgs::ImageConstPtr &imagePattern, std::string encoding, ros::Publisher *pub)
{
    GPUSenderImagePtr t = boost::make_shared<GPUSenderImage>(&imagePattern->header, pub, getHostMem(source), encoding);
    senders.push_back(t);
    t->enqueueSend(getStream(source));
    return t;
}

GPUSenderDisparityPtr GpuStereoProcessor::enqueueSendDisparity(GpuMatSource source, const sensor_msgs::ImageConstPtr &imagePattern, ros::Publisher *pub)
{
    GPUSenderDisparityPtr t =
        boost::make_shared<GPUSenderDisparity>(&imagePattern->header, pub, getHostMem(source), block_matcher_gpu_->getBlockSize(), block_matcher_gpu_->getNumDisparities(),
                                               model_.right().fx(), model_.baseline(), block_matcher_gpu_->getMinDisparity());
    senders.push_back(t);
    t->enqueueSend(getStream(source));
    return t;
}

GPUSenderPc2Ptr GpuStereoProcessor::enqueueSendPoints(GpuMatSource points_source, GpuMatSource color_source, const sensor_msgs::ImageConstPtr &imagePattern, ros::Publisher *pub)
{
    GPUSenderPc2Ptr t = boost::make_shared<GPUSenderPc2>(&imagePattern->header, pub, getHostMem(points_source), getHostMem(color_source));
    senders.push_back(t);
    t->enqueueSend(getStream(points_source));
    return t;
}

void GpuStereoProcessor::rectifyImage(GpuMatSource source, GpuMatSource dest, cv::InterpolationFlags interpolation)
{
    assert(model_.initialized());
    auto srcHm = getHostMem(source);
    auto dstHm = getHostMem(dest);
    dstHm->create(srcHm->rows, srcHm->cols, srcHm->type());
    if ((source & GPU_MAT_SIDE_L) == GPU_MAT_SIDE_L)
    {
        model_.left().rectifyImageGPU(*srcHm, *dstHm, interpolation, getStream(source));
    }
    else
    {
        model_.right().rectifyImageGPU(*srcHm, *dstHm, interpolation, getStream(source));
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
    boost::timer perf_timer;
    assert(model_.initialized());
    // Fixed-point disparity is 16 times the true value: d = d_fp / 16.0 = x_l - x_r.
    static const int DPP        = 16; // disparities per pixel
    static const double inv_dpp = 1.0 / DPP;

    // Block matcher produces 16-bit signed (fixed point) disparity image
    auto lgpu    = getHostMem(left);
    auto rgpu    = getHostMem(right);
    auto dgpu    = getHostMem(disparity);
    double shift = -(model_.left().cx() - model_.right().cx());

    GpuMatSource disparity_f32 = static_cast<GpuMatSource>((disparity & GPU_MAT_SIDE_MASK) | GPU_MAT_SRC_DISPARITY_32F);
    auto dgpu32                = getHostMem(disparity_f32);

    double dur_prep = perf_timer.elapsed() * 1000.0;
    perf_timer.restart();
    block_matcher_gpu_->compute(*lgpu, *rgpu, *dgpu, getStream(disparity));
    double dur_disparity = perf_timer.elapsed() * 1000.0;
    perf_timer.restart();
    //         side left or right \___________________________/    ^---plus disparity 32F selector
    // TODO:the x offset will be different for right side for now just use assert to prevent the use of Right disparity
    assert((disparity & GPU_MAT_SIDE_MASK) == GPU_MAT_SIDE_L);

    //    cv::Mat disp;
    //    dgpu->download(disp);
    //    printStats("Disparity clean", disp);
    // dgpu32->create(dgpu->size(), CV_32FC1);
    // dgpu->createGpuMatHeader().convertTo(dgpu32->createGpuMatHeader(), CV_32FC1, inv_dpp, shift);
    // dgpu->convertTo(*dgpu32, CV_32FC1, 1      , shift);
    double dur_convertToCV_32FC1 = perf_timer.elapsed() * 1000.0;
    perf_timer.restart();
    //    ROS_DEBUG("computeDisparity; prep:%.3f;  disparity:%.3f;  convert to CV_32FC1:%.3f; TOTAL:%.2f ", dur_prep, dur_disparity, dur_convertToCV_32FC1,
    //             (dur_prep + dur_disparity + dur_convertToCV_32FC1));

    //    dgpu32->download(disp);
    //    printStats("Disparity scaled", disp);
}

void GpuStereoProcessor::computeDisparityBare(cv::InputArray left, cv::InputArray right, cv::OutputArray disparity)
{
    assert(model_.initialized());

    block_matcher_gpu_->compute(left, right, disparity);
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

void GpuStereoProcessor::computeDisparityImage(GpuMatSource disparity_src, GpuMatSource disp_image_dest)
{
    assert(model_.initialized());
    auto disparity = getHostMem(disparity_src);
    auto image     = getHostMem(disp_image_dest);
    auto ndisp     = block_matcher_gpu_->getNumDisparities();
    cv::cuda::drawColorDisp(*disparity, *image, ndisp, getStream(disparity_src));
}

void GpuStereoProcessor::projectDisparityTo3DPoints(GpuMatSource disparity_src, GpuMatSource points_src)
{

    assert(model_.initialized());

    auto disp_gpu       = getHostMem(disparity_src);
    auto points_src_gpu = getHostMem(points_src);
    auto strm           = getStream(disparity_src);

    model_.projectDisparityImageTo3dGPU(*disp_gpu,       /* disparity gpu mat */
                                        *points_src_gpu, /* points gpu mat */
                                        true,            /* handle missing points */
                                        strm             /* gpu stream */
                                        );
}

void GpuStereoProcessor::waitForStream(GpuMatSource stream_source) { getStream(stream_source).waitForCompletion(); }

void GpuStereoProcessor::waitForAllStreams()
{
    l_strm.waitForCompletion();
    r_strm.waitForCompletion();
}

void GpuStereoProcessor::filterSpeckles(GpuMatSource disparity_src)
{
    if (maxSpeckleSize_ > 0)
    {
        getStream(disparity_src).waitForCompletion();
        auto disHmem = getHostMem(disparity_src);

        filterSpeckles(*disHmem);
    }
}

void GpuStereoProcessor::filterSpeckles(cv::InputOutputArray disparity)
{

    // buvffer size calculations based on /data/git/opencv/modules/calib3d/src/stereosgbm.cpp:2291 filterSpecklesImpl function
    int width      = disparity.cols();
    int height     = disparity.rows();
    int npixels    = width * height;
    size_t bufSize = npixels * (int)(sizeof(cv::Point_<short>) + sizeof(int) + sizeof(uchar));
    if (!_speclesBuf.isContinuous() || _speclesBuf.empty() || _speclesBuf.cols * _speclesBuf.rows * _speclesBuf.elemSize() < bufSize)
    {
        _speclesBuf.reserveBuffer(bufSize);
    }

    cv::Mat disp16S;
    cv::Mat temp = disparity.getMat();
    temp.convertTo(disp16S, CV_16SC1);
    cv::filterSpeckles(disp16S, 0, maxSpeckleSize_, maxSpeckleDiff_, _speclesBuf);
    disp16S.convertTo(temp, CV_8UC1);
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
        ROS_DEBUG("ARRAY STATS:%s; channel:%d; min:%f; max:%f; mean:%f;", name.c_str(), i, min, max, mean_Val);
    }
}

} // namespace
