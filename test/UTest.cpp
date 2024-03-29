#include <boost/filesystem.hpp>
#include <cv_bridge/cv_bridge.h>
#include <gpuimageproc/GPUStereoProcessor.h>
#include <gtest/gtest.h>
#include <image_geometry/pinhole_camera_model.h>
#include <opencv2/core.hpp>
#include <boost/timer.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <unistd.h>

#include <execinfo.h>
#include <fstream>
#include <iostream>
#include <signal.h>

boost::filesystem::path test_data_path = boost::filesystem::path("/data/git/temp_ws/src/gpuimageproc/test");

using namespace gpuimageproc;

class CudaStereoBMTf : public testing::Test
{
  public:
    const uint8_t BLUE  = 10;
    const uint8_t GREEN = 20;
    const uint8_t RED   = 30;
    const uint8_t GRAY  = 22;

  protected:
    virtual void SetUp()
    {
        testImgColorBgr.create(1, 1, CV_8UC3);
        testImgColorBgr.data[0] = BLUE;
        testImgColorBgr.data[1] = GREEN;
        testImgColorBgr.data[2] = RED;

        testImgColorRgb.create(1, 1, CV_8UC3);
        testImgColorRgb.data[0] = RED;
        testImgColorRgb.data[1] = GREEN;
        testImgColorRgb.data[2] = BLUE;

        testImgMono.create(1, 1, CV_8UC1);
        testImgMono.data[0] = GRAY;
    }
    virtual void TearDown() {}

    cv::Mat readImage(const std::string &fileName, int flags)
    {
        std::string path = (test_data_path / fileName).string();
        return cv::imread(path, flags);
    }

    void writeCSV(std::string filename, cv::Mat &m)
    {
        std::ofstream myfile(filename);
        myfile << cv::format(m, cv::Formatter::FMT_CSV);
        myfile.close();
    }

    void writeMAT(std::string filename, cv::Mat &m)
    {
        std::ofstream myfile(filename);
        myfile << cv::format(m, cv::Formatter::FMT_MATLAB);
        myfile.close();
    }

    void initStereoModelKitchen()
    {
        auto left_file  = (test_data_path / "stereobm/test_data/left.yaml").string();
        auto right_file = (test_data_path / "stereobm/test_data/right.yaml").string();
        stereo_processor_.initStereoModel(left_file, right_file);
        stereo_processor_.setPreFilterType(cv::cuda::StereoBM::PREFILTER_XSOBEL);
        stereo_processor_.setMaxSpeckleDiff(200);
        stereo_processor_.setMaxSpeckleSize(200);
    }

    void configureProcessor()
    {
        stereo_processor_.setBlockSize(11);
        stereo_processor_.setMaxSpeckleSize(0);
        stereo_processor_.setMinDisparity(0);
        stereo_processor_.setNumDisparities(128);
        stereo_processor_.setPreFilterType(0);
        stereo_processor_.setRefineDisparity(0);
        stereo_processor_.setTextureThreshold(0);
    }

    void loadImagesKitchen()
    {
        l_raw_  = readImage("stereobm/test_data/left-0022.png", cv::IMREAD_GRAYSCALE);
        r_raw_  = readImage("stereobm/test_data/right-0022.png", cv::IMREAD_GRAYSCALE);
        l_rect_ = readImage("stereobm/test_data/left-0022_rect.png", cv::IMREAD_GRAYSCALE);
        r_rect_ = readImage("stereobm/test_data/right-0022_rect.png", cv::IMREAD_GRAYSCALE);
    }

    void loadImagesAloe()
    {
        l_raw_     = readImage("stereobm/test_data/aloe-L.png", cv::IMREAD_UNCHANGED);
        r_raw_     = readImage("stereobm/test_data/aloe-R.png", cv::IMREAD_UNCHANGED);
        disparity_ = readImage("stereobm/test_data/aloe-disp.png", cv::IMREAD_UNCHANGED);
    }

    bool mat_are_same(cv::Mat &m1, cv::Mat &m2)
    {
        if (m1.size() != m2.size())
            return false;
        if (m1.type() != m2.type())
            return false;
        int nz_cnt = cv::countNonZero(m1 != m2);
        if (nz_cnt > 0)
        {
            return false;
        }
        return true;
    }

    /**
     * @brief mat_are_similar calculate average error per pixel and compare it to threshold
     * @param m1
     * @param m2
     * @param error_threshold per pixel
     * @return
     */
    bool mat_are_similar(cv::Mat &m1, cv::Mat &m2, double error_threshold)
    {
        if (m1.size() != m2.size())
            return false;
        if (m1.type() != m2.type())
            return false;
        auto mean = cv::mean(m1)[0];

        cv::Mat errors;
        cv::subtract(m1, m2, errors);

        //        cv::Mat errors_n;
        //        cv::normalize(errors, errors_n, 0, 255, cv::NORM_MINMAX);
        //        cv::imshow("normalized errors", errors_n);
        //        cv::waitKey(0);

        auto sum_of_errors = cv::sum(errors)[0];
        sum_of_errors      = sum_of_errors / (m1.rows * m1.cols);

        if (sum_of_errors > error_threshold)
        {
            return false;
        }
        return true;
    }

    void createStereoWithEpipolar(cv::Mat &left, cv::Mat &right, const std::string &path_to_save)
    {
        cv::Mat img_to_show;

        cv::hconcat(left, right, img_to_show);

        //      cv::resize(img_to_show, img_to_show,
        //                 cv::Size(VGA_WIDTH*2, VGA_HEIGHT),
        //                 (0, 0), (0, 0), cv::INTER_LINEAR);

        // draw epipolar lines to visualize rectification
        for (int j = 0; j < img_to_show.rows; j += 24)
        {
            line(img_to_show, cv::Point(0, j), cv::Point(img_to_show.cols, j), cv::Scalar(255, 0, 0, 255), 1, 8);
        }

        // cv::imshow("Rectified Stereo Imgs with epipolar lines", img_to_show);
        cv::imwrite(path_to_save, img_to_show);
    }

    gpuimageproc::GpuStereoProcessor stereo_processor_;
    cv::Mat testImgColorBgr, testImgColorRgb, testImgMono;

    cv::Mat l_raw_, r_raw_;
    cv::Mat l_rect_, r_rect_;
    cv::Mat disparity_;
};

TEST_F(CudaStereoBMTf, GpuTransfer)
{
    loadImagesAloe();
    stereo_processor_.uploadMat(GPU_MAT_SRC_L_RECT_MONO, l_raw_);
    stereo_processor_.waitForStream(GPU_MAT_SRC_L_RECT_MONO);
    cv::Mat left_image2;
    stereo_processor_.downloadMat(GPU_MAT_SRC_L_RECT_MONO, left_image2);

    ASSERT_TRUE(mat_are_same(l_raw_, left_image2));
}

TEST_F(CudaStereoBMTf, GpuColorConversionColorMono)
{
    // upload color image
    stereo_processor_.uploadMat(GPU_MAT_SRC_L_RAW, testImgColorBgr, sensor_msgs::image_encodings::BGR8);
    stereo_processor_.convertRawToColor(GPU_MAT_SIDE_L);
    stereo_processor_.convertRawToMono(GPU_MAT_SIDE_L);
    stereo_processor_.waitForStream(GPU_MAT_SIDE_L);

    cv::Mat color, mono;
    stereo_processor_.downloadMat(GPU_MAT_SRC_L_COLOR, color);
    stereo_processor_.downloadMat(GPU_MAT_SRC_L_MONO, mono);

    ASSERT_EQ(CudaStereoBMTf::GRAY, mono.data[0]);

    ASSERT_EQ(CudaStereoBMTf::BLUE, color.data[0]);
    ASSERT_EQ(CudaStereoBMTf::GREEN, color.data[1]);
    ASSERT_EQ(CudaStereoBMTf::RED, color.data[2]);
}

TEST_F(CudaStereoBMTf, GpuColorConversionColorMono2)
{
    // upload color image
    stereo_processor_.uploadMat(GPU_MAT_SRC_L_RAW, testImgColorRgb, sensor_msgs::image_encodings::RGB8);
    stereo_processor_.convertRawToColor(GPU_MAT_SIDE_L);
    stereo_processor_.convertRawToMono(GPU_MAT_SIDE_L);
    stereo_processor_.waitForStream(GPU_MAT_SIDE_L);

    cv::Mat color, mono;
    stereo_processor_.downloadMat(GPU_MAT_SRC_L_COLOR, color);
    stereo_processor_.downloadMat(GPU_MAT_SRC_L_MONO, mono);

    ASSERT_EQ(CudaStereoBMTf::GRAY, mono.data[0]);

    ASSERT_EQ(CudaStereoBMTf::BLUE, color.data[0]);
    ASSERT_EQ(CudaStereoBMTf::GREEN, color.data[1]);
    ASSERT_EQ(CudaStereoBMTf::RED, color.data[2]);
}

TEST_F(CudaStereoBMTf, GpuColorConversionMonoColor)
{

    stereo_processor_.uploadMat(GPU_MAT_SRC_L_RAW, testImgMono, sensor_msgs::image_encodings::MONO8);
    stereo_processor_.convertRawToColor(GPU_MAT_SIDE_L);
    stereo_processor_.convertRawToMono(GPU_MAT_SIDE_L);
    stereo_processor_.waitForStream(GPU_MAT_SIDE_L);

    cv::Mat color, mono;
    stereo_processor_.downloadMat(GPU_MAT_SRC_L_COLOR, color);
    stereo_processor_.downloadMat(GPU_MAT_SRC_L_MONO, mono);

    ASSERT_EQ(CudaStereoBMTf::GRAY, mono.data[0]);

    ASSERT_EQ(CudaStereoBMTf::GRAY, color.data[0]);
    ASSERT_EQ(CudaStereoBMTf::GRAY, color.data[1]);
    ASSERT_EQ(CudaStereoBMTf::GRAY, color.data[2]);
}

TEST_F(CudaStereoBMTf, RectifyMonoCpu)
{
    cv::Mat l_rect, r_rect;

    loadImagesKitchen();
    initStereoModelKitchen();
    stereo_processor_.rectifyImageLeft(l_raw_, l_rect, cv::INTER_LINEAR);
    stereo_processor_.rectifyImageRight(r_raw_, r_rect, cv::INTER_LINEAR);
    ASSERT_TRUE(mat_are_same(l_rect, l_rect_));
    ASSERT_TRUE(mat_are_same(r_rect, r_rect_));
    createStereoWithEpipolar(l_rect, r_rect, (test_data_path / "stereobm/RectifyMonoCpu_rect_.png").string());
    cv::imwrite((test_data_path / "stereobm/RectifyMonoCpu_l_rect.png").string(), l_rect);
    cv::imwrite((test_data_path / "stereobm/RectifyMonoCpu_r_rect.png").string(), r_rect);
}

TEST_F(CudaStereoBMTf, RectifyMonoGpu)
{
    cv::Mat l_rect, r_rect;

    loadImagesKitchen();
    initStereoModelKitchen();
    stereo_processor_.uploadMat(GPU_MAT_SRC_L_RAW, l_raw_);
    stereo_processor_.uploadMat(GPU_MAT_SRC_R_RAW, r_raw_);

    stereo_processor_.rectifyImage(GPU_MAT_SRC_L_RAW, GPU_MAT_SRC_L_RECT_MONO, cv::INTER_LINEAR);
    stereo_processor_.rectifyImage(GPU_MAT_SRC_R_RAW, GPU_MAT_SRC_R_RECT_MONO, cv::INTER_LINEAR);

    stereo_processor_.downloadMat(GPU_MAT_SRC_L_RECT_MONO, l_rect);
    stereo_processor_.downloadMat(GPU_MAT_SRC_R_RECT_MONO, r_rect);

    createStereoWithEpipolar(l_rect, r_rect, (test_data_path / "stereobm/RectifyMonoGpu_rect_.png").string());
    cv::imwrite((test_data_path / "stereobm/RectifyMonoGpu_l_rect.png").string(), l_rect);
    cv::imwrite((test_data_path / "stereobm/RectifyMonoGpu_r_rect.png").string(), r_rect);
    //    cv::Mat l_show, r_show;
    //    cv::hconcat(l_rect,l_rect_, l_show);
    //    cv::imshow("Left", l_show);
    //    cv::hconcat(r_rect,r_rect_, r_show);
    //    cv::imshow("Right", r_show);
    //    cv::waitKey(0);
    ASSERT_TRUE(mat_are_similar(l_rect, l_rect_, 0.1));
    ASSERT_TRUE(mat_are_similar(r_rect, r_rect_, 0.1));
}

TEST_F(CudaStereoBMTf, DisparityGpu)
{
    loadImagesKitchen();
    initStereoModelKitchen();
    stereo_processor_.uploadMat(GPU_MAT_SRC_L_RAW, l_raw_);
    stereo_processor_.uploadMat(GPU_MAT_SRC_R_RAW, r_raw_);

    stereo_processor_.rectifyImage(GPU_MAT_SRC_L_RAW, GPU_MAT_SRC_L_RECT_MONO, cv::INTER_LINEAR);
    stereo_processor_.rectifyImage(GPU_MAT_SRC_R_RAW, GPU_MAT_SRC_R_RECT_MONO, cv::INTER_LINEAR);
    cv::Mat l_rect_cpu, r_rect_cpu;
    cv::Mat l_rect_gpu, r_rect_gpu, disparity_cpu, disparity_gpu_cv_8u;
    stereo_processor_.computeDisparity(GPU_MAT_SRC_L_RECT_MONO, GPU_MAT_SRC_R_RECT_MONO, GPU_MAT_SRC_L_DISPARITY);
    std_msgs::Header mh;
    auto img    = cv_bridge::CvImage(mh, sensor_msgs::image_encodings::MONO8, l_raw_).toImageMsg();
    auto sender = stereo_processor_.enqueueSendDisparity(GPU_MAT_SRC_L_DISPARITY_32F, img, (ros::Publisher *)NULL);

    stereo_processor_.downloadMat(GPU_MAT_SRC_L_RECT_MONO, l_rect_gpu);
    stereo_processor_.downloadMat(GPU_MAT_SRC_R_RECT_MONO, r_rect_gpu);
    stereo_processor_.downloadMat(GPU_MAT_SRC_L_DISPARITY, disparity_gpu_cv_8u);

    stereo_processor_.computeDisparity(l_rect_gpu, r_rect_gpu, disparity_cpu);

    stereo_processor_.waitForAllStreams();

    stereo_processor_.rectifyImageLeft(l_raw_, l_rect_cpu, cv::INTER_LINEAR);
    stereo_processor_.rectifyImageRight(r_raw_, r_rect_cpu, cv::INTER_LINEAR);

    createStereoWithEpipolar(l_rect_gpu, r_rect_gpu, (test_data_path / "stereobm/DisparityGpu_rect_gpu.png").string());
    createStereoWithEpipolar(l_rect_cpu, r_rect_cpu, (test_data_path / "stereobm/DisparityGpu_rect_cpu.png").string());
    cv::imwrite((test_data_path / "stereobm/DisparityGpu_l_rect_cpu.png").string(), l_rect_cpu);
    cv::imwrite((test_data_path / "stereobm/DisparityGpu_r_rect_cpu.png").string(), r_rect_cpu);
    cv::imwrite((test_data_path / "stereobm/DisparityGpu_l_rect_gpu.png").string(), l_rect_gpu);
    cv::imwrite((test_data_path / "stereobm/DisparityGpu_r_rect_gpu.png").string(), r_rect_gpu);
    assert(sender->wasDataSent());
    auto hMemImage   = sender->getDisparityHostMem();
    cv::Mat disp_gpu = hMemImage->createMatHeader();
    cv::imwrite((test_data_path / "stereobm/DisparityGpu_l_disp_gpu.png").string(), disp_gpu);
    cv::imwrite((test_data_path / "stereobm/DisparityGpu_l_disp_gpu_8u.png").string(), disparity_gpu_cv_8u);
    cv::imwrite((test_data_path / "stereobm/DisparityGpu_l_disp_cpu.png").string(), disparity_cpu);
    // ASSERT_TRUE(mat_are_same(l_rect, l_rect_));
    // ASSERT_TRUE(mat_are_same(r_rect, r_rect_));
}

TEST_F(CudaStereoBMTf, ExportDisparitiesToCSV)
{
    loadImagesKitchen();
    initStereoModelKitchen();
    configureProcessor();

    cv::Mat disparity_cpu, disparity_gpu;
    cv::Mat l_rect_small, r_rect_small;
    cv::Size dst_size = l_rect_.size();
    double scale      = 200.0 / dst_size.width;
    dst_size.width    = 200;
    dst_size.height *= scale;

    cv::resize(l_rect_, l_rect_small, dst_size, 0, 0, cv::INTER_AREA);
    cv::resize(r_rect_, r_rect_small, dst_size, 0, 0, cv::INTER_AREA);
    cv::imwrite((test_data_path / "stereobm/rect_l_small.png").string(), l_rect_small);
    cv::imwrite((test_data_path / "stereobm/rect_r_small.png").string(), r_rect_small);

    stereo_processor_.uploadMat(GPU_MAT_SRC_L_RECT_MONO, l_rect_small);
    stereo_processor_.uploadMat(GPU_MAT_SRC_R_RECT_MONO, r_rect_small);

    stereo_processor_.computeDisparity(GPU_MAT_SRC_L_RECT_MONO, GPU_MAT_SRC_R_RECT_MONO, GPU_MAT_SRC_L_DISPARITY);
    stereo_processor_.computeDisparity(l_rect_small, r_rect_small, disparity_cpu);

    stereo_processor_.downloadMat(GPU_MAT_SRC_L_DISPARITY, disparity_gpu);

    writeCSV((test_data_path / "stereobm/disparity_gpu.csv").string(), disparity_gpu);
    writeCSV((test_data_path / "stereobm/disparity_cpu.csv").string(), disparity_cpu);
    writeMAT((test_data_path / "stereobm/disparity_gpu.mat").string(), disparity_gpu);
    writeMAT((test_data_path / "stereobm/disparity_cpu.mat").string(), disparity_cpu);
}

TEST_F(CudaStereoBMTf, PointCloud)
{
    loadImagesKitchen();
    initStereoModelKitchen();
    stereo_processor_.uploadMat(GPU_MAT_SRC_L_RAW, l_raw_, sensor_msgs::image_encodings::MONO8);
    stereo_processor_.uploadMat(GPU_MAT_SRC_R_RAW, r_raw_, sensor_msgs::image_encodings::MONO8);
    stereo_processor_.convertRawToColor(GPU_MAT_SIDE_L);

    stereo_processor_.rectifyImage(GPU_MAT_SRC_L_COLOR, GPU_MAT_SRC_L_RECT_COLOR, cv::INTER_LINEAR);
    stereo_processor_.rectifyImage(GPU_MAT_SRC_L_RAW, GPU_MAT_SRC_L_RECT_MONO, cv::INTER_LINEAR);
    stereo_processor_.rectifyImage(GPU_MAT_SRC_R_RAW, GPU_MAT_SRC_R_RECT_MONO, cv::INTER_LINEAR);

    stereo_processor_.computeDisparity(GPU_MAT_SRC_L_RECT_MONO, GPU_MAT_SRC_R_RECT_MONO, GPU_MAT_SRC_L_DISPARITY);
    stereo_processor_.projectDisparityTo3DPoints(GPU_MAT_SRC_L_DISPARITY_32F, GPU_MAT_SRC_L_POINTS2);

    std_msgs::Header mh;
    auto img    = cv_bridge::CvImage(mh, sensor_msgs::image_encodings::MONO8, l_raw_).toImageMsg();
    auto sender = stereo_processor_.enqueueSendPoints(GPU_MAT_SRC_L_POINTS2, GPU_MAT_SRC_L_RECT_COLOR, img, (ros::Publisher *)NULL);

    cv::Mat l_rect_gpu, r_rect_gpu, l_disparity_8U, l_disparity_32F, l_points;

    stereo_processor_.downloadMat(GPU_MAT_SRC_L_RECT_MONO, l_rect_gpu);
    stereo_processor_.downloadMat(GPU_MAT_SRC_R_RECT_MONO, r_rect_gpu);
    stereo_processor_.downloadMat(GPU_MAT_SRC_L_DISPARITY, l_disparity_8U);
    stereo_processor_.downloadMat(GPU_MAT_SRC_L_DISPARITY_32F, l_disparity_32F);
    stereo_processor_.downloadMat(GPU_MAT_SRC_L_POINTS2, l_points);
    const cv::Mat_<cv::Vec3f> points(l_points);

    stereo_processor_.waitForAllStreams();
    auto point_msgPtr = sender->getPointMessage();

    // cv::imshow("Disparity", l_disparity_8U);
    // cvWaitKey(0);
}

TEST_F(CudaStereoBMTf, DisparityTiming)
{
    loadImagesKitchen();
    initStereoModelKitchen();
    cv::cuda::HostMem lHM(cv::cuda::HostMem::SHARED);
    cv::cuda::HostMem rHM(cv::cuda::HostMem::SHARED);
    cv::cuda::HostMem dHM(cv::cuda::HostMem::SHARED);
    cv::cuda::GpuMat lGpuMat, rGpuMat, dGpuMat;

    lHM.create(l_rect_.rows, l_rect_.cols, l_rect_.type());
    l_rect_.copyTo(lHM.createMatHeader());
    rHM.create(r_rect_.rows, r_rect_.cols, r_rect_.type());
    r_rect_.copyTo(rHM.createMatHeader());
    rHM.create(l_rect_.rows, l_rect_.cols, r_rect_.type());
    dHM.create(l_rect_.rows, l_rect_.cols, CV_8UC1);

    lGpuMat.upload(l_rect_);
    rGpuMat.upload(r_rect_);
    dGpuMat.create(l_rect_.rows, l_rect_.cols,CV_8UC1);

    boost::timer perf_timer;
    stereo_processor_.computeDisparityBare(lHM,rHM,dHM);
    double dur_HMem = perf_timer.elapsed();
    perf_timer.restart();
    stereo_processor_.computeDisparityBare(lGpuMat,rGpuMat,dGpuMat);
    double dur_GpuMat = perf_timer.elapsed();
    std::cout << "compute disparity on HostMem:" << (dur_HMem * 1000.0) << std::endl;
    std::cout << "compute disparity on GpuMat:" << (dur_GpuMat * 1000.0) << std::endl;

}

void handler(int sig)
{
    void *array[10];
    size_t size;

    // get void*'s for all entries on the stack
    size = backtrace(array, 10);

    // print out all the frames to stderr
    fprintf(stderr, "Error: signal %d:\n", sig);
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    exit(1);
}

int main(int argc, char **argv)
{
    signal(SIGSEGV, handler); // install our handler
    signal(SIGFPE, handler);  // install our handler
    if (argc < 2)
    {
        printf("test data path is required!\n");
        return -1;
    }
    auto p = boost::filesystem::path(argv[1]);
    if (boost::filesystem::exists(p))
    {
        test_data_path = p;
    }
    printf("test data path:%s\n", test_data_path.c_str());

    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
