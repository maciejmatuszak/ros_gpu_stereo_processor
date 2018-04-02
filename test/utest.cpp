#include "image_geometry/pinhole_camera_model.h"
#include <boost/filesystem.hpp>
#include <gpuimageproc/gpustereoprocessor.h>
#include <gtest/gtest.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cv_bridge/cv_bridge.h>
#include <stdio.h>
#include <unistd.h>

#include <execinfo.h>
#include <iostream>
#include <signal.h>

boost::filesystem::path test_data_path = boost::filesystem::path("/data/git/temp_ws/src/gpuimageproc/test");

using namespace gpuimageproc;

class CudaStereoBMTf : public testing::Test
{
  protected:
    virtual void SetUp() {}
    virtual void TearDown() {}

    cv::Mat readImage(const std::string &fileName, int flags)
    {
        std::string path = (test_data_path / fileName).string();
        return cv::imread(path, flags);
    }
    void initStereoModelKitchen()
    {
        auto left_file  = (test_data_path / "stereobm/test_data/left.yaml").string();
        auto right_file = (test_data_path / "stereobm/test_data/right.yaml").string();
        stereo_processor_.initStereoModel(left_file, right_file);
    }

    void initStereoModelEuroc_scale_0()
    {
        auto left_file  = (test_data_path / "stereobm/euroc_left_scale_0.yaml").string();
        auto right_file = (test_data_path / "stereobm/euroc_right_scale_0.yaml").string();
        stereo_processor_.initStereoModel(left_file, right_file);
    }

    void initStereoModelEuroc_scale_100()
    {
        auto left_file  = (test_data_path / "stereobm/euroc_left_scale_100.yaml").string();
        auto right_file = (test_data_path / "stereobm/euroc_right_scale_100.yaml").string();
        stereo_processor_.initStereoModel(left_file, right_file);
    }

    void loadImagesLab()
    {
        l_raw_  = readImage("stereobm/lab_left_raw.png", cv::IMREAD_GRAYSCALE);
        r_raw_  = readImage("stereobm/lab_right_raw.png", cv::IMREAD_GRAYSCALE);
        l_rect_ = readImage("stereobm/bag_24271_left_rect.png", cv::IMREAD_GRAYSCALE);
        r_rect_ = readImage("stereobm/bag_24271_right_rect.png", cv::IMREAD_GRAYSCALE);
    }

    void loadImagesKitchen()
    {
        l_raw_ = readImage("stereobm/test_data/left-0022.png", cv::IMREAD_GRAYSCALE);
        r_raw_ = readImage("stereobm/test_data/right-0022.png", cv::IMREAD_GRAYSCALE);
    }

    void loadImagesAloe()
    {
        l_raw_     = readImage("stereobm/aloe-L.png", cv::IMREAD_GRAYSCALE);
        r_raw_     = readImage("stereobm/aloe-R.png", cv::IMREAD_GRAYSCALE);
        disparity_ = readImage("stereobm/aloe-disp.png", cv::IMREAD_GRAYSCALE);
    }

    void loadImagesEuroc()
    {
        l_raw_ = readImage("stereobm/left-0003.png", cv::IMREAD_GRAYSCALE);
        ASSERT_FALSE(l_raw_.empty());
        r_raw_ = readImage("stereobm/right-0003.png", cv::IMREAD_GRAYSCALE);
        ASSERT_FALSE(r_raw_.empty());
        disparity_ = readImage("stereobm/aloe-disp.png", cv::IMREAD_GRAYSCALE);
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

        cv::imshow("Rectified Stereo Imgs with epipolar lines", img_to_show);
        cv::imwrite(path_to_save, img_to_show);
    }

    gpuimageproc::GpuStereoProcessor stereo_processor_;
    cv::Mat l_raw_, r_raw_;
    cv::Mat l_rect_, r_rect_;
    cv::Mat disparity_;
};

// TEST_F(CudaStereoBMTf, GpuTransfer)
//{
//    loadImagesAloe();
//    stereo_processor_.uploadMat(GPU_MAT_SRC_L_RECT_MONO, l_rect_);
//    stereo_processor_.waitForStream(GPU_MAT_SRC_L_RECT_MONO);
//    cv::Mat left_image2;
//    stereo_processor_.downloadMat(GPU_MAT_SRC_L_RECT_MONO, left_image2);

//    ASSERT_TRUE(mat_are_same(l_rect_, left_image2));
//}

// TEST_F(CudaStereoBMTf, GpuDisparity)
//{
//    loadImagesAloe();
//    stereo_processor_.uploadMat(GPU_MAT_SRC_L_RECT_MONO, l_rect_);
//    stereo_processor_.uploadMat(GPU_MAT_SRC_R_RECT_MONO, r_rect_);
//    stereo_processor_.computeDisparity(GPU_MAT_SRC_L_RECT_MONO, GPU_MAT_SRC_R_RECT_MONO, GPU_MAT_SRC_L_DISPARITY);
//    stereo_processor_.waitForStream(GPU_MAT_SRC_L_RECT_MONO);

//    cv::Mat disparity_img;
//    stereo_processor_.downloadMat(GPU_MAT_SRC_L_DISPARITY, disparity_img);
//    ASSERT_TRUE(mat_are_same(disparity_img, disparity_));
//}

// TEST_F(CudaStereoBMTf, RectifyMonoCpu)
//{
//    cv::Mat l_rect, r_rect;

//    loadImagesEuroc();
//    initStereoModelEuroc_scale_100();
//    stereo_processor_.rectifyImageLeft(l_raw_, l_rect, cv::INTER_LINEAR);
//    stereo_processor_.rectifyImageRight(r_raw_, r_rect, cv::INTER_LINEAR);
//    //createStereoWithEpipolar(l_rect, r_rect, (test_data_path / "stereobm/RectifyMonoCpu_rect_.png").string());
//    //cv::imwrite((test_data_path / "stereobm/RectifyMonoCpu_l_rect.png").string(), l_rect);
//    //cv::imwrite((test_data_path / "stereobm/RectifyMonoCpu_r_rect.png").string(), r_rect);

//}

TEST_F(CudaStereoBMTf, RectifyMonoGpu)
{
    loadImagesLab();
    initStereoModelKitchen();
    stereo_processor_.uploadMat(GPU_MAT_SRC_L_RAW, l_raw_);
    stereo_processor_.uploadMat(GPU_MAT_SRC_R_RAW, r_raw_);

    stereo_processor_.rectifyImage(GPU_MAT_SRC_R_RAW, GPU_MAT_SRC_R_RECT_MONO, cv::INTER_LINEAR);
    stereo_processor_.rectifyImage(GPU_MAT_SRC_L_RAW, GPU_MAT_SRC_L_RECT_MONO, cv::INTER_LINEAR);
    cv::Mat l_rect_cpu, r_rect_cpu;
    cv::Mat l_rect_gpu, r_rect_gpu, disparity_gpu, disparity_cpu;
    stereo_processor_.computeDisparity(GPU_MAT_SRC_L_RECT_MONO, GPU_MAT_SRC_R_RECT_MONO, GPU_MAT_SRC_L_DISPARITY);
    std_msgs::Header mh;
    auto img = cv_bridge::CvImage(mh, sensor_msgs::image_encodings::MONO8, l_raw_).toImageMsg();
    stereo_processor_.enqueueSendDisparity(GPU_MAT_SRC_L_DISPARITY, img, (ros::Publisher *)NULL);

    stereo_processor_.downloadMat(GPU_MAT_SRC_L_RECT_MONO, l_rect_gpu);
    stereo_processor_.downloadMat(GPU_MAT_SRC_R_RECT_MONO, r_rect_gpu);
    stereo_processor_.downloadMat(GPU_MAT_SRC_L_DISPARITY, disparity_gpu);
    stereo_processor_.computeDisparity(l_rect_gpu, r_rect_gpu, disparity_cpu);

    stereo_processor_.rectifyImageLeft(l_raw_, l_rect_cpu, cv::INTER_LINEAR);
    stereo_processor_.rectifyImageRight(r_raw_, r_rect_cpu, cv::INTER_LINEAR);

    createStereoWithEpipolar(l_rect_gpu, r_rect_gpu, (test_data_path / "stereobm/RectifyMonoGpu_rect_gpu.png").string());
    createStereoWithEpipolar(l_rect_cpu, r_rect_cpu, (test_data_path / "stereobm/RectifyMonoGpu_rect_cpu.png").string());
    createStereoWithEpipolar(l_rect_gpu, disparity_gpu, (test_data_path / "stereobm/RectifyMonoGpu_rect_disp.png").string());
    cv::imwrite((test_data_path / "stereobm/RectifyMonoGpu_l_rect_cpu.png").string(), l_rect_cpu);
    cv::imwrite((test_data_path / "stereobm/RectifyMonoGpu_r_rect_cpu.png").string(), r_rect_cpu);
    cv::imwrite((test_data_path / "stereobm/RectifyMonoGpu_l_rect_gpu.png").string(), l_rect_gpu);
    cv::imwrite((test_data_path / "stereobm/RectifyMonoGpu_r_rect_gpu.png").string(), r_rect_gpu);
    cv::imwrite((test_data_path / "stereobm/RectifyMonoGpu_l_disp_gpu.png").string(), disparity_gpu);
    cv::imwrite((test_data_path / "stereobm/RectifyMonoGpu_l_disp_cpu.png").string(), disparity_cpu);
    // ASSERT_TRUE(mat_are_same(l_rect, l_rect_));
    // ASSERT_TRUE(mat_are_same(r_rect, r_rect_));
}

// TEST_F(CudaStereoBMTf, RectifyMonoAndDisparity)
//{
//    cv::Mat l_rect, r_rect, disparity;

//    loadImagesLab();
//    initStereoModel();
//    stereo_processor_.uploadMat(GPU_MAT_SRC_L_RAW, l_raw_);
//    stereo_processor_.uploadMat(GPU_MAT_SRC_R_RAW, r_raw_);
//    stereo_processor_.rectifyImageLeft(l_raw_, l_rect, cv::INTER_LINEAR);
//    stereo_processor_.rectifyImageRight(r_raw_, r_rect, cv::INTER_LINEAR);

//    //stereo_processor_.rectifyImage(GPU_MAT_SRC_L_RAW, GPU_MAT_SRC_L_RECT_MONO, cv::INTER_LINEAR);
//    //stereo_processor_.rectifyImage(GPU_MAT_SRC_R_RAW, GPU_MAT_SRC_R_RECT_MONO, cv::INTER_LINEAR);
//    //stereo_processor_.computeDisparity(GPU_MAT_SRC_L_RECT_MONO, GPU_MAT_SRC_R_RECT_MONO, GPU_MAT_SRC_L_DISPARITY);

//    //stereo_processor_.downloadMat(GPU_MAT_SRC_L_RECT_MONO, l_rect);
//    //stereo_processor_.downloadMat(GPU_MAT_SRC_R_RECT_MONO, r_rect);
//    //stereo_processor_.downloadMat(GPU_MAT_SRC_L_DISPARITY, disparity);
//    ASSERT_TRUE(mat_are_same(l_rect, l_rect_));
//    ASSERT_TRUE(mat_are_same(r_rect, r_rect_));
//    cv::imwrite((test_data_path / "stereobm/bag_24271_disparity.png").string(), disparity);
//}

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
