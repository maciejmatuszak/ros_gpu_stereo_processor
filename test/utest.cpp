#include "image_geometry/pinhole_camera_model.h"
#include <boost/filesystem.hpp>
#include <gpuimageproc/gpustereoprocessor.h>
#include <gtest/gtest.h>
#include <opencv2/core/cuda.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <unistd.h>
#define GetCurrentDir getcwd

#include <iostream>

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
    void initStereoModel() { stereo_processor_.initStereoModel((test_data_path / "stereobm/left_info.yaml").string(), (test_data_path / "stereobm/right_info.yaml").string()); }

    void loadImagesLab()
    {
        l_raw_ = readImage("stereobm/lab_left_raw.png", cv::IMREAD_GRAYSCALE);
        r_raw_ = readImage("stereobm/lab_right_raw.png", cv::IMREAD_GRAYSCALE);
        l_rect_ = readImage("stereobm/bag_24271_left_rect.png", cv::IMREAD_GRAYSCALE);
        r_rect_ = readImage("stereobm/bag_24271_right_rect.png", cv::IMREAD_GRAYSCALE);
    }

    void loadImagesAloe()
    {
        l_rect_ = readImage("stereobm/aloe-L.png", cv::IMREAD_GRAYSCALE);
        r_rect_ = readImage("stereobm/aloe-R.png", cv::IMREAD_GRAYSCALE);
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

    void createStereoWithEpipolar(cv::Mat& left, cv::Mat& right, const std::string& path_to_save)
    {
      cv::Mat img_to_show;

      cv::hconcat(left,
                  right,
                  img_to_show);

//      cv::resize(img_to_show, img_to_show,
//                 cv::Size(VGA_WIDTH*2, VGA_HEIGHT),
//                 (0, 0), (0, 0), cv::INTER_LINEAR);

      // draw epipolar lines to visualize rectification
      for(int j = 0; j < img_to_show.rows; j += 24 ){
        line(img_to_show, cv::Point(0, j),
             cv::Point(img_to_show.cols, j),
             cv::Scalar(255, 0, 0, 255), 1, 8);
      }

      cv::imshow("Rectified Stereo Imgs with epipolar lines", img_to_show);
      cv::imwrite(path_to_save, img_to_show);
    }

    gpuimageproc::GpuStereoProcessor stereo_processor_;
    cv::Mat l_raw_, r_raw_;
    cv::Mat l_rect_, r_rect_;
    cv::Mat disparity_;
};

TEST_F(CudaStereoBMTf, GpuTransfer)
{
    loadImagesAloe();
    stereo_processor_.uploadMat(GPU_MAT_SRC_L_RECT_MONO, l_rect_);
    stereo_processor_.waitForStream(GPU_MAT_SRC_L_RECT_MONO);
    cv::Mat left_image2;
    stereo_processor_.downloadMat(GPU_MAT_SRC_L_RECT_MONO, left_image2);

    ASSERT_TRUE(mat_are_same(l_rect_, left_image2));
}

TEST_F(CudaStereoBMTf, GpuDisparity)
{
    loadImagesAloe();
    stereo_processor_.uploadMat(GPU_MAT_SRC_L_RECT_MONO, l_rect_);
    stereo_processor_.uploadMat(GPU_MAT_SRC_R_RECT_MONO, r_rect_);
    stereo_processor_.computeDisparity(GPU_MAT_SRC_L_RECT_MONO, GPU_MAT_SRC_R_RECT_MONO, GPU_MAT_SRC_L_DISPARITY);
    stereo_processor_.waitForStream(GPU_MAT_SRC_L_RECT_MONO);

    cv::Mat disparity_img;
    stereo_processor_.downloadMat(GPU_MAT_SRC_L_DISPARITY, disparity_img);
    ASSERT_TRUE(mat_are_same(disparity_img, disparity_));
}

TEST_F(CudaStereoBMTf, RectifyMono)
{
    loadImagesLab();
    initStereoModel();
    stereo_processor_.uploadMat(GPU_MAT_SRC_L_RAW, l_raw_);
    stereo_processor_.uploadMat(GPU_MAT_SRC_R_RAW, r_raw_);

    stereo_processor_.rectifyImage(GPU_MAT_SRC_L_RAW, GPU_MAT_SRC_L_RECT_MONO, cv::INTER_LINEAR);
    stereo_processor_.rectifyImage(GPU_MAT_SRC_R_RAW, GPU_MAT_SRC_R_RECT_MONO, cv::INTER_LINEAR);
    cv::Mat l_rect, r_rect;
    stereo_processor_.downloadMat(GPU_MAT_SRC_L_RECT_MONO, l_rect);
    stereo_processor_.downloadMat(GPU_MAT_SRC_R_RECT_MONO, r_rect);
    cv::imwrite((test_data_path / "stereobm/lab_left_rect_.png").string(), l_rect);
    cv::imwrite((test_data_path / "stereobm/lab_right_rect_.png").string(), r_rect);
    createStereoWithEpipolar(l_rect, r_rect, (test_data_path / "stereobm/lab_stereo_rect_w_epi.png").string());
    ASSERT_TRUE(mat_are_same(l_rect, l_rect_));
    ASSERT_TRUE(mat_are_same(r_rect, r_rect_));
}

TEST_F(CudaStereoBMTf, RectifyMonoAndDisparity)
{
    loadImagesLab();
    initStereoModel();
    stereo_processor_.uploadMat(GPU_MAT_SRC_L_RAW, l_raw_);
    stereo_processor_.uploadMat(GPU_MAT_SRC_R_RAW, r_raw_);

    stereo_processor_.rectifyImage(GPU_MAT_SRC_L_RAW, GPU_MAT_SRC_L_RECT_MONO, cv::INTER_LINEAR);
    stereo_processor_.rectifyImage(GPU_MAT_SRC_R_RAW, GPU_MAT_SRC_R_RECT_MONO, cv::INTER_LINEAR);
    stereo_processor_.computeDisparity(GPU_MAT_SRC_L_RECT_MONO, GPU_MAT_SRC_R_RECT_MONO, GPU_MAT_SRC_L_DISPARITY);
    cv::Mat l_rect, r_rect, disparity;
    stereo_processor_.downloadMat(GPU_MAT_SRC_L_RECT_MONO, l_rect);
    stereo_processor_.downloadMat(GPU_MAT_SRC_R_RECT_MONO, r_rect);
    stereo_processor_.downloadMat(GPU_MAT_SRC_L_DISPARITY, disparity);
    ASSERT_TRUE(mat_are_same(l_rect, l_rect_));
    ASSERT_TRUE(mat_are_same(r_rect, r_rect_));
    cv::imwrite((test_data_path / "stereobm/bag_24271_disparity.png").string(), disparity);
}

int main(int argc, char **argv)
{
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
