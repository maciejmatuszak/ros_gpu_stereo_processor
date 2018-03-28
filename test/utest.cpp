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

std::string GetCurrentWorkingDir(void)
{
    char buff[FILENAME_MAX];
    GetCurrentDir((char *)buff, FILENAME_MAX);
    std::string current_working_dir(buff);
    return current_working_dir;
}
using namespace gpuimageproc;


class CudaStereoBMTf : public testing::Test
{
  protected:
    virtual void SetUp()
    {
        left_image  = readImage("gpuimageproc/test/stereobm/aloe-L.png", cv::IMREAD_GRAYSCALE);
        right_image = readImage("gpuimageproc/test/stereobm/aloe-R.png", cv::IMREAD_GRAYSCALE);
        disp_gold   = readImage("gpuimageproc/test/stereobm/aloe-disp.png", cv::IMREAD_GRAYSCALE);
    }
    virtual void TearDown() {}

    cv::Mat readImage(const std::string &fileName, int flags)
    {
        std::string path = resolvePath(fileName);
        return cv::imread(path, flags);
    }

    inline std::string resolvePath(const std::string &relPath)
    {
        namespace fs = boost::filesystem;
        auto baseDir = fs::current_path();
        while (baseDir.has_parent_path())
        {
            auto combinePath = baseDir / relPath;
            if (fs::exists(combinePath))
            {
                return combinePath.string();
            }
            baseDir = baseDir.parent_path();
        }
        throw std::runtime_error("File not found!");
    }

    cv::Mat left_image;
    cv::Mat right_image;
    cv::Mat disp_gold;

    gpuimageproc::GpuStereoProcessor stereo_processor_;
};

TEST_F(CudaStereoBMTf, HasAccessToData)
{

    ASSERT_FALSE(left_image.empty());
    ASSERT_FALSE(right_image.empty());
    ASSERT_FALSE(disp_gold.empty());

    // EXPECT_MAT_NEAR(disp_gold, disp, 0.0);
}

TEST_F(CudaStereoBMTf, GpuTransfer)
{
    stereo_processor_.uploadMat(GPU_MAT_SRC_L_RECT_MONO, left_image);
    stereo_processor_.waitForStream(GPU_MAT_SRC_L_RECT_MONO);
    cv::Mat left_image2;
    stereo_processor_.downloadMat(GPU_MAT_SRC_L_RECT_MONO, left_image2);
    int nz_cnt = cv::countNonZero(left_image != left_image2);
    ASSERT_EQ(0, nz_cnt);
    // EXPECT_MAT_NEAR(disp_gold, disp, 0.0);
}


TEST_F(CudaStereoBMTf, GpuDisparity)
{
    stereo_processor_.uploadMat(GPU_MAT_SRC_L_RECT_MONO, left_image);
    stereo_processor_.uploadMat(GPU_MAT_SRC_R_RECT_MONO, right_image);
    stereo_processor_.computeDisparity(GPU_MAT_SRC_L_RECT_MONO, GPU_MAT_SRC_R_RECT_MONO, GPU_MAT_SRC_L_DISPARITY);
    stereo_processor_.waitForStream(GPU_MAT_SRC_L_RECT_MONO);
    cv::Mat disparity_img;
    stereo_processor_.downloadMat(GPU_MAT_SRC_L_DISPARITY, disparity_img);
    int nz_cnt = cv::countNonZero(disparity_img != disp_gold);
    ASSERT_EQ(0, nz_cnt);
    // EXPECT_MAT_NEAR(disp_gold, disp, 0.0);
}


int main(int argc, char **argv)
{
    int i;
    printf("argc: %d\n", argc);
    for (i = 0; i < argc; i++)
    {
        printf("argv[%d]: %s\n", i, argv[i]);
    }
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
