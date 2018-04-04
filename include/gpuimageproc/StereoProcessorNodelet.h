#pragma once

#include <nodelet/nodelet.h>
#include "gpuimageproc/StereoProcessor.h"

namespace gpuimageproc
{

class StereoProcNodelet: public nodelet::Nodelet
{
public:
    void onInit();

protected:
    boost::shared_ptr<StereoProcessor> stereoProcessorPtr;
};

} //nodelet

