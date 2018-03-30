#pragma once

#include <nodelet/nodelet.h>
#include "gpuimageproc/stereoproc.h"

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

