#pragma once

#include "gpuimageproc/StereoProcessor.h"
#include <nodelet/nodelet.h>

namespace gpuimageproc
{

class StereoProcNodelet : public nodelet::Nodelet
{
  public:
    void onInit();

  protected:
    boost::shared_ptr<StereoProcessor> stereoProcessorPtr;
};

} // nodelet
