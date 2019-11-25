#pragma once

#include <visionaray/matrix_camera.h>

struct FrameState
{
  visionaray::matrix_camera camera;

  float delta;
};
