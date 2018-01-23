#pragma once

#include <virvo/math/aabb.h>
#include <virvo/math/forward.h>
#include <virvo/vvexport.h>

class vvVolDesc;

namespace virvo
{
namespace stats
{

  struct VVAPI EntropyRegion
  {
    float entropy;
    aabbi bbox;
  };

  bool operator<(const EntropyRegion& a, const EntropyRegion& b)
  {
    return a.entropy < b.entropy;
  }

  VVAPI void makeHistogram(const vvVolDesc* vd,
      aabbi bbox,
      int chan1,
      int numChan,
      int* numBuckets,
      int* count);

  VVAPI float entropy(const vvVolDesc* vd,
      aabbi bbox,
      int chan1,
      int numChan);

  // Compute into how many regions vd needs to be
  // subdivided given regionSize
  VVAPI vec3i numEntropyRegions(const vvVolDesc* vd,
      vec3i regionSize = vec3i(8,8,8));

  // Subdivide volume into regions, calculate entropy
  // for each region. Result is stored in dst, which
  // is an allocated array that can hold numRegions
  // floating point values
  VVAPI void entropyRegions(const vvVolDesc* vd,
      EntropyRegion* dst,
      vec3i regionSize = vec3i(8,8,8));

} // stats
} // virvo

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
