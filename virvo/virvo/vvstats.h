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

  // Calculate histogram from voxels in bbox
  VVAPI void makeHistogram(const vvVolDesc* vd,
      aabbi bbox,
      int chan1,
      int numChan,
      int* numBuckets,
      int* count);

  // Calculate histogram from voxels in index list
  VVAPI void makeHistogram(const vvVolDesc* vd,
      const vec3i* indices,
      size_t numIndices,
      int chan1,
      int numChan,
      int* numBuckets,
      int* count);

  // Calculate entropy from voxels in bbox
  VVAPI float entropy(const vvVolDesc* vd,
      aabbi bbox,
      int chan1,
      int numChan);

  // Calculate entropy from voxels in index list
  VVAPI float entropy(const vvVolDesc& vd,
      const vec3i* indices,
      size_t numIndices,
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

  VVAPI void makeSphericalNeighborhood(const vvVolDesc& vd,
      vec3i center,
      int radius,
      std::vector<vec3i>& indices);

} // stats
} // virvo

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
