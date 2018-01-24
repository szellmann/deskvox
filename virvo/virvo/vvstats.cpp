#include <algorithm>
#include <fstream>
#include <functional>
#include <numeric>
#include <sstream>
#include <vector>

#include "vvclock.h"
#include "vvfileio.h"
#include "vvstats.h"
#include "vvtoolshed.h"
#include "vvvoldesc.h"

namespace virvo
{
namespace stats
{

void makeHistogram(const vvVolDesc* vd,
    aabbi bbox,
    int chan1,
    int numChan,
    int* buckets,
    int* count)
{
  int totalBuckets = std::accumulate(buckets, buckets+numChan, 1, std::multiplies<int>());
  std::fill(count, count+totalBuckets, 0);        // initialize counter array

  int frame = 0;

  //vvStopwatch sw;sw.start();
  for (int z=bbox.min.z; z != bbox.max.z; ++z)
    for (int y=bbox.min.y; y != bbox.max.y; ++y)
      for (int x=bbox.min.x; x != bbox.max.x; ++x)
  {
    int dstIndex = 0;                             // index into histogram array
    int factor = 1;                               // multiplication factor for dstIndex
    for (int c=0; c<numChan; ++c) 
    {
      size_t index = (z*vd->vox[0]*vd->vox[1] + y*vd->vox[0] + x) * vd->getBPV();
      float voxVal = vd->getChannelValue(frame, index, chan1+c);

      // Bucket index with respect to channel c
      int bucketIndex = (int)((voxVal - vd->range(c).x)
          * (buckets[c] / (vd->range(c).y-vd->range(c).x)));
      bucketIndex = ts_clamp(bucketIndex, 0, buckets[c]-1);

      dstIndex += bucketIndex * factor;
      factor *= buckets[c];
    }    
    ++count[dstIndex];
  }
  //std::cout << sw.getTime() << '\n';
}

void makeHistogram(const vvVolDesc* vd,
    const vec3i* indices,
    size_t numIndices,
    int chan1,
    int numChan,
    int* buckets,
    int* count)
{
  int totalBuckets = std::accumulate(buckets, buckets+numChan, 1, std::multiplies<int>());
  std::fill(count, count+totalBuckets, 0);        // initialize counter array

  int frame = 0;

  //vvStopwatch sw;sw.start();
  for (size_t i=0; i<numIndices; ++i)
  {
    int x = indices[i].x;
    int y = indices[i].y;
    int z = indices[i].z;

    int dstIndex = 0;                             // index into histogram array
    int factor = 1;                               // multiplication factor for dstIndex
    for (int c=0; c<numChan; ++c) 
    {
      size_t index = (z*vd->vox[0]*vd->vox[1] + y*vd->vox[0] + x) * vd->getBPV();
      float voxVal = vd->getChannelValue(frame, index, chan1+c);

      // Bucket index with respect to channel c
      int bucketIndex = (int)((voxVal - vd->range(c).x)
          * (buckets[c] / (vd->range(c).y-vd->range(c).x)));
      bucketIndex = ts_clamp(bucketIndex, 0, buckets[c]-1);

      dstIndex += bucketIndex * factor;
      factor *= buckets[c];
    }    
    ++count[dstIndex];
  }
  //std::cout << sw.getTime() << '\n';
}

float entropy(const vvVolDesc* vd,
    aabbi bbox,
    int chan1,
    int numChan)
{
  // TODO: figure out how to make the following compatible w/ more than one channel
  // ...
  assert(chan1 == 0 && numChan == 1);

  assert(vd->bpc <= 2);

  int numVoxels = bbox.size().x * bbox.size().y * bbox.size().z;

  int numBuckets[] = { vd->bpc == 1 ? 255 : 65535 };

  std::vector<int> histogram(numBuckets[0]);

  makeHistogram(vd, bbox, chan1, numChan, numBuckets, histogram.data());

  std::vector<float> probs(numBuckets[0]);

  for (int i=0; i<numBuckets[0]; ++i)
    probs[i] = histogram[i]/static_cast<float>(numVoxels);

  float result = 0.0f;
  for (int i=0; i<numBuckets[0]; ++i)
    if (probs[i] > 0.f)
      result += probs[i]*log2(probs[i]);

  return -result;
}

float entropy(const vvVolDesc& vd,
    const vec3i* indices,
    size_t numIndices,
    int chan1,
    int numChan)
{
  // TODO: figure out how to make the following compatible w/ more than one channel
  // ...
  assert(chan1 == 0 && numChan == 1);

  assert(vd.bpc <= 2);

  int numVoxels(numIndices);

  int numBuckets[] = { vd.bpc == 1 ? 255 : 65535 };

  std::vector<int> histogram(numBuckets[0]);

  makeHistogram(&vd, indices, numIndices, chan1, numChan, numBuckets, histogram.data());

  std::vector<float> probs(numBuckets[0]);

  for (int i=0; i<numBuckets[0]; ++i)
    probs[i] = histogram[i]/static_cast<float>(numVoxels);

  float result = 0.0f;
  for (int i=0; i<numBuckets[0]; ++i)
    if (probs[i] > 0.f)
      result += probs[i]*log2(probs[i]);

  return -result;
}

vec3i numEntropyRegions(const vvVolDesc* vd, vec3i regionSize)
{
  vec3i result;
  for (int i=0; i<3; ++i)
  {
    result[i] = (int)vd->vox[i] / regionSize[i];
    if (result[i] * regionSize[i] < vd->vox[i])
      ++result[i];
  }
  return result;
}

void entropyRegions(const vvVolDesc* vd, EntropyRegion* dst, vec3i regionSize)
{
  int sizex = (int)vd->vox[0];
  int sizey = (int)vd->vox[1];
  int sizez = (int)vd->vox[2];

  int i = 0;

  for (int z=0; z<sizez; z+=regionSize.z)
    for (int y=0; y<sizey; y+=regionSize.y)
      for (int x=0; x<sizex; x+=regionSize.x)
  {
    int maxx = std::min(x + regionSize.x, sizex);
    int maxy = std::min(y + regionSize.y, sizey);
    int maxz = std::min(z + regionSize.z, sizez);

    dst[i].bbox = aabbi(vec3i(x,y,z), vec3i(maxx,maxy,maxz));
    dst[i].entropy = entropy(vd, dst[i].bbox, 0, 1);
    ++i;
  }
}

void makeSphericalNeighborhood(const vvVolDesc& vd, vec3i center, int radius, std::vector<vec3i>& indices)
{
  aabbi bbox(center-radius, center+radius);

  for (int z=bbox.min.z; z <= bbox.max.z; ++z)
    for (int y=bbox.min.y; y <= bbox.max.y; ++y)
      for (int x=bbox.min.x; x <= bbox.max.x; ++x)
  {
    if (x < 0 || x >= vd.vox[0] || y < 0 || y >= vd.vox[1] || z < 0 || z >= vd.vox[2])
      continue;

    vec3f cf(center);
    vec3f xyz(x,y,z);

    if (length(xyz-cf) <= static_cast<float>(radius))
        indices.push_back(vec3i(x,y,z));
  }
}

} // stats
} // virvo

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
