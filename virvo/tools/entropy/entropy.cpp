#include <algorithm>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <virvo/vvfileio.h>
#include <virvo/vvstats.h>
#include <virvo/vvvoldesc.h>

using namespace virvo;

void saveEntropyRegions(const stats::EntropyRegion* regions, vec3i numRegions, std::string filename, bool overwrite)
{
  int size = numRegions.x * numRegions.y * numRegions.z;

  float min = (*std::min_element(regions, regions+size)).entropy;
  float max = (*std::max_element(regions, regions+size)).entropy;

  std::vector<uchar> bytes(size);

  for (int i=0; i<size; ++i)
  {
    bytes[i] = (int)(((regions[i].entropy-min) / (max-min)) * 255.f);
  }

  vvVolDesc vd;
  vd.setFilename(filename.c_str());
  vd.bpc = 1;
  vd.chan = 1;
  for (int i=0; i<3; ++i)
  {
    vd.vox[i] = numRegions[i];
    vd.dist[i] = 1.0f;
  }

  vd.addFrame((uchar*)bytes.data(), vvVolDesc::NO_DELETE);
  ++vd.frames;

  vvFileIO fio;
  if (fio.saveVolumeData(&vd, overwrite) != vvFileIO::OK)
    std::cerr << "Error!\n";
}

void saveImage(std::string filename, std::vector<float> const& data)
{
  constexpr int H = 150;
  std::ofstream file(filename);
  file << "P3\n";
  file << data.size() << ' ' << H << '\n';
  file << "255\n";

  for (int y=0; y<H; ++y)
  {
    float m=(H-y-1)/(float)H;

    for (int x=0; x<data.size(); ++x)
    {
      if (data[x] < m)
        file << "  0   0   0 ";
      else
        file << "255 255 255 ";
    }
    file << '\n';
  }
}

void saveHistogram(std::string filename, const std::vector<int>& histogram)
{
  // Normalize data
  int max = 0;
  for (auto h : histogram)
  {
    if (h > max) max = h;
  }

  std::vector<float> data(histogram.size());

  for (size_t i=0; i<data.size(); ++i)
  {
    data[i] = logf(float(histogram[i])) / logf(float(max));
  }

  saveImage(filename, data);
}

// 1st int in pairs should be a sorted sequence..
void saveOccurrences(std::string filename, const std::vector<std::pair<int, int>>& occurrences)
{
  // Normalize data
  int max = 0;
  for (auto o : occurrences)
  {
    if (o.second > max) max = o.second;
  }

  std::vector<float> data(occurrences.size());

  for (size_t i=0; i<data.size(); ++i)
  {
    data[i] = logf(float(occurrences[i].second)) / logf(float(max));
  }

  saveImage(filename, data);
}

int main(int argc, char** argv)
{
  if (argc < 2)
    std::cerr << "Usage: entropy <filename>\n";

  vvVolDesc vd(argv[1]);

  vvFileIO fio;
  if (fio.loadVolumeData(&vd, vvFileIO::ALL_DATA) != vvFileIO::OK)
  {
    std::cerr << "Error loading volume file" << std::endl;
    return EXIT_FAILURE;
  }
  else vd.printInfoLine();

  vec3i regionSize(16,16,16);
  vec3i num = stats::numEntropyRegions(&vd, regionSize);
  std::vector<stats::EntropyRegion> dst(num.x*num.y*num.z);
  stats::entropyRegions(&vd, dst.data(), regionSize);
  saveEntropyRegions(dst.data(), num, "/Users/stefan/entropy.rvf", true);

  // Sort regions by entropy
  std::sort(dst.begin(), dst.end());

  // Make two groups using the median: low vs. high entropy
  // TODO: different heuristics?
  float m = (dst[0].entropy + dst[dst.size()-1].entropy) / 2;

  // Simple linear search
  int medianIndex = 0;
  for (size_t i=0; i<dst.size(); ++i)
  {
    if (dst[i].entropy > m)
    {
      medianIndex = i;
      break;
    }
  }

  // For each voxel count if it is in the low and/or
  // high entropy list
  typedef std::pair<int/*index*/, int/*occurrences in lo/hi entropy regions*/> Voxel;
  std::vector<Voxel> lo;
  std::vector<Voxel> hi;

  int numVoxels = vd.bpc == 1 ? 255 : 65535;

  for (int i=0; i<numVoxels; ++i)
  {
    lo.push_back(std::make_pair(i,0));
    hi.push_back(std::make_pair(i,0));
  }

  // Build local histograms for low/high entropy regions,
  // get the dominant voxels and increase their respective count

  int numBuckets[] = { vd.bpc == 1 ? 255 : 65535 };
  for (size_t i=0; i<dst.size(); ++i)
  {
    std::vector<int> histogram(numBuckets[0]);

    stats::makeHistogram(&vd, dst[i].bbox, 0, 1, numBuckets, histogram.data());

    // TODO: maybe use the N=2,3,.. dominant voxels
    int dominantVoxelOccurrences = 0;
    int dominantVoxel = -1;
    for (size_t j=0; j<histogram.size(); ++j)
    {
      if (histogram[j] > dominantVoxelOccurrences)
      {
        dominantVoxelOccurrences = histogram[j];
        dominantVoxel = j;
      }
    }

    if (i < medianIndex)
      ++lo[dominantVoxel].second;
    else
      ++hi[dominantVoxel].second;
  }

  std::vector<int> globalHistogram(numBuckets[0]);
  aabbi bbox(vec3i(0,0,0), vec3i(vd.vox));
  stats::makeHistogram(&vd, bbox, 0, 1, numBuckets, globalHistogram.data());
  saveHistogram("/Users/stefan/histo.ppm", globalHistogram);
  saveOccurrences("/Users/stefan/lo.ppm", lo);
  saveOccurrences("/Users/stefan/hi.ppm", hi);

  std::sort(lo.begin(), lo.end(), [](const Voxel& a, const Voxel& b){ return a.second < b.second; });
  std::sort(hi.begin(), hi.end(), [](const Voxel& a, const Voxel& b){ return a.second < b.second; });

  std::cout << lo[lo.size()-1].first << '\n';
  std::cout << hi[hi.size()-1].first << '\n';

  return 0;
}

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
