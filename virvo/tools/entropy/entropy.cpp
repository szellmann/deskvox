#include <iostream>
#include <string>

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

  vec3i regionSize(4,4,4);
  vec3i num = stats::numEntropyRegions(&vd, regionSize);
  std::vector<stats::EntropyRegion> dst(num.x*num.y*num.z);
  stats::entropyRegions(&vd, dst.data(), regionSize);
  saveEntropyRegions(dst.data(), num, "/Users/stefan/entropy.rvf", true);

  return 0;
}

// vim: sw=2:expandtab:softtabstop=2:ts=2:cino=\:0g0t0
