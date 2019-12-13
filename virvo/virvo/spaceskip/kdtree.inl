// Virvo - Virtual Reality Volume Rendering
// Copyright (C) 1999-2003 University of Stuttgart, 2004-2005 Brown University
// Contact: Jurgen P. Schulze, jschulze@ucsd.edu
//
// This file is part of Virvo.
//
// Virvo is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library (see license.txt); if not, write to the
// Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA

#include <virvo/vvclock.h>

template <typename Tex>
void KdTree::updateTransfunc(Tex transfunc)
{
  using namespace visionaray;

#ifdef BUILD_TIMING
  vvStopwatch sw; sw.start();
#endif
  psvt.build(transfunc);
#ifdef BUILD_TIMING
  //std::cout << std::fixed << std::setprecision(3) << "svt update: " << sw.getTime() << " sec.\n";
  std::cout << sw.getTime() << '\t';
#endif

#ifdef BUILD_TIMING
  sw.start();
#endif
  Node root;
  root.bbox = psvt.boundary(aabbi(vec3i(0), vec3i(vox[0], vox[1], vox[2])));

  nodes.clear();
  nodes.emplace_back(root);
  node_splitting(0);
#ifdef BUILD_TIMING
  //std::cout << "splitting: " << sw.getTime() << " sec.\n";
  std::cout << sw.getTime() << " \n";
#endif

#ifdef STATISTICS
  // Occupancy: see cudakdtree.cu

  // Number of voxels bound in leaves
  visionaray::vec3 eye(1,1,1);
  size_t vol = 0;
  size_t numLeaves = 0;
  size_t numNodes = 0;
  traverse(0 /*root*/, eye, [&numLeaves,&numNodes,&vol](Node const& n)
  {
    if (n.left == -1 && n.right == -1)
    {
      auto s = n.bbox.size();
      vol += size_t(s.x) * s.y * s.z;
      ++numLeaves;
    }
    ++numNodes;
  }, true);
  std::cout << vol << " voxels bound in " << numLeaves << " leaf nodes (" << numNodes << " nodes total)\n";
  std::cout << "Fraction bound: " << double(vol) / double(size_t(vox[0]) * vox[1] * vox[2]) << '\n';
#endif
}
