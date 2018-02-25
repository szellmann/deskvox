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

#ifndef NDEBUG
#include <iostream>
#include <ostream>
#endif
#include <vector>

#include <GL/glew.h>

#include <thrust/device_vector.h>

#undef MATH_NAMESPACE

#include <visionaray/math/math.h>
#include <visionaray/texture/texture.h>
#include <visionaray/pixel_format.h>
#include <visionaray/pixel_traits.h>
#include <visionaray/point_light.h>
#include <visionaray/render_target.h>
#include <visionaray/result_record.h>
#include <visionaray/scheduler.h>

#undef MATH_NAMESPACE

#include "cuda/utils.h"
#include "gl/util.h"
#include "vvclock.h"
#include "vvcudarendertarget.h"
#include "vvskipraycaster.h"
#include "vvtextureutil.h"
#include "vvvoldesc.h"

using namespace visionaray;

#define FRAME_TIMING 0
#define BUILD_TIMING 1
#define KDTREE       1

//-------------------------------------------------------------------------------------------------
// Summed-volume table
//

template <typename T>
struct SVT
{
    void reset(aabbi bbox);
    void reset(vvVolDesc const& vd, aabbi bbox, int channel = 0);

    template <typename Tex>
    void apply(Tex transfunc);

    void build();

    T& operator()(int x, int y, int z)
    {
        return data_[z * width * height + y * width + x];
    }

    T& at(int x, int y, int z)
    {
        return data_[z * width * height + y * width + x];
    }

    T const& at(int x, int y, int z) const
    {
        return data_[z * width * height + y * width + x];
    }

    T border_at(int x, int y, int z) const
    {
        if (x < 0 || y < 0 || z < 0)
            return 0;

        return data_[z * width * height + y * width + x];
    }

    T last() const
    {
        return data_.back();
    }

    T* data()
    {
        return data_.data();
    }

    T const* data() const
    {
        return data_.data();
    }

    T get_count(basic_aabb<int> bounds) const
    {
        bounds.min -= vec3i(1);
        bounds.max -= vec3i(1);

        return border_at(bounds.max.x, bounds.max.y, bounds.max.z)
             - border_at(bounds.max.x, bounds.max.y, bounds.min.z)
             - border_at(bounds.max.x, bounds.min.y, bounds.max.z)
             - border_at(bounds.min.x, bounds.max.y, bounds.max.z)
             + border_at(bounds.min.x, bounds.min.y, bounds.max.z)
             + border_at(bounds.min.x, bounds.max.y, bounds.min.z)
             + border_at(bounds.max.x, bounds.min.y, bounds.min.z)
             - border_at(bounds.min.x, bounds.min.y, bounds.min.z);
    }

    // Channel values from volume description
    std::vector<float> voxels_;
    // SVT array
    std::vector<T> data_;
    int width;
    int height;
    int depth;
};

template <typename T>
void SVT<T>::reset(aabbi bbox)
{
    data_.resize(bbox.size().x * bbox.size().y * bbox.size().z);
    width  = bbox.size().x;
    height = bbox.size().y;
    depth  = bbox.size().z;
}

template <typename T>
void SVT<T>::reset(vvVolDesc const& vd, aabbi bbox, int channel)
{
    voxels_.resize(bbox.size().x * bbox.size().y * bbox.size().z);
    data_.resize(bbox.size().x * bbox.size().y * bbox.size().z);
    width  = bbox.size().x;
    height = bbox.size().y;
    depth  = bbox.size().z;


    for (int z = 0; z < depth; ++z)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                size_t index = z * width * height + y * width + x;
                voxels_[index] = vd.getChannelValue(0,
                            bbox.min.x + x,
                            bbox.min.y + y,
                            bbox.min.z + z,
                            channel);
            }
        }
    }
}

template <typename T>
template <typename Tex>
void SVT<T>::apply(Tex transfunc)
{
    for (int z = 0; z < depth; ++z)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                size_t index = z * width * height + y * width + x;
                if (tex1D(transfunc, voxels_[index]).w < 0.0001)
                    at(x, y, z) = T(0);
                else
                    at(x, y, z) = T(1);
            }
        }
    }
}

template <typename T>
void SVT<T>::build()
{
    // Build summed volume table

    // Init 0-border voxel
    //at(0, 0, 0) = at(0, 0, 0);

    // Init 0-border edges (prefix sum)
    for (int x=1; x<width; ++x)
    {
        at(x, 0, 0) = at(x, 0, 0) + at(x-1, 0, 0);
    }

    for (int y=1; y<height; ++y)
    {
        at(0, y, 0) = at(0, y, 0) + at(0, y-1, 0);
    }

    for (int z=1; z<depth; ++z)
    {
        at(0, 0, z) = at(0, 0, z) + at(0, 0, z-1);
    }


    // Init 0-border planes (summed-area tables)
    for (int y=1; y<height; ++y)
    {
        for (int x=1; x<width; ++x)
        {
            at(x, y, 0) = at(x, y, 0)
                + at(x-1, y, 0) + at(x, y-1, 0)
                - at(x-1, y-1, 0);
        }
    }

    for (int z=1; z<depth; ++z)
    {
        for (int y=1; y<height; ++y)
        {
            at(0, y, z) = at(0, y, z)
                + at(0, y-1, z) + at(0, y, z-1)
                - at(0, y-1, z-1);
        }
    }

    for (int x=1; x<width; ++x)
    {
        for (int z=1; z<depth; ++z)
        {
            at(x, 0, z) = at(x, 0, z)
                + at(x-1, 0, z) + at(x, 0, z-1)
                - at(x-1, 0, z-1);
        }
    }


    // Build up SVT
    for (int z=1; z<depth; ++z)
    {
        for (int y=1; y<height; ++y)
        {
            for (int x=1; x<width; ++x)
            {
                at(x, y, z) = at(x, y, z) + at(x-1, y-1, z-1)
                    + at(x-1, y, z) - at(x, y-1, z-1)
                    + at(x, y-1, z) - at(x-1, y, z-1)
                    + at(x, y, z-1) - at(x-1, y-1, z);
            }
        }
    }
}


//-------------------------------------------------------------------------------------------------
// two-level SVT
//

struct TLSVT
{
    typedef SVT<uint16_t> svt_t;

    void reset(vvVolDesc const& vd, aabbi bbox, int channel = 0);

    template <typename Tex>
    void build(Tex transfunc);

    uint64_t get_count(aabbi bounds) const;

    vec3i bricksize = vec3i(32, 32, 32);

    vec3i num_svts;
    std::vector<svt_t> svts;
};

void TLSVT::reset(vvVolDesc const& vd, aabbi bbox, int channel)
{
    num_svts = vec3i(div_up(bbox.max.x, bricksize.x),
                     div_up(bbox.max.y, bricksize.y),
                     div_up(bbox.max.z, bricksize.z));

    svts.resize(num_svts.x * num_svts.y * num_svts.z);


    // Fill with volume channel values

    int bz = 0;
    for (int z = 0; z < num_svts.z; ++z)
    {
        int by = 0;
        for (int y = 0; y < num_svts.y; ++y)
        {
            int bx = 0;
            for (int x = 0; x < num_svts.x; ++x)
            {
                vec3i bmin(bx, by, bz);
                vec3i bmax(min(bbox.max.x, bx + bricksize.x),
                           min(bbox.max.y, by + bricksize.y),
                           min(bbox.max.z, bz + bricksize.z));
                svts[z * num_svts.x * num_svts.y + y * num_svts.x + x].reset(vd, aabbi(bmin, bmax), channel);

                bx += bricksize.x;
            }

            by += bricksize.y;
        }

        bz += bricksize.z;
    }
}

template <typename Tex>
void TLSVT::build(Tex transfunc)
{
    #pragma omp parallel for
    for (size_t i = 0; i < svts.size(); ++i)
    {
        svts[i].apply(transfunc);
        svts[i].build();
    }
}

uint64_t TLSVT::get_count(aabbi bounds) const
{
    vec3i min_brick = bounds.min / bricksize;
    vec3i min_bpos = bounds.min - min_brick * bricksize;

    vec3i max_brick = bounds.max / bricksize;
    vec3i max_bpos = bounds.max - max_brick * bricksize;

    uint64_t count = 0;

    for (int bz = min_brick.z; bz <= max_brick.z; ++bz)
    {
        int minz = bz == min_brick.z ? min_bpos.z : 0;
        int maxz = bz == max_brick.z ? max_bpos.z : bricksize.z;

        for (int by = min_brick.y; by <= max_brick.y; ++by)
        {
            int miny = by == min_brick.y ? min_bpos.y : 0;
            int maxy = by == max_brick.y ? max_bpos.y : bricksize.y;

            for (int bx = min_brick.x; bx <= max_brick.x; ++bx)
            {
                int minx = bx == min_brick.x ? min_bpos.x : 0;
                int maxx = bx == max_brick.x ? max_bpos.x : bricksize.x;

                // 32**3 bricks can store 16 bit values, but the
                // overall count will generally not fit in 16 bits
                count += static_cast<uint64_t>(svts[bz * num_svts.x * num_svts.y + by * num_svts.x + bx].get_count(
                        aabbi(vec3i(minx, miny, minz), vec3i(maxx, maxy, maxz))));
            }
        }
    }

    return count;
}


//-------------------------------------------------------------------------------------------------
// Kd-tree (Vidal et al. 2008)
//

struct KdTree
{
    struct Node;
    typedef std::unique_ptr<Node> NodePtr;

    struct Node
    {
        aabbi bbox;
        NodePtr left  = nullptr;
        NodePtr right = nullptr;
        int axis = -1;
        int splitpos = -1;
        int depth;
    };

    template <typename Func>
    void traverse(NodePtr const& n, vec3 eye, Func f) const
    {
        if (n != nullptr)
        {
            f(n);

            if (n->axis >= 0)
            {
                int spi = n->splitpos;
                if (n->axis == 1 || n->axis == 2)
                    spi = vox[n->axis] - spi - 1;
                float splitpos = (spi - vox[n->axis]/2.f) * dist[n->axis] * scale;

                // TODO: puh..
                if (n->axis == 0 && eye[n->axis] < splitpos || n->axis == 1 && eye[n->axis] >= splitpos || n->axis == 2 && eye[n->axis] >= splitpos)
                {
                    traverse(n->left, eye, f);
                    traverse(n->right, eye, f);
                }
                else
                {
                    traverse(n->right, eye, f);
                    traverse(n->left, eye, f);
                }
            }
        }
    }

    TLSVT hsvt;

    NodePtr root = nullptr;

    vec3i vox;
    vec3 dist;
    float scale;

    void updateVolume(vvVolDesc const& vd, int channel = 0);

    template <typename Tex>
    void updateTransfunc(Tex transfunc);

    void node_splitting(NodePtr& n);
    aabbi boundary(aabbi bbox) const;

    std::vector<aabb> get_leaf_nodes(vec3 eye) const;

    // Need OpenGL context!
    void renderGL() const;
    // Need OpenGL context!
    void renderGL(NodePtr const& n) const;
};

void KdTree::updateVolume(vvVolDesc const& vd, int channel)
{
    vox = vec3i(vd.vox.x, vd.vox.y, vd.vox.z);
    dist = vec3(vd.getDist().x, vd.getDist().y, vd.getDist().z);
    scale = vd._scale;

    hsvt.reset(vd, aabbi(vec3i(0), vox), channel);
}

template <typename Tex>
void KdTree::updateTransfunc(Tex transfunc)
{
#ifdef BUILD_TIMING
    vvStopwatch sw; sw.start();
#endif
    hsvt.build(transfunc);
#ifdef BUILD_TIMING
    std::cout << std::fixed << std::setprecision(3) << "svt update: " << sw.getTime() << " sec.\n";
#endif

#ifdef BUILD_TIMING
    sw.start();
#endif
    root.reset(new Node);
    root->bbox = boundary(aabbi(vec3i(0), vec3i(vox[0], vox[1], vox[2])));
    root->depth = 0;
    node_splitting(root);
#ifdef BUILD_TIMING
    std::cout << "splitting: " << sw.getTime() << " sec.\n";
#endif
}

void KdTree::node_splitting(KdTree::NodePtr& n)
{
    // Halting criterion 1.)
    if (volume(n->bbox) < volume(root->bbox) / 10)
        return;

    // Split along longest axis
    vec3i len = n->bbox.max - n->bbox.min;

    int axis = 0;
    if (len.y > len.x && len.y > len.z)
        axis = 1;
    else if (len.z > len.x && len.z > len.y)
        axis = 2;

    int lmax = len[axis];

    static const int dl = 4; // ``we set dl to be 4 for 256^3 data sets..''
//  static const int dl = 8; // ``.. and 8 for 512^3 data sets.''

    // Halting criterion 1.b) (should not really get here..)
    if (lmax < dl)
        return;

    int num_planes = lmax / dl;

    int min_cost = INT_MAX;
    int best_p = -1;

    aabbi lbox = n->bbox;
    aabbi rbox = n->bbox;

    int first = lbox.min[axis];

    int vol = volume(n->bbox);

    for (int p = 1; p < num_planes; ++p)
    {
        aabbi ltmp = n->bbox;
        aabbi rtmp = n->bbox;

        ltmp.max[axis] = first + dl * p;
        rtmp.min[axis] = first + dl * p;

        ltmp = boundary(ltmp);
        rtmp = boundary(rtmp);

        int c = volume(ltmp) + volume(rtmp);

        // empty-space volume
        int ev = vol - c;

        // Halting criterion 2.)
        if (ev <= vol / 20)
            continue;

        if (c < min_cost)
        {
            min_cost = c;
            lbox = ltmp;
            rbox = rtmp;
            best_p = p;
        }
    }

    // Halting criterion 2.)
    if (best_p < 0)
        return;

    // Store split plane for traversal
    n->axis = axis;
    n->splitpos = first + dl * best_p;

    n->left.reset(new Node);
    n->left->bbox = lbox;
    n->left->depth = n->depth + 1;
    node_splitting(n->left);

    n->right.reset(new Node);
    n->right->bbox = rbox;
    n->right->depth = n->depth + 1;
    node_splitting(n->right);
}

// produce a boundary around the *non-empty* voxels in bbox
aabbi KdTree::boundary(aabbi bbox) const
{
    aabbi bounds = bbox;

    // Search for the minimal volume bounding box
    // that contains #voxels contained in bbox!
    uint16_t voxels = hsvt.get_count(bounds);


    // X boundary from left
    for (int x = bounds.min.x; x < bounds.max.x; ++x)
    {
        aabbi lbox = bounds;
        lbox.min.x = x;

        if (hsvt.get_count(lbox) == voxels)
        {
            bounds = lbox;
        }
        else
        {
            break;
        }
    }


    // Y boundary from left
    for (int y = bounds.min.y; y < bounds.max.y; ++y)
    {
        aabbi lbox = bounds;
        lbox.min.y = y;

        if (hsvt.get_count(lbox) == voxels)
        {
            bounds = lbox;
        }
        else
        {
            break;
        }
    }


    // Z boundary from left
    for (int z = bounds.min.z; z < bounds.max.z; ++z)
    {
        aabbi lbox = bounds;
        lbox.min.z = z;

        if (hsvt.get_count(lbox) == voxels)
        {
            bounds = lbox;
        }
        else
        {
            break;
        }
    }


    // X boundary from right
    for (int x = bounds.max.x; x > bounds.min.x; --x)
    {
        aabbi rbox = bounds;
        rbox.max.x = x;

        if (hsvt.get_count(rbox) == voxels)
        {
            bounds = rbox;
        }
        else
        {
            break;
        }
    }


    // Y boundary from right
    for (int y = bounds.max.y; y > bounds.min.y; --y)
    {
        aabbi rbox = bounds;
        rbox.max.y = y;

        if (hsvt.get_count(rbox) == voxels)
        {
            bounds = rbox;
        }
        else
        {
            break;
        }
    }


    // Z boundary from right
    for (int z = bounds.max.z; z > bounds.min.z; --z)
    {
        aabbi rbox = bounds;
        rbox.max.z = z;

        if (hsvt.get_count(rbox) == voxels)
        {
            bounds = rbox;
        }
        else
        {
            break;
        }
    }
//std::cout << bbox.min << ' ' << bbox.max << '\n';
//std::cout << bounds.min << ' ' << bounds.max << '\n';
//std::cout << '\n';

    return bounds;
}

std::vector<aabb> KdTree::get_leaf_nodes(vec3 eye) const
{
    std::vector<aabb> result;

    traverse(root, eye, [&result,this,eye](NodePtr const& n)
    {
        if (n->left == nullptr && n->right == nullptr)
        {
            auto bbox = n->bbox;
            bbox.max.y = vox[1] - bbox.max.y - 1;
            bbox.min.y = vox[1] - bbox.min.y - 1;
            bbox.max.z = vox[2] - bbox.max.z - 1;
            bbox.min.z = vox[2] - bbox.min.z - 1;
            vec3 bmin = (vec3(bbox.min) - vec3(vox)/2.f) * dist * scale;
            vec3 bmax = (vec3(bbox.max) - vec3(vox)/2.f) * dist * scale;
//std::cout << length(aabb(bmin, bmax).center() - eye) << '\n';

            result.push_back(aabb(bmin, bmax));
        }
    });

    return result;
}

void KdTree::renderGL() const
{
    renderGL(root);
}

void KdTree::renderGL(KdTree::NodePtr const& n) const
{
    if (n != nullptr)
    {
        if (n->left == nullptr && n->right == nullptr)
        {
            auto bbox = n->bbox;
            bbox.max.y = vox[1] - bbox.max.y - 1;
            bbox.min.y = vox[1] - bbox.min.y - 1;
            bbox.max.z = vox[2] - bbox.max.z - 1;
            bbox.min.z = vox[2] - bbox.min.z - 1;
            vec3 bmin = (vec3(bbox.min) - vec3(vox)/2.f) * dist * scale;
            vec3 bmax = (vec3(bbox.max) - vec3(vox)/2.f) * dist * scale;

            glBegin(GL_LINES);
            glColor3f(0,0,0);

            glVertex3f(bmin.x, bmin.y, bmin.z);
            glVertex3f(bmax.x, bmin.y, bmin.z);

            glVertex3f(bmax.x, bmin.y, bmin.z);
            glVertex3f(bmax.x, bmax.y, bmin.z);

            glVertex3f(bmax.x, bmax.y, bmin.z);
            glVertex3f(bmin.x, bmax.y, bmin.z);

            glVertex3f(bmin.x, bmax.y, bmin.z);
            glVertex3f(bmin.x, bmin.y, bmin.z);

            //
            glVertex3f(bmin.x, bmin.y, bmax.z);
            glVertex3f(bmax.x, bmin.y, bmax.z);

            glVertex3f(bmax.x, bmin.y, bmax.z);
            glVertex3f(bmax.x, bmax.y, bmax.z);

            glVertex3f(bmax.x, bmax.y, bmax.z);
            glVertex3f(bmin.x, bmax.y, bmax.z);

            glVertex3f(bmin.x, bmax.y, bmax.z);
            glVertex3f(bmin.x, bmin.y, bmax.z);

            //
            glVertex3f(bmin.x, bmin.y, bmin.z);
            glVertex3f(bmin.x, bmin.y, bmax.z);

            glVertex3f(bmax.x, bmin.y, bmin.z);
            glVertex3f(bmax.x, bmin.y, bmax.z);

            glVertex3f(bmax.x, bmax.y, bmin.z);
            glVertex3f(bmax.x, bmax.y, bmax.z);

            glVertex3f(bmin.x, bmax.y, bmin.z);
            glVertex3f(bmin.x, bmax.y, bmax.z);
            glEnd();
        }

        renderGL(n->left);
        renderGL(n->right);
    }
}


//-------------------------------------------------------------------------------------------------
// Volume rendering kernel
//

struct Kernel
{
    template <typename R, typename T, typename C>
    VSNRAY_FUNC
    void integrate(R ray, T t, T tmax, C& dst) const
    {
        integrate(ray, t, tmax, dst, delta);
    }

    template <typename R, typename T, typename C>
    VSNRAY_FUNC
    void integrate(R ray, T t, T tmax, C& dst, T dt) const
    {
        while (t < tmax)
        {
            auto pos = ray.ori + ray.dir * t;
            vector<3, T> tex_coord(
                    ( pos.x + (bbox.size().x / 2) ) / bbox.size().x,
                    (-pos.y + (bbox.size().y / 2) ) / bbox.size().y,
                    (-pos.z + (bbox.size().z / 2) ) / bbox.size().z
                    );

            T voxel = tex3D(volume, tex_coord);
            C color = tex1D(transfunc, voxel);

            // opacity correction
            color.w = 1.0f - pow(1.0f - color.w, dt);

            // premultiplied alpha
            color.xyz() *= color.w;

            // compositing
            dst += color * (1.0f - dst.w);

            // step on
            t += dt;
        }
    }

    template <typename R>
    VSNRAY_FUNC
    result_record<typename R::scalar_type> operator()(R ray) const
    {
        using S = typename R::scalar_type;
        using C = vector<4, S>;

        result_record<S> result;
        result.color = C(0.0);

        auto hit_rec = intersect(ray, bbox);
        result.hit = hit_rec.hit;

        if (!hit_rec.hit)
            return result;

        auto t = max(S(0.0f), hit_rec.tnear);
        auto tmax = hit_rec.tfar;

#if KDTREE
        for (int i = 0; i < num_kd_tree_leaves; ++i)
        {
            auto kd_hit_rec = intersect(ray, kd_tree_leaves[i]);

            if (kd_hit_rec.hit)
            {
                auto kd_t = max(t, kd_hit_rec.tnear);
                auto kd_tmax = min(tmax, kd_hit_rec.tfar);

                integrate(ray, kd_t, kd_tmax, result.color);

                t = kd_tmax;
            }
        }

#else
        integrate(ray, t, tmax, result.color);
#endif

        return result;
    }

    cuda_texture<unorm<8>, 3>::ref_type volume;
    cuda_texture<vec4, 1>::ref_type transfunc;

    aabb bbox;
    vec3i vox;
    float delta;
    bool local_shading;
    point_light<float> light;

    aabb* kd_tree_leaves;
    int num_kd_tree_leaves;
};


//-------------------------------------------------------------------------------------------------
// Private implementation
//

struct vvSkipRayCaster::Impl
{
    Impl() : sched(8, 8) {}

    using R = basic_ray<float>;

    cuda_sched<R> sched;

    std::vector<cuda_texture<unorm<8>, 3>> volumes;

#if KDTREE
    KdTree kdtree;
#endif

    cuda_texture<vec4, 1> transfunc;
};


//-------------------------------------------------------------------------------------------------
// Wrapper to consolidate virvo and Visionaray render targets
//

class virvo_render_target
{
public:

    static const pixel_format CF = PF_RGBA32F;
    static const pixel_format DF = PF_UNSPECIFIED;

    using color_type = typename pixel_traits<CF>::type;
    using depth_type = typename pixel_traits<DF>::type;

    using ref_type = render_target_ref<CF, DF>;

public:

    virvo_render_target(int w, int h, color_type* c, depth_type* d)
        : width_(w)
        , height_(h)
        , color_(c)
        , depth_(d)
    {
    }

    int width() const { return width_; }
    int height() const { return height_; }

    color_type* color() { return color_; }
    depth_type* depth() { return depth_; }

    color_type const* color() const { return color_; }
    depth_type const* depth() const { return depth_; }

    ref_type ref() { return { color(), depth(), width(), height() }; }

    void begin_frame() {}
    void end_frame() {}

    int width_;
    int height_;

    color_type* color_;
    depth_type* depth_;
};


//-------------------------------------------------------------------------------------------------
// Skip ray caster
//

vvSkipRayCaster::vvSkipRayCaster(vvVolDesc* vd, vvRenderState renderState)
    : vvRenderer(vd, renderState)
    , impl_(new Impl)
{
    rendererType = SKIPRAYREND;

    glewInit();

    virvo::cuda::initGlInterop();

    virvo::RenderTarget* rt = virvo::PixelUnpackBufferRT::create(virvo::PF_RGBA32F, virvo::PF_UNSPECIFIED);

    // no direct rendering
    if (rt == NULL)
    {
        rt = virvo::DeviceBufferRT::create(virvo::PF_RGBA32F, virvo::PF_UNSPECIFIED);
    }
    setRenderTarget(rt);

    updateVolumeData();
    updateTransferFunction();
}

vvSkipRayCaster::~vvSkipRayCaster()
{
}

void vvSkipRayCaster::renderVolumeGL()
{
    mat4 view_matrix;
    mat4 proj_matrix;
    recti viewport;

    glGetFloatv(GL_MODELVIEW_MATRIX, view_matrix.data());
    glGetFloatv(GL_PROJECTION_MATRIX, proj_matrix.data());
    glGetIntegerv(GL_VIEWPORT, viewport.data());

    virvo::RenderTarget* rt = getRenderTarget();

    assert(rt);

    virvo_render_target virvo_rt(
        rt->width(),
        rt->height(),
        static_cast<virvo_render_target::color_type*>(rt->deviceColor()),
        static_cast<virvo_render_target::depth_type*>(rt->deviceDepth())
        );

    auto sparams = make_sched_params(
        view_matrix,
        proj_matrix,
        virvo_rt
        );

    // determine ray integration step size (aka delta)
    int axis = 0;
    if (vd->getSize()[1] / vd->vox[1] < vd->getSize()[axis] / vd->vox[axis])
    {
        axis = 1;
    }
    if (vd->getSize()[2] / vd->vox[2] < vd->getSize()[axis] / vd->vox[axis])
    {
        axis = 2;
    }

    float delta = (vd->getSize()[axis] / vd->vox[axis]) / _quality;

    auto bbox = vd->getBoundingBox();

    // Lights
    point_light<float> light;

    if (getParameter(VV_LIGHTING))
    {
        assert( glIsEnabled(GL_LIGHTING) );
        auto l = virvo::gl::getLight(GL_LIGHT0);
        vec4 lpos(l.position.x, l.position.y, l.position.z, l.position.w);

        light.set_position( (inverse(view_matrix) * lpos).xyz() );
        light.set_cl(vec3(l.diffuse.x, l.diffuse.y, l.diffuse.z));
        light.set_kl(l.diffuse.w);
        light.set_constant_attenuation(l.constant_attenuation);
        light.set_linear_attenuation(l.linear_attenuation);
        light.set_quadratic_attenuation(l.quadratic_attenuation);
    }

    Kernel kernel;

    kernel.volume = cuda_texture<unorm<8>, 3>::ref_type(impl_->volumes[vd->getCurrentFrame()]);
    kernel.transfunc = cuda_texture<vec4, 1>::ref_type(impl_->transfunc);

    kernel.bbox          = aabb(vec3(bbox.min.data()), vec3(bbox.max.data()));
    kernel.vox           = vec3i(vd->vox[0], vd->vox[1], vd->vox[2]);
    kernel.delta         = delta;
    kernel.local_shading = getParameter(VV_LIGHTING);
    kernel.light         = light;

#if KDTREE
    vec3 eye(getEyePosition().x, getEyePosition().y, getEyePosition().z);
    auto leaves = impl_->kdtree.get_leaf_nodes(eye);
    thrust::device_vector<aabb> d_leaves(leaves);
    kernel.kd_tree_leaves = thrust::raw_pointer_cast(d_leaves.data());
    kernel.num_kd_tree_leaves = static_cast<int>(d_leaves.size());
#endif

#if FRAME_TIMING
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
#endif

    impl_->sched.frame(kernel, sparams);

#if FRAME_TIMING
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    std::cout << std::fixed << std::setprecision(3) << "Elapsed time: " << ms << "ms\n";
#endif

#if KDTREE
    if (_boundaries)
        impl_->kdtree.renderGL();
#endif
}

void vvSkipRayCaster::updateTransferFunction()
{
    std::vector<vec4> tf(256 * 1 * 1);
    vd->computeTFTexture(0, 256, 1, 1, reinterpret_cast<float*>(tf.data()));

    impl_->transfunc = cuda_texture<vec4, 1>(tf.size());
    impl_->transfunc.reset(tf.data());
    impl_->transfunc.set_address_mode(Clamp);
    impl_->transfunc.set_filter_mode(Nearest);

#if KDTREE
    texture_ref<vec4, 1> tf_ref(tf.size());
    tf_ref.reset(tf.data());
    tf_ref.set_address_mode(Clamp);
    tf_ref.set_filter_mode(Nearest);
#ifdef BUILD_TIMING
    vvStopwatch sw; sw.start();
#endif
    impl_->kdtree.updateTransfunc(tf_ref);
#ifdef BUILD_TIMING
    std::cout << "kdtree construction: " << sw.getTime() << " sec.\n";
#endif
#endif
}

void vvSkipRayCaster::updateVolumeData()
{
    vvRenderer::updateVolumeData();

#if KDTREE
    impl_->kdtree.updateVolume(*vd, 0);
#endif


    // Init GPU textures
    tex_filter_mode filter_mode = getParameter(VV_SLICEINT).asInt() == virvo::Linear ? Linear : Nearest;

    virvo::PixelFormat texture_format = virvo::PF_R8;

    impl_->volumes.resize(vd->frames);


    virvo::TextureUtil tu(vd);
    for (int f = 0; f < vd->frames; ++f)
    {
        virvo::TextureUtil::Pointer tex_data = nullptr;

        tex_data = tu.getTexture(virvo::vec3i(0),
            virvo::vec3i(vd->vox),
            texture_format,
            virvo::TextureUtil::All,
            f);

        impl_->volumes[f] = cuda_texture<unorm<8>, 3>(vd->vox[0], vd->vox[1], vd->vox[2]);
        impl_->volumes[f].reset(reinterpret_cast<unorm<8> const*>(tex_data));
        impl_->volumes[f].set_address_mode(Clamp);
        impl_->volumes[f].set_filter_mode(filter_mode);
    }
}

bool vvSkipRayCaster::checkParameter(vvRenderer::ParameterType param, vvParam const& value) const
{
    return true;
}

void vvSkipRayCaster::setParameter(vvRenderer::ParameterType param, const vvParam& value)
{
    switch (param)
    {
    case VV_SLICEINT:
        {
            if (_interpolation != static_cast< virvo::tex_filter_mode >(value.asInt()))
            {
                _interpolation = static_cast< virvo::tex_filter_mode >(value.asInt());
                tex_filter_mode filter_mode = _interpolation == virvo::Linear ? Linear : Nearest;

                for (auto& tex : impl_->volumes)
                {
                    tex.set_filter_mode(filter_mode);
                }
            }
        }
        break;

    default:
        vvRenderer::setParameter(param, value);
        break;
    }
}

bool vvSkipRayCaster::instantClassification() const
{
    return true;
}
