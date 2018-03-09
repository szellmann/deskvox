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

// Dylan -- added imports
#include <sstream>
#define DEBUG_CUDA 1
// End Dylan


#include <cassert>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>

#include <GL/glew.h>

#ifdef VV_ARCH_CUDA
#include <thrust/device_vector.h>
#endif

#undef MATH_NAMESPACE

#include <visionaray/detail/pixel_access.h> // detail (TODO?)!
#include <visionaray/math/math.h>
#include <visionaray/texture/texture.h>
#include <visionaray/aligned_vector.h>
#include <visionaray/material.h>
#include <visionaray/packet_traits.h>
#include <visionaray/pixel_format.h>
#include <visionaray/pixel_traits.h>
#include <visionaray/point_light.h>
#include <visionaray/render_target.h>
#include <visionaray/scheduler.h>
#include <visionaray/shade_record.h>
#include <visionaray/variant.h>

#ifdef VV_ARCH_CUDA
#include <visionaray/cuda/pixel_pack_buffer.h>
#endif

#undef MATH_NAMESPACE

#include "gl/util.h"
#include "vvcudarendertarget.h"
#include "vvraycaster.h"
#include "vvtextureutil.h"
#include "vvtoolshed.h"
#include "vvvoldesc.h"
#include "../../../../Program Files/NVIDIA GPU Computing Toolkit/CUDA/v9.1/include/cuda_runtime_api.h"
#include "cuda/texture.h"

#ifdef VV_ARCH_CUDA
#include "cuda/utils.h"
#endif

using namespace visionaray;


//-------------------------------------------------------------------------------------------------
// Global typedefs
//

#if defined(VV_ARCH_CUDA)
using ray_type          = basic_ray<float>;
using sched_type        = cuda_sched<ray_type>;
using transfunc_type    = cuda_texture<vec4,      1>;
using volume8_type      = cuda_texture<unorm< 8>, 3>;
using volume16_type     = cuda_texture<unorm<16>, 3>;
using volume32_type     = cuda_texture<float,     3>;
using pit_type          = cuda_texture<float, 3>;            // TODO: Create a volume of color values
using ext_vol_type      = cuda_texture<float, 3>;           // TODO: Utliize this type
#else
#if defined(VV_ARCH_SSE2) || defined(VV_ARCH_SSE4_1)
using ray_type = basic_ray<simd::float4>;
#elif defined(VV_ARCH_AVX) || defined(VV_ARCH_AVX2)
using ray_type = basic_ray<simd::float8>;
#else
using ray_type = basic_ray<float>;
#endif
using sched_type        = tiled_sched<ray_type>;
using transfunc_type    = texture<vec4,      1>;
using volume8_type      = texture<unorm< 8>, 3>;
using volume16_type     = texture<unorm<16>, 3>;
using volume32_type     = texture<float,     3>;
using pit_type          = texture<float, 3>;                 // TODO: Create a volume of color values
using ext_vol_type      = texture<float, 3>;                // TODO: Utliize this type
#endif

//-------------------------------------------------------------------------------------------------
// Ray type, depends upon target architecture
//

template <typename T>
struct SVT
{
    void reset(aabbi bbox);
    void reset(vvVolDesc const& vd, aabbi bbox, int channel = 0);

    template <typename Tex>
    void build(Tex transfunc, std::vector<vec4> values_as_vector);

    aabbi boundary(aabbi bbox) const;

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
//#if defined(VV_ARCH_CUDA)
//        return thrust::raw_pointer_cast(data_.data());
//#else
        return data_.data();
//#endif
        // return data_.data();
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

//#if defined(VV_ARCH_CUDA)
//    thrust::device_vector<float> voxels_;
//    thrust::device_vector<T> data_;
//#else
    std::vector<float> voxels_;
    std::vector<T> data_;
//#endif
    int width;
    int height;
    int depth;
};

template <typename T>
void SVT<T>::reset(aabbi bbox)
{
    data_.resize(bbox.size().x * bbox.size().y * bbox.size().z);
    width = bbox.size().x;
    height = bbox.size().y;
    depth = bbox.size().z;
}

template <typename T>
void SVT<T>::reset(vvVolDesc const& vd, aabbi bbox, int channel)
{
    voxels_.resize(bbox.size().x * bbox.size().y * bbox.size().z);
    data_.resize(bbox.size().x * bbox.size().y * bbox.size().z);
    width = bbox.size().x;
    height = bbox.size().y;
    depth = bbox.size().z;


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
void SVT<T>::build(Tex transfunc, std::vector<vec4> values_as_vector)
{
    // Apply transfer function
    for (int z = 0; z < depth; ++z)
    {
        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                size_t index = z * width * height + y * width + x;
                // My code:
                float coordinate = voxels_[index];

                /*
                vec4 color = tex1D(transfunc, coordinate);
                vec4 color1 = tex1D(transfunc, 0.f);
                vec4 color2 = tex1D(transfunc, 0.25f);
                vec4 color3 = tex1D(transfunc, 0.5f);
                vec4 color4 = tex1D(transfunc, 1.f);
                vec4 color5 = tex1D(transfunc, 125.f);
                float value = color.w;
                */

                // Texture holds garbage values -- use the values_as_vector
                coordinate *= values_as_vector.size() - 1;
                float first_weight = 1 - (coordinate - static_cast<int>(coordinate));
                float value = first_weight * values_as_vector[static_cast<int>(coordinate)].w + (1 - first_weight) * values_as_vector[min(static_cast<int>(coordinate) + 1, static_cast<int>(values_as_vector.size())-1)].w;
                at(x, y, z) = value;
                // End Dylan Code
                /*
                if (tex1D(transfunc, voxels_[index]).w < 0.0001)
                    at(x, y, z) = T(0);
                else
                    at(x, y, z) = T(1);
                */
            }
        }
    }


    // Build summed volume table

    // Init 0-border voxel
    //at(0, 0, 0) = at(0, 0, 0);

    // Init 0-border edges (prefix sum)
    for (int x = 1; x<width; ++x)
    {
        at(x, 0, 0) = at(x, 0, 0) + at(x - 1, 0, 0);
    }

    for (int y = 1; y<height; ++y)
    {
        at(0, y, 0) = at(0, y, 0) + at(0, y - 1, 0);
    }

    for (int z = 1; z<depth; ++z)
    {
        at(0, 0, z) = at(0, 0, z) + at(0, 0, z - 1);
    }


    // Init 0-border planes (summed-area tables)
    for (int y = 1; y<height; ++y)
    {
        for (int x = 1; x<width; ++x)
        {
            at(x, y, 0) = at(x, y, 0)
                + at(x - 1, y, 0) + at(x, y - 1, 0)
                - at(x - 1, y - 1, 0);
        }
    }

    for (int z = 1; z<depth; ++z)
    {
        for (int y = 1; y<height; ++y)
        {
            at(0, y, z) = at(0, y, z)
                + at(0, y - 1, z) + at(0, y, z - 1)
                - at(0, y - 1, z - 1);
        }
    }

    for (int x = 1; x<width; ++x)
    {
        for (int z = 1; z<depth; ++z)
        {
            at(x, 0, z) = at(x, 0, z)
                + at(x - 1, 0, z) + at(x, 0, z - 1)
                - at(x - 1, 0, z - 1);
        }
    }


    // Build up SVT
    for (int z = 1; z<depth; ++z)
    {
        for (int y = 1; y<height; ++y)
        {
            for (int x = 1; x<width; ++x)
            {
                at(x, y, z) = at(x, y, z) + at(x - 1, y - 1, z - 1)
                    + at(x - 1, y, z) - at(x, y - 1, z - 1)
                    + at(x, y - 1, z) - at(x - 1, y, z - 1)
                    + at(x, y, z - 1) - at(x - 1, y - 1, z);
            }
        }
    }
}



//-------------------------------------------------------------------------------------------------
// Misc. helpers
//

template <typename T, typename Tex>
VSNRAY_FUNC
inline vector<3, T> gradient(Tex const& tex, vector<3, T> tex_coord)
{
    vector<3, T> s1;
    vector<3, T> s2;

    float DELTA = 0.01f;

    s1.x = tex3D(tex, tex_coord + vector<3, T>(DELTA, 0.0f, 0.0f));
    s2.x = tex3D(tex, tex_coord - vector<3, T>(DELTA, 0.0f, 0.0f));
    // signs for y and z are swapped because of texture orientation
    s1.y = tex3D(tex, tex_coord - vector<3, T>(0.0f, DELTA, 0.0f));
    s2.y = tex3D(tex, tex_coord + vector<3, T>(0.0f, DELTA, 0.0f));
    s1.z = tex3D(tex, tex_coord - vector<3, T>(0.0f, 0.0f, DELTA));
    s2.z = tex3D(tex, tex_coord + vector<3, T>(0.0f, 0.0f, DELTA));

    return s2 - s1;
}

template <typename F, typename I>
VSNRAY_FUNC
inline F normalize_depth(I const& depth, pixel_format depth_format, F /* */)
{
    if (depth_format == PF_DEPTH24_STENCIL8)
    {
        auto d = (depth & 0xFFFFFF00) >> 8;
        return F(d) / 16777215.0f;
    }

    // Assume PF_DEPTH32F
    return reinterpret_as_float(depth);
}

template <typename I1, typename I2, typename Params>
VSNRAY_FUNC
inline void get_depth(I1 x, I1 y, I2& depth_raw, Params const& params)
{
    // Get depth value from visionaray buffer
    // dst format equals src format because our implementation
    // takes care of the conversion itself in the rendering kernel
    if (params.depth_format == PF_DEPTH24_STENCIL8)
    {
        detail::pixel_access::get( // detail (TODO?)!
                pixel_format_constant<PF_DEPTH24_STENCIL8>{},   // dst format
                pixel_format_constant<PF_DEPTH24_STENCIL8>{},   // src format
                x,
                y,
                params.viewport.w,
                params.viewport.h,
                depth_raw,
                params.depth_buffer
                );
    }
    else
    {
        // Assume PF_DEPTH32F
        detail::pixel_access::get( // detail (TODO?)!
                pixel_format_constant<PF_DEPTH32F>{},           // dst format
                pixel_format_constant<PF_DEPTH32F>{},           // src format
                x,
                y,
                params.viewport.w,
                params.viewport.h,
                depth_raw,
                params.depth_buffer
                );
    }
}

VSNRAY_FUNC
inline vec3 gatherv(vec3 const* base_addr, int index)
{
    return base_addr[index];
}

template <
    typename T,
    typename I,
    typename = typename std::enable_if<simd::is_simd_vector<T>::value>::type
    >
VSNRAY_CPU_FUNC
inline vector<3, T> gatherv(vector<3, T> const* base_addr, I const& index)
{
    // basically like visionaray::simd::gather, but
    // base_addr points to vec3's of simd-vectors

    typename simd::aligned_array<I>::type indices;
    store(indices, index);

    array<vector<3, float>, simd::num_elements<T>::value> arr;

    for (int i = 0; i < simd::num_elements<T>::value; ++i)
    {
        auto vecs = unpack(base_addr[indices[i]]);
        arr[i] = vecs[i];
    }

    return simd::pack(arr);
}


//-------------------------------------------------------------------------------------------------
// Clip sphere, hit_record stores both tnear and tfar (in contrast to basic_sphere)!
//

struct clip_sphere : basic_sphere<float>
{
};

template <typename T>
struct clip_sphere_hit_record
{
    using M = typename simd::mask_type<T>::type;

    M hit   = M(false);
    T tnear =  numeric_limits<T>::max();
    T tfar  = -numeric_limits<T>::max();
};

template <typename T>
VSNRAY_FUNC
inline clip_sphere_hit_record<T> intersect(basic_ray<T> const& ray, clip_sphere const& sphere)
{

    typedef basic_ray<T> ray_type;
    typedef vector<3, T> vec_type;

    ray_type r = ray;
    r.ori -= vec_type( sphere.center );

    auto A = dot(r.dir, r.dir);
    auto B = dot(r.dir, r.ori) * T(2.0);
    auto C = dot(r.ori, r.ori) - sphere.radius * sphere.radius;

    // solve Ax**2 + Bx + C
    auto disc = B * B - T(4.0) * A * C;
    auto valid = disc >= T(0.0);

    auto root_disc = select(valid, sqrt(disc), disc);

    auto q = select( B < T(0.0), T(-0.5) * (B - root_disc), T(-0.5) * (B + root_disc) );

    auto t1 = q / A;
    auto t2 = C / q;

    clip_sphere_hit_record<T> result;
    result.hit = valid;
    result.tnear   = select( valid, select( t1 > t2, t2, t1 ), T(-1.0) );
    result.tfar    = select( valid, select( t1 > t2, t1, t2 ), T(-1.0) );
    return result;
}


//-------------------------------------------------------------------------------------------------
// Clip clone
//

struct clip_cone
{
    vec3 tip;       // position of the cone's tip
    vec3 axis;      // unit vector pointing from tip into the cone
    float theta;    // *half* angle between axis and cone surface
};

template <typename T>
struct clip_cone_hit_record : clip_sphere_hit_record<T>
{
};

template <typename T>
VSNRAY_FUNC
inline clip_cone_hit_record<T> intersect(basic_ray<T> const& ray, clip_cone const& cone)
{
    using R = basic_ray<T>;
    using V = vector<3, T>;

    R r = ray;
    r.ori -= V(cone.tip);

    T cos2_theta(cos(cone.theta) * cos(cone.theta));

    auto A = dot(r.dir, V(cone.axis)) * dot(r.dir, V(cone.axis)) - cos2_theta;
    auto B = T(2.0) * (dot(r.dir, V(cone.axis)) * dot(r.ori, V(cone.axis)) - dot(r.dir, r.ori) * cos2_theta);
    auto C = dot(r.ori, V(cone.axis)) * dot(r.ori, V(cone.axis)) - dot(r.ori, r.ori) * cos2_theta;

    // solve Ax**2 + Bx + C
    auto disc = B * B - T(4.0) * A * C;
    auto valid = disc >= T(0.0);

    auto root_disc = select(valid, sqrt(disc), disc);

    auto q = select( B < T(0.0), T(-0.5) * (B - root_disc), T(-0.5) * (B + root_disc) );

    auto t1 = q / A;
    auto t2 = C / q;

    auto isect_pos1 = V(ray.ori) + V(ray.dir) * t1;
    auto hits_shadow_cone1 = dot(isect_pos1 - V(cone.tip), V(cone.axis)) > T(0.0);

    auto isect_pos2 = V(ray.ori) + V(ray.dir) * t2;
    auto hits_shadow_cone2 = dot(isect_pos2 - V(cone.tip), V(cone.axis)) > T(0.0);

    t1 = select(hits_shadow_cone1, T(-1.0), t1);
    t2 = select(hits_shadow_cone2, T(-1.0), t2);

    valid &= dot(ray.dir, V(cone.axis)) >= T(0.0);

    clip_cone_hit_record<T> result;
    result.hit = valid;
    result.tnear   = select( valid, select( t1 > t2, t2, t1 ), T(-1.0) );
    result.tfar    = select( valid, select( t1 > t2, t1, t2 ), T(-1.0) );
    return result;
}


//-------------------------------------------------------------------------------------------------
// Clip box, basically an aabb, but intersect() returns a hit record containing the
// plane normal of the box' side where the ray entered
//

struct clip_box : basic_aabb<float>
{
    using base_type = basic_aabb<float>;

    clip_box() = default;
    VSNRAY_FUNC clip_box(vec3 const& min, vec3 const& max)
        : base_type(min, max)
    {
    }
};

template <typename T>
struct clip_box_hit_record : hit_record<basic_ray<T>, basic_aabb<float>>
{
    vector<3, T> normal;
};

template <typename T>
VSNRAY_FUNC
inline clip_box_hit_record<T> intersect(basic_ray<T> const& ray, clip_box const& box)
{
    auto hr = intersect(ray, static_cast<clip_box::base_type>(box));

    // calculate normal
    vector<3, float> normals[6] {
            {  1.0f,  0.0f,  0.0f },
            { -1.0f,  0.0f,  0.0f },
            {  0.0f,  1.0f,  0.0f },
            {  0.0f, -1.0f,  0.0f },
            {  0.0f,  0.0f,  1.0f },
            {  0.0f,  0.0f, -1.0f }
            };

    auto isect_pos = ray.ori + ray.dir * hr.tnear;
    auto dir = normalize(isect_pos - vector<3, T>(box.center()));
    auto cosa = dot(dir, vector<3, T>(normals[0]));

    vector<3, T> normal(normals[0]);

    for (int i = 1; i < 6; ++i)
    {
        T dp    = dot(dir, vector<3, T>(normals[i]));
        normal  = select(dp > cosa, normals[i], normal);
        cosa    = select(dp > cosa, dp, cosa);
    }

    clip_box_hit_record<T> result;
    result.hit    = hr.hit;
    result.tnear  = hr.tnear;
    result.tfar   = hr.tfar;
    result.normal = normal;
    return result;
}


//-------------------------------------------------------------------------------------------------
// Clip plane (just another name for plane)
//

using clip_plane = basic_plane<3, float>;


//-------------------------------------------------------------------------------------------------
// Create clip intervals and deduce clip normals from primitive list
//

template <typename T>
struct clip_object_visitor
{
public:

    enum { MAX_INTERVALS = 64 };

    struct RT
    {
        int num_intervals;
        vector<2, T> intervals[MAX_INTERVALS];
        vector<3, T> normal;
    };

    using return_type = RT;

public:

    // Create with ray and tnear / tfar obtained from ray / bbox intersection
    VSNRAY_FUNC
    clip_object_visitor(basic_ray<T> const& ray, T const& tnear, T const& tfar)
        : ray_(ray)
        , tnear_(tnear)
        , tfar_(tfar)
    {
    }

    // Clip plane
    VSNRAY_FUNC
    return_type operator()(clip_plane const& ref) const
    {
        auto hit_rec = intersect(ray_, ref);
        auto ndotd = dot(ray_.dir, vector<3, T>(ref.normal));

        return_type result;
        result.num_intervals = 1;
        result.intervals[0].x = select(ndotd >  0.0f, hit_rec.t, tnear_);
        result.intervals[0].y = select(ndotd <= 0.0f, hit_rec.t, tfar_);
        result.normal     = ref.normal;
        return result;
    }

    // Clip sphere
    VSNRAY_FUNC
    return_type operator()(clip_sphere const& ref) const
    {
        using V = vector<3, T>;

        auto hit_rec = intersect(ray_, ref);

        return_type result;
        result.num_intervals = 1;
        result.intervals[0].x = select(hit_rec.tnear > tnear_, hit_rec.tnear, tnear_);
        result.intervals[0].y = select(hit_rec.tfar  < tfar_,  hit_rec.tfar,  tfar_);

        // normal at tfar, pointing inwards
        V isect_pos = ray_.ori + result.intervals[0].y * ray_.dir;
        result.normal = -(isect_pos - V(ref.center)) / T(ref.radius);

        return result;
    }

    // Clip cone
    VSNRAY_FUNC
    return_type operator()(clip_cone const& ref) const
    {
        using V = vector<3, T>;

        auto hit_rec = intersect(ray_, ref);

        return_type result;
        result.num_intervals = 1;
        result.intervals[0].x = select(hit_rec.tnear > tnear_, hit_rec.tnear, tnear_);
        result.intervals[0].y = select(hit_rec.tfar  < tfar_,  hit_rec.tfar,  tfar_);

        // normal at tfar, pointing inwards
        V isect_pos = ray_.ori + result.intervals[0].y * ray_.dir;
        V tmp = isect_pos - V(ref.tip);
        result.normal = normalize(tmp * dot(V(ref.axis), tmp) / dot(tmp, tmp) - V(ref.axis));

        return result;
    }

private:

    basic_ray<T>    ray_;
    T               tnear_;
    T               tfar_;
};


//-------------------------------------------------------------------------------------------------
// Wrapper that either uses CUDA/GL interop or simple CPU <- GPU transfer to make the
// OpenGL depth buffer available to the Visionaray kernel
//

#ifdef VV_ARCH_CUDA

struct depth_buffer_type : cuda::pixel_pack_buffer
{
    unsigned const* data() const
    {
        return static_cast<unsigned const*>(cuda::pixel_pack_buffer::data());
    }
};

#else

struct depth_buffer_type
{
    void map(recti viewport, pixel_format format)
    {
        auto info = map_pixel_format(format);

        buffer.resize((viewport.w - viewport.x) * (viewport.h - viewport.y));

        glReadPixels(
                viewport.x,
                viewport.y,
                viewport.w,
                viewport.h,
                info.format,
                info.type,
                buffer.data()
                );
    }

    void unmap()
    {
    }

    unsigned const* data() const
    {
        return buffer.data();
    }

    aligned_vector<unsigned> buffer;
};

#endif

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
// Volume kernel params
//

struct volume_kernel_params
{
    enum projection_mode
    {
        AlphaCompositing,
        MaxIntensity,
        MinIntensity,
        DRR,
        ShowExtinction,
        ShowPit,
    };

    using clip_object    = variant<clip_plane, clip_sphere, clip_cone>;
    using transfunc_ref  = typename transfunc_type::ref_type;
    using pit_type_ref   = typename pit_type::ref_type;
    using ext_vol_type_ref = typename ext_vol_type::ref_type;

    clip_box                    bbox;
    float                       delta;
    int                         num_channels;
    transfunc_ref const*        transfuncs;
    vec2 const*                 ranges;
    unsigned const*             depth_buffer;
    pixel_format                depth_format;
    projection_mode             mode;
    bool                        depth_test;
    bool                        opacity_correction;
    bool                        early_ray_termination;
    bool                        local_shading;
    mat4                        camera_matrix_inv;
    recti                       viewport;
    point_light<float>          light;
    pit_type_ref                pit;
    //SVT<float> const*           extinction_volume;
    ext_vol_type_ref            extinction_volume;
    float                       ambient_radius;
    float                       albedo;
    float                       anisotropy;

    struct
    {
        clip_object const*      begin;
        clip_object const*      end;
    } clip_objects;
};


//-------------------------------------------------------------------------------------------------
// Visionaray volume rendering kernel
//

template <typename Volume, typename Volume32>
struct volume_kernel
{
    using Params = volume_kernel_params;
    using VolRef = typename Volume::ref_type;
    using Vol32Ref = typename Volume::ref_type;

    VSNRAY_FUNC
    explicit volume_kernel(Params const& p, VolRef const* vols)
        : params(p)
        , volumes(vols)
    {
    }

    template <typename R>
    VSNRAY_FUNC
    result_record<typename R::scalar_type> operator()(R ray, int x, int y) const
    {
        using S    = typename R::scalar_type;
        using I    = typename simd::int_type<S>::type;
        using Mask = typename simd::mask_type<S>::type;
        using Mat4 = matrix<4, 4, S>;
        using C    = vector<4, S>;

        // Dylan -- Additional Variables -- ALGO Preconditions (line 1)
        const float SAMPLING_PLACEHOLDER = 0.7f;
        const int VOLUME_SIZE = pow(2 * params.ambient_radius + 1, 3);
        float mean_extinction;
        S T = 1.f;
        S tau = 0.f;
        C L_out;
        auto L_d_intermediate = params.light.intensity(params.light.position());
        auto L_d = C(
            L_d_intermediate.x,
            L_d_intermediate.y,
            L_d_intermediate.z,
            1.0f
            );
        // End Dylan

        result_record<S> result;
        result.color = C(0.0);

        auto hit_rec = intersect(ray, params.bbox);
        auto tmax = hit_rec.tfar;

        // convert depth buffer(x,y) to "t" coordinates
        if (params.depth_test)
        {
            // unproject (win to obj)
            I depth_raw(0);
            get_depth(x, y, depth_raw, params);
            S depth = normalize_depth(depth_raw, params.depth_format, S{});

            vector<3, S> win(expand_pixel<S>().x(x), expand_pixel<S>().y(y), depth);
            vector<4, S> u(
                    S(2.0 * (win[0] - params.viewport[0]) / params.viewport[2] - 1.0),
                    S(2.0 * (win[1] - params.viewport[1]) / params.viewport[3] - 1.0),
                    S(2.0 * win[2] - 1.0),
                    S(1.0)
                    );

            vector<4, S> v = Mat4(params.camera_matrix_inv) * u;
            vector<3, S> obj = v.xyz() / v.w;

            // convert to "t" coordinates
            tmax = length(obj - ray.ori);
        }


        auto t = max(S(0.0f), hit_rec.tnear);
        tmax = min(hit_rec.tfar, tmax);


        // calculate intervals clipped by planes, spheres, etc., along with the
        // normals of the farthest intersection in view direction
        const int MaxClipIntervals = 64;
        vector<2, S> clip_intervals[MaxClipIntervals];
        vector<3, S> clip_normals[MaxClipIntervals];

        auto num_clip_objects = min(
                MaxClipIntervals - 1, // room for bbox normal, which is the last clip object
                static_cast<int>(params.clip_objects.end - params.clip_objects.begin)
                );

        for (auto it = params.clip_objects.begin; it != params.clip_objects.end; ++it)
        {
            clip_object_visitor<S> visitor(ray, t, tmax);
            auto clip_data = apply_visitor(visitor, *it);

            clip_intervals[it - params.clip_objects.begin] = clip_data.intervals[0];
            clip_normals[it - params.clip_objects.begin] = clip_data.normal;
        }

        // treat the bbox entry plane as a clip
        // object that contributes a shading normal
        clip_normals[num_clip_objects] = hit_rec.normal;


        // calculate the volume rendering integral
        while (visionaray::any(t < tmax))
        {
            Mask clipped(false);

            S tnext = t + params.delta;
            for (int i = 0; i < num_clip_objects; ++i)
            {
                clipped |= t >= clip_intervals[i].x && t <= clip_intervals[i].y;
                tnext = select(
                        t >= clip_intervals[i].x && t <= clip_intervals[i].y && tnext < clip_intervals[i].y,
                        clip_intervals[i].y,
                        tnext
                        );
            }

            if (!visionaray::all(clipped))
            {
                auto pos = ray.ori + ray.dir * t;
                auto tex_coord = vector<3, S>(
                        ( pos.x + (params.bbox.size().x / 2) ) / params.bbox.size().x,
                        (-pos.y + (params.bbox.size().y / 2) ) / params.bbox.size().y,
                        (-pos.z + (params.bbox.size().z / 2) ) / params.bbox.size().z
                        );

                C color(0.0);                                                               // ALGO line 10

                for (int i = 0; i < params.num_channels; ++i)                               // THIS LOOP CAN BE CONSIDERED TO HAVE ITERATION OF 1
                {
                    S voxel = tex3D(volumes[i], tex_coord);                                // ALGO line 5/6
                    C colori = tex1D(params.transfuncs[i], voxel);                          // ALGO line 7 -- Note: extinction coefficient is colori.w

                    // TODO: Sample Ambient Exctinction Volume      --      ALGO: line 9 -- sigma_hat
                    
                    float max_x = tex_coord.x + params.ambient_radius / params.bbox.size().x;
                    float min_x = tex_coord.x - params.ambient_radius / params.bbox.size().x;
                    float max_y = tex_coord.y + params.ambient_radius / params.bbox.size().y;
                    float min_y = tex_coord.y - params.ambient_radius / params.bbox.size().y;
                    float max_z = tex_coord.z + params.ambient_radius / params.bbox.size().z;
                    float min_z = tex_coord.z - params.ambient_radius / params.bbox.size().z;

                    float XYZ_corner = tex3D(params.extinction_volume, vec3(max_x, max_y, max_z));
                    float XYz_corner = tex3D(params.extinction_volume, vec3(max_x, max_y, min_z));
                    float XyZ_corner = tex3D(params.extinction_volume, vec3(max_x, min_y, max_z));
                    float xYZ_corner = tex3D(params.extinction_volume, vec3(min_x, max_y, max_z));
                    float Xyz_corner = tex3D(params.extinction_volume, vec3(max_x, min_y, min_z));
                    float xYz_corner = tex3D(params.extinction_volume, vec3(min_x, max_y, min_z));
                    float xyZ_corner = tex3D(params.extinction_volume, vec3(min_x, min_y, max_z));
                    float xyz_corner = tex3D(params.extinction_volume, vec3(min_x, min_y, min_z));

                    mean_extinction = XYZ_corner -
                                            XYz_corner -
                                            XyZ_corner - 
                                            xYZ_corner +
                                            xyZ_corner +
                                            Xyz_corner +
                                            xYz_corner -
                                            xyz_corner;
                    mean_extinction /= VOLUME_SIZE;

                    // TODO: DELETE THIS
                    //mean_extinction = tex_coord.x;
                    //mean_extinction = SAMPLING_PLACEHOLDER;
                    //float mean_extinction = params.extinction_volume->at(tex_coord.x, tex_coord.y, tex_coord.z);                           // ALGO line 9
                    //float mean_extinction = params.extinction_volume->get_count(aabb(tex_coord + params.ambient_radius, tex_coord - params.ambient_radius));                           // ALGO line 9

                    auto do_shade = params.local_shading && colori.w >= 0.1f;

                    if (visionaray::any(do_shade))
                    {
                        // TODO: make this modifiable
                        plastic<S> mat;
                        mat.ca() = from_rgb(vector<3, S>(0.3f, 0.3f, 0.3f));
                        mat.cd() = from_rgb(vector<3, S>(0.8f, 0.8f, 0.8f));
                        mat.cs() = from_rgb(vector<3, S>(0.8f, 0.8f, 0.8f));
                        mat.ka() = 1.0f;
                        mat.kd() = 1.0f;
                        mat.ks() = 1.0f;
                        mat.specular_exp() = 1000.0f;

                        // calculate shading
                        auto grad = gradient(volumes[i], tex_coord);
                        auto normal = normalize(grad);
                        
                        auto float_eq = [&](S const& a, S const& b) { return abs(a - b) < params.delta * S(0.5); };

                        Mask at_boundary = float_eq(t, hit_rec.tnear);
                        I clip_normal_index = select(
                                at_boundary,
                                I(num_clip_objects), // bbox normal is stored at last position in the list
                                I(0)
                                );

                        for (int i = 0; i < num_clip_objects; ++i)
                        {
                            Mask hit = float_eq(t, clip_intervals[i].y + params.delta); // TODO: understand why +delta
                            clip_normal_index = select(hit, I(i), clip_normal_index);
                            at_boundary |= hit;
                        }

                        if (visionaray::any(at_boundary))
                        {
                            auto boundary_normal = gatherv(clip_normals, clip_normal_index);
                            normal = select(
                                    at_boundary,
                                    boundary_normal * colori.w + normal * (S(1.0) - colori.w),
                                    normal
                                    );
                        }

                        do_shade &= length(grad) != 0.0f;

                        shade_record<S> sr;
                        sr.normal = normal;
                        sr.geometric_normal = normal;
                        sr.view_dir = -ray.dir;
                        sr.tex_color = vector<3, S>(1.0);
                        sr.light_dir = normalize(params.light.position());
                        sr.light_intensity = params.light.intensity(pos);

                        auto shaded_clr = mat.shade(sr);

                        // TODO: Consider whether this line needs to be commented out
                        colori.xyz() = mul(
                                colori.xyz(),
                                to_rgb(shaded_clr),
                                do_shade,
                                colori.xyz()
                                );



                        // Dylan
                        
                        // TODO: sample preint. table and radiance cache       ALGO: lines 15 and 16
                        
                        // Handle lighting from light source and sample cached info
                        auto position_delta = vector<3, S>(
                            pos.x - params.light.position().x,
                            pos.y - params.light.position().y,
                            pos.z - params.light.position().z
                            );
                        auto light_dir = normalize(position_delta);                             // ALGO line 12

                        auto scaled_light_direction = vector<3, S>(
                            params.ambient_radius * light_dir.x,
                            params.ambient_radius * light_dir.y,
                            params.ambient_radius * light_dir.z
                            );

                        S angle = dot(ray.dir, light_dir);
                        S theta_j = acos(angle);                                                    // ALGO line 14

                        vector<3, S> pit_coordinate = vector<3, S>(
                            params.anisotropy,
                            theta_j, 
                            params.ambient_radius * mean_extinction
                            );
                        C L_out(tex3D(params.pit, pit_coordinate));                                // ALGO line 15
                        
                        
                        colori += C(
                            L_out.x * L_d.x,
                            L_out.y * L_d.y,
                            L_out.z * L_d.z,
                            L_out.w * L_d.w
                            );                                                                      // ALGO line 17

                        // Update L_d
                        pit_coordinate = vector<3, S>(
                            params.anisotropy,
                            0.f,
                            static_cast<float>(params.ambient_radius * mean_extinction)
                            );
                        C L_out_for_d (tex3D(params.pit, pit_coordinate));
                        L_d = C(
                            L_out_for_d.x * L_d.x,
                            L_out_for_d.y * L_d.y,
                            L_out_for_d.z * L_d.z,
                            L_out_for_d.w * L_d.w
                            );                                                                      // ALGO line 16
                        // End Dylan
                    }

                    // Conglomerate info and step
                    const S scattering_coefficient = (S(1.0f) - result.color.w) * S(params.albedo);                   // ALGO line 8
                    colori.w = scattering_coefficient * params.delta * T;                                           // ALGO line 19 Part A

                    /* Not optional
                    if (params.opacity_correction)
                    {
                        colori.w = 1.0f - pow(1.0f - colori.w, params.delta);
                    }*/

                    // premultiplied alpha
                    colori.xyz() *= colori.w;
                    color += colori;                                                                // ALGO line 19 Part B

                    auto pos_not = vector<3, S>(
                        pos.x + params.ambient_radius * ray.dir.x,
                        pos.y + params.ambient_radius * ray.dir.y,
                        pos.z + params.ambient_radius * ray.dir.z
                        );                                                                          // pos_not emulates the center of the mesoscopic sphere
                    auto tex_coord_not = vector<3, S>(
                        (pos_not.x + (params.bbox.size().x / 2)) / params.bbox.size().x,
                        (-pos_not.y + (params.bbox.size().y / 2)) / params.bbox.size().y,
                        (-pos_not.z + (params.bbox.size().z / 2)) / params.bbox.size().z
                        );
                    S voxel_not = tex3D(volumes[i], tex_coord_not);
                    C color_not = tex1D(params.transfuncs[i], voxel_not);                           // ALGO line 20


                    tau += color_not.w * S(params.delta);                              // ALGO line 21
                    T = exp(-tau);                                                  // ALGO line 22
                }

                // compositing
                if (params.mode == Params::AlphaCompositing)
                {
                    result.color += select(
                            t < tmax && !clipped,
                            color * (1.0f - result.color.w),
                            C(0.0)
                            );

                    // early-ray termination - don't traverse w/o a contribution
                    if (params.early_ray_termination && visionaray::all(result.color.w >= 0.999f))
                    {
                        break;
                    }
                }
                else if (params.mode == Params::MaxIntensity)
                {
                    result.color = select(
                            t < tmax && !clipped,
                            max(color, result.color),
                            result.color
                            );
                }
                else if (params.mode == Params::MinIntensity)
                {
                    result.color = select(
                            t < tmax && !clipped,
                            min(color, result.color),
                            result.color
                            );
                }
                else if (params.mode == Params::DRR)
                {
                    result.color += select(
                            t < tmax && !clipped,
                            color,
                            C(0.0)
                            );
                }
                else if (params.mode == Params::ShowExtinction)
                {
                    result.color = C(result.color.x + mean_extinction, result.color.x + mean_extinction, result.color.x + mean_extinction, 1.0);
                    //result.color = C(1.f, 1.f, 1.f, result.color.w + mean_extinction);
                }

                else if (params.mode == Params::ShowPit)
                {
                    //result.color = C(0.0, 0.0, 0.0, 1.0);
                    result.color += L_out;
                }
            }
            // step on
            t = tnext;                                          // ALGO line 23
        }
        if (params.mode == Params::ShowExtinction)
        {
            // Invert white and black (white means highly ambient and black means heavily dense
            result.color = C(1 - result.color.x, 1 - result.color.y, 1 - result.color.z, 1.0);
        }
        result.hit = hit_rec.hit;
        return result;
    }

    Params params;
    VolRef const* volumes;
};


//-------------------------------------------------------------------------------------------------
// Private implementation
//

struct vvRayCaster::Impl
{
#if defined(VV_ARCH_CUDA)
    Impl()
        : sched(8, 8)
    {
    }
#else
    Impl()
        : sched(vvToolshed::getNumProcessors())
    {
        char* num_threads = getenv("VV_NUM_THREADS");
        if (num_threads != nullptr)
        {
            std::string str(num_threads);
            sched.reset(std::stoi(str));
        }
    }
#endif

    using params_type = volume_kernel_params;

    sched_type                      sched;
    params_type                     params;
    pit_type                        pit;
    SVT<float>                      extinction_volume_svt;
    ext_vol_type                    extinction_volume_texture;
    //ext_vol_type                  extinction_volume_svt;
    std::vector<volume8_type>       volumes8;
    std::vector<volume16_type>      volumes16;
    std::vector<volume32_type>      volumes32;
    //transfunc_cpu_type            cpu_transfunc;
    std::vector<vec4>              transfunc_values;
    std::vector<transfunc_type>     transfuncs;
    depth_buffer_type               depth_buffer;

    // Internal storage format for textures
    virvo::PixelFormat              texture_format = virvo::PF_R8;

    void loadPreintegrationTable(float albedo);
    void updateVolumeTextures(vvVolDesc* vd, vvRenderer* renderer);
    void updateTransfuncTexture(vvVolDesc* vd, vvRenderer* renderer);

    template <typename Volumes>
    void updateVolumeTexturesImpl(vvVolDesc* vd, vvRenderer* renderer, Volumes& volume);
};

void vvRayCaster::Impl::updateVolumeTextures(vvVolDesc* vd, vvRenderer* renderer)
{
    if (texture_format == virvo::PF_R8)
    {
        updateVolumeTexturesImpl(vd, renderer, volumes8);
    }
    else if (texture_format == virvo::PF_R16UI)
    {
        updateVolumeTexturesImpl(vd, renderer, volumes16);
    }
    else if (texture_format == virvo::PF_R32F)
    {
        updateVolumeTexturesImpl(vd, renderer, volumes32);
    }
}

void vvRayCaster::Impl::updateTransfuncTexture(vvVolDesc* vd, vvRenderer* /*renderer*/)
{
    transfuncs.resize(vd->tf.size());
    for (size_t i = 0; i < vd->tf.size(); ++i)
    {
        aligned_vector<vec4> tf(256 * 1 * 1);
        vd->computeTFTexture(i, 256, 1, 1, reinterpret_cast<float*>(tf.data()));
        
        transfuncs[i] = transfunc_type(tf.size());
        transfuncs[i].reset(tf.data());
        transfuncs[i].set_address_mode(Clamp);
        transfuncs[i].set_filter_mode(Nearest);
        
        transfunc_values.resize(tf.size());
        for (int i = 0; i < tf.size(); i++)
        {
            transfunc_values[i] = tf[i];
        }
        /*
        if (i == 0)
        {
            cpu_transfunc = transfunc_cpu_type(tf.size());
            cpu_transfunc.reset(tf.data());
            cpu_transfunc.set_address_mode(Clamp);
            cpu_transfunc.set_filter_mode(Nearest);
        }*/
    }
    

    /*  THESE METHODS NEED TO BE DONE ON TEXTURE/CUDA_TEXTURE OBJECTS
    params.extinction_volume.reset(extinction_volume_svt.data());
    params.extinction_volume.set_address_mode(Clamp);
    params.extinction_volume.set_filter_mode(Nearest);
    */
}

template <typename Volumes>
void vvRayCaster::Impl::updateVolumeTexturesImpl(vvVolDesc* vd, vvRenderer* renderer, Volumes& volumes)
{
    using Volume = typename Volumes::value_type;

    tex_filter_mode filter_mode = renderer->getParameter(vvRenderer::VV_SLICEINT).asInt() == virvo::Linear ? Linear : Nearest;
    tex_address_mode address_mode = Clamp;

    volumes.resize(vd->frames * vd->getChan());

    virvo::TextureUtil tu(vd);
    for (size_t f = 0; f < vd->frames; ++f)
    {
        for (int c = 0; c < vd->getChan(); ++c)
        {
            virvo::TextureUtil::Pointer tex_data = nullptr;

            virvo::TextureUtil::Channels channelbits = 1ULL << c;

            tex_data = tu.getTexture(virvo::vec3i(0),
                virvo::vec3i(vd->vox),
                texture_format,
                channelbits,
                f);

            size_t index = f * vd->getChan() + c;

            volumes[index] = Volume(vd->vox[0], vd->vox[1], vd->vox[2]);
            volumes[index].reset(reinterpret_cast<typename Volume::value_type const*>(tex_data));
            volumes[index].set_address_mode(address_mode);
            volumes[index].set_filter_mode(filter_mode);
        }
    }
}


//-------------------------------------------------------------------------------------------------
// Load Preintegration Table
//

void vvRayCaster::Impl::loadPreintegrationTable(float albedo)
{
    std::vector<float> pit_data;
    const int albedo_flag = static_cast<int>(albedo * 10);
    FILE* fp;
    int			m_sigmat_res;			// resolution of ambient extinction coefficient 
    int			m_theta_res;			// resolution of angular bins
    for (int i = 1; i < 20; i++)
    {
        std::stringstream stream;
        stream << "E:/Research Workspace/research resources/AVSTableAlbedo0"<< albedo_flag <<"/AVSTable2D_a0" << albedo_flag << "_g";
        if (i < 10) {
            stream << '-';
        }

        stream << "0" << abs(10 - i) << ".tab";
        std::string filename = stream.str();
        fp = fopen(filename.c_str(), "rb");

        if (fp == nullptr) {
            std::cout << "Could not load from file: " << filename << std::endl;
            throw;
        }


        float		m_a;					// albedo
        float		m_g;					// anisotropy parameter for Henyey-Greenstein phase function
        int			m_N;					// number of path samples
        
        float		m_sigmat_min;			// minimum value of ambient extinction coefficient
        float		m_sigmat_max;			// maximum value of ambient extinction coefficient
        float		m_sphere_r;				// radius of sphere 
        float*		m_L;					// radiance data (theta, sigmat)

                                            // table parameters


        fread(&m_a, sizeof(float), 1, fp);
        fread(&m_g, sizeof(float), 1, fp);
        fread(&m_N, sizeof(int), 1, fp);
        fread(&m_sigmat_res, sizeof(int), 1, fp);
        fread(&m_theta_res, sizeof(int), 1, fp);
        fread(&m_sigmat_min, sizeof(float), 1, fp);
        fread(&m_sigmat_max, sizeof(float), 1, fp);
        fread(&m_sphere_r, sizeof(float), 1, fp);

        // radiance data
        // m_L = new float[m_theta_res * m_sigmat_res];
        size_t old_size = pit_data.size();
        pit_data.resize(pit_data.size() + m_theta_res * m_sigmat_res);
        fread(pit_data.data() + old_size, sizeof(float), m_sigmat_res * m_theta_res, fp);      // NOTE: Added * 3 term for color channels

        
        // TODO: Figure out how to fill the volume with the preintegration table
        // m_L represents a single layer of values
        /*
        for (int i = 0; i < m_theta_res * m_sigmat_res; i++)
        {
            //vec4 color(m_L[i], 1.0f);
            //pit_data.push_back(m_L[i]);
        }*/
        //delete[] m_L;
        fclose(fp);
    }
#if DEBUG_CUDA
    std::cerr << __LINE__ << ' ' << cudaGetErrorString(cudaGetLastError()) << '\n';
#endif

    pit = pit_type(19, m_theta_res, m_sigmat_res);
    pit.reset(&pit_data[0]);
    pit.set_address_mode(Clamp);
    pit.set_filter_mode(Nearest);
}

float* createExtinctionVolume(vvVolDesc* vd, transfunc_type transfunc, int radius)
{
    const int volume = pow(radius * 2 + 1, 1);
    std::vector<std::vector<std::vector<float>>> table;

    // table resizing
    table.resize(vd->vox[0]);
    for (int i = 0; i < vd->vox[0]; i++)
    {
        table[i].resize(vd->vox[1]);

        for (int j = 0; j < vd->vox[1]; j++)
        {
            table[i][j].resize(vd->vox[2]);
        }
    }

    for (int i = 0; i < vd->vox[0]; i++)
    {
        for (int j = 0; j < vd->vox[1]; j++)
        {
            for (int k = 0; k < vd->vox[2]; k++)
            {
                // regions
                for (int i_delta = -radius; i_delta <= radius; i_delta++)
                {
                    int neighbor_i = i + i_delta;
                    neighbor_i = max(neighbor_i, 0);
                    neighbor_i = min(neighbor_i, static_cast<int>(vd->vox[0]) - 1);
                    for (int j_delta = -radius; j_delta <= radius; j_delta++)
                    {
                        int neighbor_j = j + j_delta;
                        neighbor_j = max(neighbor_j, 0);
                        neighbor_j = min(neighbor_j, static_cast<int>(vd->vox[1]) - 1);
                        for (int k_delta = -radius; k_delta <= radius; k_delta++)
                        {
                            int neighbor_k = k + k_delta;
                            neighbor_k = max(neighbor_k, 0);
                            neighbor_k = min(neighbor_k, static_cast<int>(vd->vox[2]) - 1);

                            // Sum mean neighbor
                            // SO CLOSE!!!
                            // TODO: This seems wrong...

                            table[i][j][k] += vd->getChannelValue(0, neighbor_i, neighbor_j, neighbor_k, 0) / volume;
                            //table[i][j][k] += transfunc(vd->getChannelValue(0, neighbor_i, neighbor_j, neighbor_k, 0)) / volume;
                            //table[i][j][k] += ((*vd)(0, neighbor_i, neighbor_j, neighbor_k)).w / volume;
                            //table[i][j][k] += transfunc((*vd)(0, neighbor_i, neighbor_j, neighbor_k)).w / volume;
                        }
                    }
                }
            }
        }
    }
    return &(table[0][0][0]);
}

//-------------------------------------------------------------------------------------------------
// Public interface
//

vvRayCaster::vvRayCaster(vvVolDesc* vd, vvRenderState renderState)
    : vvRenderer(vd, renderState)
    , impl_(new Impl)
{
    rendererType = RAYREND;

    glewInit();

#if defined(VV_ARCH_CUDA)
    virvo::cuda::initGlInterop();

    virvo::RenderTarget* rt = virvo::PixelUnpackBufferRT::create(virvo::PF_RGBA32F, virvo::PF_UNSPECIFIED);

    // no direct rendering
    if (rt == NULL)
    {
        rt = virvo::DeviceBufferRT::create(virvo::PF_RGBA32F, virvo::PF_UNSPECIFIED);
    }
    setRenderTarget(rt);
#else
    setRenderTarget(virvo::HostBufferRT::create(virvo::PF_RGBA32F, virvo::PF_UNSPECIFIED));
#endif

    // Handle Preintegration tables

    // Dylan Added Code:
    std::cerr << __LINE__ << ' ' << cudaGetErrorString(cudaGetLastError()) << '\n';

    impl_->params.albedo = 0.9;
    impl_->params.anisotropy = 0.0f;
    impl_->params.ambient_radius = 1;      //impl_->params.delta / 4;
    impl_->loadPreintegrationTable(impl_->params.albedo);
    impl_->params.pit = typename pit_type::ref_type(impl_->pit);
    
    // TODO: Move extinction volume code to be handled whenever the transfer function is updated
    
#if DEBUG_CUDA
    std::cerr << __LINE__ << ' ' << cudaGetErrorString(cudaGetLastError()) << '\n';
#endif
    impl_->extinction_volume_svt = SVT<float>();
    virvo::aabb vd_bb = vd->getBoundingBox();
    aabbi bounding_box;
    bounding_box.min = vec3i(0, 0, 0);
    bounding_box.max = vec3i(vd->getBoundingBox().size().x, vd->getBoundingBox().size().y, vd->getBoundingBox().size().z);
    impl_->extinction_volume_svt.reset(vd, bounding_box);
#if DEBUG_CUDA
    std::cerr << __LINE__ << ' ' << cudaGetErrorString(cudaGetLastError()) << '\n';
#endif
    impl_->extinction_volume_texture = ext_vol_type(vd->getSize().x, vd->getSize().y, vd->getSize().z);// vd->getSize().x, vd->getSize().y, vd->getSize().z);
    impl_->extinction_volume_texture.reset(impl_->extinction_volume_svt.data());
    impl_->extinction_volume_texture.set_address_mode(Clamp);
    impl_->extinction_volume_texture.set_filter_mode(Nearest);

    impl_->params.extinction_volume = typename ext_vol_type::ref_type(impl_->extinction_volume_texture);
#if DEBUG_CUDA
    std::cerr << __LINE__ << ' ' << cudaGetErrorString(cudaGetLastError()) << '\n';
#endif
    
    // TODO: Add extinction volume stuff
    //impl_->params.extinction_volume_svt.reset(createExtinctionVolume(vd, impl_->params.transfuncs[0], impl_->params.ambient_radius));


    // Dylan End

    updateVolumeData();
    updateTransferFunction();
}

vvRayCaster::~vvRayCaster()
{
}

void vvRayCaster::renderVolumeGL()
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

    // Get OpenGL depth buffer to clip against
    pixel_format depth_format = PF_UNSPECIFIED;

    bool depth_test = glIsEnabled(GL_DEPTH_TEST);

    if (depth_test)
    {
        GLint depth_bits = 0;
        glGetFramebufferAttachmentParameteriv(
                GL_FRAMEBUFFER,
                GL_DEPTH,
                GL_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE,
                &depth_bits
                );

        GLint stencil_bits = 0;
        glGetFramebufferAttachmentParameteriv(
                GL_FRAMEBUFFER,
                GL_STENCIL,
                GL_FRAMEBUFFER_ATTACHMENT_STENCIL_SIZE,
                &stencil_bits
                );


        // TODO: make this more general
        // 24-bit depth buffer and 8-bit stencil buffer
        // is however a quite common case
        depth_format = (depth_bits == 24 && stencil_bits == 8) ? PF_DEPTH24_STENCIL8 : PF_DEPTH32F;

#ifdef __APPLE__
        // PIXEL_PACK_BUFFER with unsigned does not work
        // on Mac OS X, default to 32-bit floating point
        // depth buffer
        depth_format = PF_DEPTH32F;
#endif

        impl_->depth_buffer.map(viewport, depth_format);
        depth_test = true;
    }

    // assemble clip objects
    aligned_vector<typename Impl::params_type::clip_object> clip_objects;

#if 0
    // OpenGL clip planes
    for (int i = 0; i < GL_MAX_CLIP_PLANES; ++i)
    {
        if (!glIsEnabled(GL_CLIP_PLANE0 + i))
        {
            continue;
        }

        GLdouble eq[4] = { 0, 0, 0, 0 };
        glGetClipPlane(GL_CLIP_PLANE0 + i, eq);

        clip_plane pl;
        pl.normal = vec3(eq[0], eq[1], eq[2]);
        pl.offset = eq[3];
        clip_objects.push_back(pl);
    }
#else
/*    auto s0 = vvClipSphere::create();
    s0->center = virvo::vec3(0, 0, 50);
    s0->radius = 50.0f;
    setParameter(VV_CLIP_OBJ0, s0);
    setParameter(VV_CLIP_OBJ_ACTIVE0, true);*/

/*    auto c0 = vvClipCone::create();
    c0->tip = virvo::vec3(0, 0, 0);
    c0->axis = virvo::vec3(0, 0, -1);
    c0->theta = 40.0f * constants::degrees_to_radians<float>();
    setParameter(VV_CLIP_OBJ0, c0);
    setParameter(VV_CLIP_OBJ_ACTIVE0, true);*/

    typedef vvRenderState::ParameterType PT;
    PT act_id = VV_CLIP_OBJ_ACTIVE0;
    PT obj_id = VV_CLIP_OBJ0;

    for ( ; act_id != VV_CLIP_OBJ_ACTIVE_LAST && obj_id != VV_CLIP_OBJ_LAST
          ; act_id = PT(act_id + 1), obj_id = PT(obj_id + 1))
    {
        if (getParameter(act_id))
        {
            auto obj = getParameter(obj_id).asClipObj();

            if (auto plane = boost::dynamic_pointer_cast<vvClipPlane>(obj))
            {
                clip_plane pl;
                pl.normal = vec3(plane->normal.x, plane->normal.y, plane->normal.z);
                pl.offset = plane->offset;
                clip_objects.push_back(pl);
            }
            else if (auto sphere = boost::dynamic_pointer_cast<vvClipSphere>(obj))
            {
                clip_sphere sp;
                sp.center = vec3(sphere->center.x, sphere->center.y, sphere->center.z);
                sp.radius = sphere->radius;
                clip_objects.push_back(sp);
            }
            else if (auto cone = boost::dynamic_pointer_cast<vvClipCone>(obj))
            {
                clip_cone co;
                co.tip = vec3(cone->tip.x, cone->tip.y, cone->tip.z);
                co.axis = vec3(cone->axis.x, cone->axis.y, cone->axis.z);
                co.theta = cone->theta;
                clip_objects.push_back(co);
            }
        }
    }
#endif


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


#ifdef VV_ARCH_CUDA
    // TODO: consolidate!
    //std::cerr << __LINE__ << ' ' << cudaGetErrorString(cudaGetLastError()) << '\n';

#if DEBUG_CUDA
    std::cerr << __LINE__ << ' ' << cudaGetErrorString(cudaGetLastError()) << '\n';
#endif
    thrust::device_vector<typename volume8_type::ref_type>  device_volumes8;
    auto volumes8_data = [&]()
    {
        device_volumes8.resize(vd->getChan());
        for (int c = 0; c < vd->getChan(); ++c)
        {
            device_volumes8[c] = typename volume8_type::ref_type(impl_->volumes8[vd->getCurrentFrame() + c]);
        }
        return thrust::raw_pointer_cast(device_volumes8.data());
    };
#if DEBUG_CUDA
    std::cerr << __LINE__ << ' ' << cudaGetErrorString(cudaGetLastError()) << '\n';
#endif
    thrust::device_vector<typename volume16_type::ref_type> device_volumes16;
    auto volumes16_data = [&]()
    {
        device_volumes16.resize(vd->getChan());
        for (int c = 0; c < vd->getChan(); ++c)
        {
            device_volumes16[c] = typename volume16_type::ref_type(impl_->volumes16[vd->getCurrentFrame() + c]);
        }
        return thrust::raw_pointer_cast(device_volumes16.data());
    };

#if DEBUG_CUDA
    std::cerr << __LINE__ << ' ' << cudaGetErrorString(cudaGetLastError()) << '\n';
#endif
    thrust::device_vector<typename volume32_type::ref_type> device_volumes32;
    auto volumes32_data = [&]()
    {
        device_volumes32.resize(vd->getChan());
        for (int c = 0; c < vd->getChan(); ++c)
        {
            device_volumes32[c] = typename volume32_type::ref_type(impl_->volumes32[vd->getCurrentFrame() + c]);
        }
        return thrust::raw_pointer_cast(device_volumes32.data());
    };

    std::vector<typename transfunc_type::ref_type> trefs;
    for (const auto &tf : impl_->transfuncs)
    {
        trefs.push_back(tf);
    }

#if DEBUG_CUDA
    std::cerr << __LINE__ << ' ' << cudaGetErrorString(cudaGetLastError()) << '\n';
#endif
    thrust::device_vector<typename transfunc_type::ref_type> device_transfuncs(trefs);
    auto transfuncs_data = [&]()
    {
        return thrust::raw_pointer_cast(device_transfuncs.data());
    };

#if DEBUG_CUDA
    thrust::device_vector<vec2> device_ranges;
#endif
    auto ranges_data = [&]()
    {
        for (int c = 0; c < vd->getChan(); ++c)
        {
            device_ranges.push_back(vec2(vd->range(c).x, vd->range(c).y));
        }

        return thrust::raw_pointer_cast(device_ranges.data());
    };
    
    thrust::device_vector<typename Impl::params_type::clip_object> device_objects(clip_objects);
    auto clip_objects_begin = [&]()
    {
        return thrust::raw_pointer_cast(device_objects.data());
    };

    auto clip_objects_end = [&]()
    {
        return clip_objects_begin() + device_objects.size();
    };

#else
    aligned_vector<typename volume8_type::ref_type>  host_volumes8;
    auto volumes8_data = [&]()
    {
        host_volumes8.resize(vd->getChan());
        for (int c = 0; c < vd->getChan(); ++c)
        {
            host_volumes8[c] = typename volume8_type::ref_type(impl_->volumes8[vd->getCurrentFrame() + c]);
        }
        return host_volumes8.data();
    };

    aligned_vector<typename volume16_type::ref_type> host_volumes16;
    auto volumes16_data = [&]()
    {
        host_volumes16.resize(vd->getChan());
        for (int c = 0; c < vd->getChan(); ++c)
        {
            host_volumes16[c] = typename volume16_type::ref_type(impl_->volumes16[vd->getCurrentFrame() + c]);
        }
        return host_volumes16.data();
    };

    aligned_vector<typename volume32_type::ref_type> host_volumes32;
    auto volumes32_data = [&]()
    {
        host_volumes32.resize(vd->getChan());
        for (int c = 0; c < vd->getChan(); ++c)
        {
            host_volumes32[c] = typename volume32_type::ref_type(impl_->volumes32[vd->getCurrentFrame() + c]);
        }
        return host_volumes32.data();
    };

    aligned_vector<typename transfunc_type::ref_type> host_transfuncs(impl_->transfuncs.size());
    auto transfuncs_data = [&]()
    {
        for (size_t i = 0; i < impl_->transfuncs.size(); ++i)
        {
            host_transfuncs[i] = typename transfunc_type::ref_type(impl_->transfuncs[i]);
        }
        return host_transfuncs.data();
    };

    aligned_vector<vec2> host_ranges;
    auto ranges_data = [&]()
    {
        for (int c = 0; c < vd->getChan(); ++c)
        {
            host_ranges.push_back(vec2(vd->range(c).x, vd->range(c).y));
        }

        return host_ranges.data();
    };

    auto clip_objects_begin = [&]()
    {
        return clip_objects.data();
    };

    auto clip_objects_end = [&]()
    {
        return clip_objects.data() + clip_objects.size();
    };
#endif

    // assemble volume kernel params and call kernel
    impl_->params.bbox                      = clip_box( vec3(bbox.min.data()), vec3(bbox.max.data()) );
    impl_->params.delta                     = delta;
    impl_->params.num_channels              = vd->getChan();
#if DEBUG_CUDA
    std::cerr << __LINE__ << ' ' << cudaGetErrorString(cudaGetLastError()) << '\n';
#endif
    impl_->params.transfuncs                = transfuncs_data();
#if DEBUG_CUDA
    std::cerr << __LINE__ << ' ' << cudaGetErrorString(cudaGetLastError()) << '\n';
#endif
    impl_->params.ranges                    = ranges_data();
    impl_->params.depth_buffer              = impl_->depth_buffer.data();
    impl_->params.depth_format              = depth_format;
    // TODO: Put this code back
    impl_->params.mode                      = Impl::params_type::ShowExtinction; //Impl::params_type::projection_mode(getParameter(VV_MIP_MODE).asInt());
    impl_->params.depth_test                = depth_test;
    impl_->params.opacity_correction        = getParameter(VV_OPCORR);
    impl_->params.early_ray_termination     = getParameter(VV_TERMINATEEARLY);
    impl_->params.local_shading             = getParameter(VV_LIGHTING);
    impl_->params.camera_matrix_inv         = inverse(proj_matrix * view_matrix);
    impl_->params.viewport                  = viewport;
    impl_->params.light                     = light;
    impl_->params.clip_objects.begin        = clip_objects_begin();
    impl_->params.clip_objects.end          = clip_objects_end();
#if DEBUG_CUDA
    std::cerr << __LINE__ << ' ' << cudaGetErrorString(cudaGetLastError()) << '\n';
#endif

    if (impl_->texture_format == virvo::PF_R8)
    {
#if DEBUG_CUDA
        std::cerr << __LINE__ << ' ' << cudaGetErrorString(cudaGetLastError()) << '\n';
#endif
        volume_kernel<volume8_type, volume32_type> kernel(impl_->params, volumes8_data());
        impl_->sched.frame(kernel, sparams);
    }
    else if (impl_->texture_format == virvo::PF_R16UI)
    { 
#if DEBUG_CUDA
        std::cerr << __LINE__ << ' ' << cudaGetErrorString(cudaGetLastError()) << '\n';
#endif
        volume_kernel<volume16_type, volume32_type> kernel(impl_->params, volumes16_data());
        impl_->sched.frame(kernel, sparams);
    }
    else if (impl_->texture_format == virvo::PF_R32F)
    {
#if DEBUG_CUDA
        std::cerr << __LINE__ << ' ' << cudaGetErrorString(cudaGetLastError()) << '\n';
#endif
        volume_kernel<volume32_type, volume32_type> kernel(impl_->params, volumes32_data());
        impl_->sched.frame(kernel, sparams);
    }

    if (depth_test)
    {
        impl_->depth_buffer.unmap();
    }
}

void vvRayCaster::updateTransferFunction()
{
    impl_->updateTransfuncTexture(vd, this);

    std::vector<typename transfunc_type::ref_type> trefs;
    //transfunc_type::ref_type tref = typename transfunc_cpu_type::ref_type(impl_->cpu_transfunc);
    for (const auto &tf : impl_->transfuncs)
    {
        trefs.push_back(tf);
    }
    
    impl_->extinction_volume_svt.build(trefs[0], impl_->transfunc_values);
    impl_->extinction_volume_texture.reset(impl_->extinction_volume_svt.data());
    impl_->extinction_volume_texture.set_address_mode(Clamp);
    impl_->extinction_volume_texture.set_filter_mode(Nearest);

    impl_->params.extinction_volume = typename ext_vol_type::ref_type(impl_->extinction_volume_texture);
    
}

void vvRayCaster::updateVolumeData()
{
    impl_->updateVolumeTextures(vd, this);

    std::vector<typename transfunc_type::ref_type> trefs;
    //transfunc_type::ref_type tref = typename transfunc_cpu_type::ref_type(impl_->cpu_transfunc);
    for (const auto &tf : impl_->transfuncs)
    {
        trefs.push_back(tf);
    }
    if (trefs.size() > 0)
    {
        impl_->extinction_volume_svt.reset(vd, aabbi(vec3i(0), vec3i(vd->vox.x, vd->vox.y, vd->vox.z)));
        impl_->extinction_volume_svt.build(trefs[0], impl_->transfunc_values);
        impl_->extinction_volume_texture.reset(impl_->extinction_volume_svt.data());
        impl_->extinction_volume_texture.set_address_mode(Clamp);
        impl_->extinction_volume_texture.set_filter_mode(Nearest);

        impl_->params.extinction_volume = typename ext_vol_type::ref_type(impl_->extinction_volume_texture);
    }
}

bool vvRayCaster::checkParameter(ParameterType param, vvParam const& value) const
{
    switch (param)
    {
    case VV_SLICEINT:
        {
            virvo::tex_filter_mode mode = static_cast< virvo::tex_filter_mode >(value.asInt());

            if (mode == virvo::Nearest || mode == virvo::Linear)
            {
                return true;
            }
        }
        return false;

    case VV_CLIP_OBJ0:
    case VV_CLIP_OBJ1:
    case VV_CLIP_OBJ2:
    case VV_CLIP_OBJ3:
    case VV_CLIP_OBJ4:
    case VV_CLIP_OBJ5:
    case VV_CLIP_OBJ6:
    case VV_CLIP_OBJ7:
        return true;

    default:
        return vvRenderer::checkParameter(param, value);
    }
}

void vvRayCaster::setParameter(ParameterType param, vvParam const& value)
{
    switch (param)
    {
    case VV_SLICEINT:
        {
            if (_interpolation != static_cast< virvo::tex_filter_mode >(value.asInt()))
            {
                _interpolation = static_cast< virvo::tex_filter_mode >(value.asInt());
                tex_filter_mode filter_mode = _interpolation == virvo::Linear ? Linear : Nearest;

                for (auto& tex : impl_->volumes8)
                {
                    tex.set_filter_mode(filter_mode);
                }

                for (auto& tex : impl_->volumes16)
                {
                    tex.set_filter_mode(filter_mode);
                }

                for (auto& tex : impl_->volumes32)
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

bool vvRayCaster::instantClassification() const
{
    return true;
}

vvRenderer* createRayCaster(vvVolDesc* vd, vvRenderState const& rs)
{
    return new vvRayCaster(vd, rs);
}
