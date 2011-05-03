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

#ifndef _VVTEXREND_H_
#define _VVTEXREND_H_

#include <vector>

// Virvo:
#include "vvbrick.h"
#include "vvexport.h"
#include "vvvoldesc.h"
#include "vvrenderer.h"
#include "vvtransfunc.h"
#include "vvbsptree.h"
#include "vvshadermanager.h"
#include "vvoffscreenbuffer.h"
#include "vvopengl.h"
#include "vvpthread.h"

struct ThreadArgs;
class vvThreadVisitor;

/*!
 * Generate more colors by adjusting this literal and \ref generateDebugColors()
 * if more than 8 threads need to be colorized for debugging.
 */
const int MAX_DEBUG_COLORS = 8;

// Uncomment to use geometry programs in addition to vertex programs.
//#define ISECT_GLSL_GEO

// Uncomment to use instanced geometry transfer rather than using vertex arrays.
//#define ISECT_GLSL_INST

//============================================================================
// Class Definitions
//============================================================================

/** Volume rendering engine using a texture-based algorithm.
  Textures can be drawn as planes or spheres. In planes mode a rendering
  quality can be given (determining the number of texture slices used), and
  the texture normal can be set according to the application's needs.<P>
  The data points are located at the grid as follows:<BR>
  The outermost data points reside at the very edge of the drawn object,
  the other values are evenly distributed inbetween.
  Make sure you define HAVE_CG in your compiler if you want to use Nvidia Cg.
  @author Juergen Schulze (schulze@cs.brown.de)
  @author Martin Aumueller
  @author Stefan Zellmann
  @see vvRenderer
*/

const int ISECT_SHADER_SEQUENCE      = 0;
const int ISECT_SHADER_V1            = 1;
const int ISECT_SHADER_V2            = 2;
const int ISECT_SHADER_BRICKMIN      = 3;
const int ISECT_SHADER_BRICKDIMINV   = 4;
const int ISECT_SHADER_TEXRANGE      = 5;
const int ISECT_SHADER_TEXMIN        = 6;
const int ISECT_SHADER_MODELVIEWPROJ = 7;
const int ISECT_SHADER_DELTA         = 8;
const int ISECT_SHADER_PLANENORMAL   = 9;
const int ISECT_SHADER_FRONTINDEX    = 10;
const int ISECT_SHADER_VERTICES      = 11;
const int ISECT_SHADER_V1_MAYBE      = 12;
const int ISECT_SHADER_V2_MAYBE      = 13;
const int ISECT_SHADER_FIRSTPLANE    = 14;

class VIRVOEXPORT vvTexRend : public vvRenderer
{
  friend class vvBrick;
  public:
    struct BrickSet
    {
      vvSLList<vvBrick*> bricks;
      int parentThreadId;

      vvVector3 center;
      float dist;

      inline bool operator<(const BrickSet& rhs) const      ///< compare bricks based upon dist to eye position
      {
        return (dist < rhs.dist);
      }
    };
    unsigned int _numThreads;                     ///< thread count
    unsigned int _usedThreads;                    ///< actually used threads, some maybe can't be used because of errors
    
    pthread_t* _threads;                          ///< worker threads
    ThreadArgs* _threadData;                      ///< args for each thread
    pthread_barrier_t _madeEmptyBricksBarrier;    ///< Workers: don't load up textures, before brick outlines are generated
    pthread_barrier_t _distributedBricksBarrier;  ///< barrier is passed when bricks are distributed eventually
    pthread_barrier_t _renderStartBarrier;        ///< barrier assures that the render loop doesn't resume until proper data is supplied
    pthread_barrier_t _compositingBarrier;        ///< barrier assures synchronization for compositing
    bool _terminateThreads;

    std::vector<GLint> _vertArray;
    std::vector<GLsizei> _elemCounts;
    std::vector<GLuint> _vertIndicesAll;
    std::vector<GLuint *> _vertIndices;
    vvOffscreenBuffer** _offscreenBuffers;
    bool _somethingChanged;                       ///< when smth changed (e.g. the transfer function, bricks will possibly be rearranged)
    vvBspTree* _bspTree;
    vvThreadVisitor* _visitor;
    int _deviationExceedCnt;

    int _numBricks[3];                            ///< number of bricks for each dimension
    enum ErrorType                                /// Error Codes
    {
      OK = 0,                                     ///< no error
      TRAM_ERROR,                                 ///< not enough texture memory
      TEX_SIZE_UNKNOWN,                           ///< size of 3D texture is unknown
      NO3DTEX,                                    ///< 3D textures not supported on this hardware
      NO_DISPLAYS_SPECIFIED,                      ///< no x-displays in _renderState, thus no multi-threading
      NO_DISPLAYS_OPENED,                         ///< no additional x-displays could be opened, thus no multi-threading
      UNSUPPORTED                                 ///< general error code
    };
    enum GeometryType                             /// Geometry textures are projected on
    {
      VV_AUTO = 0,                                ///< automatically choose best
      VV_SLICES,                                  ///< render slices parallel to xy axis plane using 2D textures
      VV_CUBIC2D,                                 ///< render slices parallel to all axis planes using 2D textures
      VV_VIEWPORT,                                ///< render planar slices using a 3D texture
      VV_SPHERICAL,                               ///< render spheres originating at viewer using a 3D texture
      VV_BRICKS                                   ///< render volume using bricking
    };
    enum VoxelType                                /// Internal data type used in textures
    {
      VV_BEST = 0,                                ///< choose best
      VV_RGBA,                                    ///< transfer function look-up done in software
      VV_SGI_LUT,                                 ///< SGI color look-up table
      VV_PAL_TEX,                                 ///< OpenGL paletted textures
      VV_TEX_SHD,                                 ///< Nvidia texture shader
      VV_PIX_SHD,                                 ///< Nvidia pixel shader
      VV_FRG_PRG,                                 ///< ARB fragment program
      VV_GLSL_SHD                                 ///< Glsl programs
    };
    enum FeatureType                              /// Rendering features
    {
      VV_MIP                                      ///< maximum intensity projection
    };
    enum SliceOrientation                         /// Slice orientation for planar 3D textures
    {
      VV_VARIABLE = 0,                            ///< choose automatically
      VV_VIEWPLANE,                               ///< parallel to view plane
      VV_CLIPPLANE,                               ///< parallel to clip plane
      VV_VIEWDIR,                                 ///< perpendicular to viewing direction
      VV_OBJECTDIR,                               ///< perpendicular to line eye-object
      VV_ORTHO                                    ///< as in orthographic projection
    };

    static const int NUM_PIXEL_SHADERS;           ///< number of pixel shaders used
  private:
    enum FragmentProgram
    {
      VV_FRAG_PROG_2D = 0,
      VV_FRAG_PROG_3D,
      VV_FRAG_PROG_PREINT,
      VV_FRAG_PROG_MAX                            // has always to be last in list
    };
    float* rgbaTF;                                ///< density to RGBA conversion table, as created by TF [0..1]
    uchar* rgbaLUT;                               ///< final RGBA conversion table, as transferred to graphics hardware (includes opacity and gamma correction)
    uchar* preintTable;                           ///< lookup table for pre-integrated rendering, as transferred to graphics hardware
    float  lutDistance;                           ///< slice distance for which LUT was computed
    int   texels[3];                              ///< width, height and depth of volume, including empty space [texels]
    float texMin[3];                              ///< minimum texture value of object [0..1] (to prevent border interpolation)
    float texMax[3];                              ///< maximum texture value of object [0..1] (to prevent border interpolation)
    int   textures;                               ///< number of textures stored in TRAM
    int   texelsize;                              ///< number of bytes/voxel transferred to OpenGL (depending on rendering mode)
    GLint internalTexFormat;                      ///< internal texture format (parameter for glTexImage...)
    GLenum texFormat;                             ///< texture format (parameter for glTexImage...)
    GLuint* texNames;                             ///< names of texture slices stored in TRAM
    GLuint pixLUTName;                            ///< name for transfer function texture
    GLuint fragProgName[VV_FRAG_PROG_MAX];        ///< names for fragment programs (for applying transfer function)
    GeometryType geomType;                        ///< rendering geometry actually used
    VoxelType voxelType;                          ///< voxel type actually used
    bool extTex3d;                                ///< true = 3D texturing supported
    bool extNonPower2;                            ///< true = NonPowerOf2 textures supported
    bool extColLUT;                               ///< true = SGI texture color lookup table supported
    bool extPalTex;                               ///< true = OpenGL 1.2 paletted textures supported
    bool extMinMax;                               ///< true = maximum/minimum intensity projections supported
    bool extTexShd;                               ///< true = Nvidia texture shader & texture shader 2 support
    bool extPixShd;                               ///< true = Nvidia pixel shader support (requires GeForce FX)
    bool extGlslShd;                              ///< true = Glsl program support
    bool extBlendEquation;                        ///< true = support for blend equation extension
    bool arbFrgPrg;                               ///< true = ARB fragment program support
    bool arbMltTex;                               ///< true = ARB multitexture support
    bool preIntegration;                          ///< true = try to use pre-integrated rendering (planar 3d textures)
    bool usePreIntegration;                       ///< true = pre-integrated rendering is actually used
    bool interpolation;                           ///< interpolation mode: true=linear interpolation (default), false=nearest neighbor
    bool opacityCorrection;                       ///< true = opacity correction on
    int  minSlice, maxSlice;                      ///< min/maximum slice to render [0..numSlices-1], -1 for no slice constraints
    bool _areEmptyBricksCreated;                  ///< true when brick outlines are created or assigned through constructor
    bool _areBricksCreated;                       ///< true after the first creation of the bricks
    bool _measureRenderTime;                      ///< if time needs not to be measured, a costly call to glFinish can be spared
    std::vector<BrickList> _brickList;            ///< contains all created bricks for all frames
    std::vector<BrickList> _nonemptyList;         ///< contains all non-transparent bricks for all frames
    BrickList _insideList;                        ///< contains all non-empty bricks inside the probe
    BrickList _sortedList;                        ///< contains all bricks inside the probe in a sorted order (back to front)
    bool _useOnlyOneBrick;                        ///< true if whole data fits in texture memory
    vvVector4 _frustum[6];                        ///< current planes of view frustum
    SliceOrientation _sliceOrientation;           ///< slice orientation for planar 3d textures
    bool _proxyGeometryOnGpu;                     ///< indicate wether proxy geometry is to be computed on gpu
    bool _proxyGeometryOnGpuSupported;            ///< indicate wether proxy geometry computation on gpu would work
    int _lastFrame;                               ///< last frame rendered
    int _numDisplays;                             ///< # additional displays for multi-gpu rendering
    const char** _displayNames;                   ///< list with displays of the form host:x.y
    BufferPrecision _multiGpuBufferPrecision;     ///< 8, 16 or 32 bit precision for the render slave offscreen buffers
    unsigned int* _displays;                      ///< display name = :x.0 ==> corresponding display: x
    unsigned int* _screens;                       ///< display name = :0.x ==> corresponding screen: x
    int _numSlaveNodes;                           ///< number of available distributed memory hosts
    vvColor _debugColors[MAX_DEBUG_COLORS];       ///< array of colors to visualize threads in dbg mode (debug level >= 2).
                                                  ///< Feel free to use these colors for similar purposes either.

    vvRenderTarget* _renderTarget;                ///< can e.g. be an offscreen buffer to use with image downscaling
                                                  ///< or an image creator making a screenshot

    vvShaderManager* _isectShader;                ///< shader performing intersection test on gpu
    vvShaderManager* _pixelShader;                ///< shader for applying transfer function on gpu

    int _currentShader;                           ///< ID of currently used fragment shader
    int _previousShader;                          ///< ID of previous shader

    vvVector3 _eye;                               ///< the current eye position
    vvAABB* _aabbMask;                            ///< mask out the relevant portion of the volume
    bool _isSlave;                                ///< let the renderer know if it is a rendering slave

    // GL state variables:
    GLboolean glsTexColTable;                     ///< stores GL_TEXTURE_COLOR_TABLE_SGI
    GLboolean glsSharedTexPal;                    ///< stores GL_SHARED_TEXTURE_PALETTE_EXT

    void makeLUTTexture(const GLuint& lutName, uchar* lutData) const;
    ErrorType makeTextures2D(const int axes);

    ErrorType setDisplayNames(const char** displayNames, const unsigned int numNames);
    ErrorType dispatchThreadedWGLContexts(); 
    ErrorType dispatchThreadedGLXContexts();
    ErrorType dispatchThreads();
    ErrorType distributeBricks();
    void notifyThreads(bool brickDataChanged, bool transferFunctionChanged);
    static void* threadFuncTexBricks(void* threadargs);
    void sortBrickList(std::vector<vvBrick*>& list, const vvVector3&, const vvVector3&, const bool);
    void performLoadBalancing();

    ErrorType makeTextures(const GLuint& lutName, uchar*& lutData);
    ErrorType makeEmptyBricks();
    ErrorType makeTextureBricks(GLuint*& privateTexNames, int* numTextures, uchar*& lutData,
                                std::vector<BrickList>& bricks, bool& areBricksCreated) const;

    void initPostClassificationStage(vvShaderManager* pixelShader, GLuint progName[VV_FRAG_PROG_MAX]);
    void initArbFragmentProgram(GLuint progName[VV_FRAG_PROG_MAX]) const;
    bool initPixelShaders(vvShaderManager* pixelShader) const;
    void enablePixelShaders(vvShaderManager* pixelShader, GLuint& lutName);
    void disablePixelShaders(vvShaderManager* pixelShader) const;

    void enableLUTMode(vvShaderManager* pixelShader, GLuint& lutName, GLuint progName[VV_FRAG_PROG_MAX]);
    void disableLUTMode(vvShaderManager* pixelShader) const;

    bool initIntersectionShader(vvShaderManager* isectShader, vvShaderManager* pixelShader = NULL) const;
    void setupIntersectionParameters(vvShaderManager* isectShader) const;
    void enableIntersectionShader(vvShaderManager* isectShader, vvShaderManager* pixelShader = NULL) const;
    void disableIntersectionShader(vvShaderManager* isectShader, vvShaderManager* pixelShader = NULL) const;

    ErrorType makeTextures3D();
    void removeTextures(GLuint*& privateTexNames, int* numTextures) const;
    ErrorType updateTextures3D(const int, const int, const int, const int, const int, const int, const bool);
    ErrorType updateTextures2D(const int, const int, const int, const int, const int, const int, const int);
    ErrorType updateTextureBricks(int, int, int, int, int, int);
    void beforeSetGLenvironment() const;
    void setGLenvironment() const;
    void unsetGLenvironment() const;
    void renderTex3DSpherical(vvMatrix*);
    void renderTex3DPlanar(vvMatrix*);
    void renderTexBricks(const vvMatrix*);
    void renderTex2DSlices(float);
    void renderTex2DCubic(AxisType, float, float, float);
    void generateDebugColors();
    VoxelType findBestVoxelType(const VoxelType) const;
    GeometryType findBestGeometry(const GeometryType, const VoxelType) const;
    void updateLUT(const float, GLuint& lutName, uchar*& lutData, float& lutDistance);
    int  getLUTSize(int*) const;
    int  getPreintTableSize() const;
    void enableNVShaders() const;
    void disableNVShaders() const;
    void enableFragProg(GLuint& lutName, GLuint progName[VV_FRAG_PROG_MAX]) const;
    void disableFragProg() const;
    void enableTexture(const GLenum target) const;
    void disableTexture(const GLenum target) const;
    bool testBrickVisibility(const vvBrick* brick, const vvMatrix& mvpMat) const;
    bool testBrickVisibility(const vvBrick*) const;
    bool intersectsFrustum(const vvVector3 &min, const vvVector3 &max) const;
    bool insideFrustum(const vvVector3 &min, const vvVector3 &max) const;
    void markBricksInFrustum(const vvVector3& probeMin, const vvVector3& probeMax);
    void updateFrustum();
    void calcAABBMask();
    void getBricksInProbe(std::vector<BrickList>& nonemptyList, BrickList& insideList, BrickList& sortedList,
                          const vvVector3, const vvVector3, bool& roiChanged, int threadId = -1); ///< threadId = -1 ==> main thread
    void computeBrickSize();
    void calcNumTexels();
    void calcNumBricks();
    void initVertArray(const int numSlices);
    void validateEmptySpaceLeaping();             ///< only leap empty bricks if tf type is compatible with this
    void evaluateLocalIllumination(vvShaderManager*& pixelShader, const vvVector3& normal);
  public:
    vvTexRend(vvVolDesc*, vvRenderState, GeometryType=VV_AUTO, VoxelType=VV_BEST,
              std::vector<BrickList>* bricks = 0,
              const char** displayNames = 0, const int numDisplays = 0,
              const BufferPrecision multiGpuBufferPrecision = VV_SHORT);
    virtual ~vvTexRend();
    void  renderVolumeGL();
    void  updateTransferFunction();
    void  updateTransferFunction(GLuint& lutName, uchar*& lutData, float& lutDistance,
                                 int& currentShader, bool& usePreIntegration);
    void  updateVolumeData();
    void  updateVolumeData(int, int, int, int, int, int);
    void  fillNonemptyList(std::vector<BrickList>& nonemptyList, std::vector<BrickList>& brickList) const;
    void  activateClippingPlane();
    void  deactivateClippingPlane();
    void  setNumLights(const int);
    bool  instantClassification() const;
    void  setViewingDirection(const vvVector3*);
    void  setObjectDirection(const vvVector3*);
    void  setParameter(ParameterType param, float newValue);
    float getParameter(ParameterType param) const;
    static bool isSupported(const GeometryType);
    static bool isSupported(const VoxelType);
    bool isSupported(const FeatureType) const;
    GeometryType getGeomType() const;
    VoxelType getVoxelType() const;
    int  getCurrentShader() const;
    void setCurrentShader(const int);
    void renderQualityDisplay();
    void printLUT() const;
    void updateBrickGeom();
    void setComputeBrickSize(const bool);
    void setBrickSize(const int);
    int getBrickSize() const;
    void setTexMemorySize(const int);
    int getTexMemorySize() const;
    vvBspTree* getBspTree() const;
    void setAABBMask(vvAABB* aabbMask);
    vvAABB* getAABBMask() const;
    vvAABB getProbedMask() const;
    void setIsSlave(const bool isSlave);
    unsigned char* getHeightFieldData(float[4][3], int&, int&);
    float getManhattenDist(float[3], float[3]) const;
    void prepareDistributedRendering(const int numSlaveNodes);
    std::vector<BrickList>** getBrickListsToDistribute();
    int getNumBrickListsToDistribute() const;
    void calcProjectedScreenRects();
    float calcQualityAndScaleImage();

    static int get2DTextureShader();
    static int getLocalIlluminationShader();
};
#endif

//============================================================================
// End of File
//============================================================================