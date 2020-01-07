#include "flash.h"

#if VV_HAVE_HDF5
#include <array>
#include <cassert>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <ostream>
#include <vector>

#include <H5Cpp.h>

#include <virvo/vvvoldesc.h>

#define MAX_STRING_LENGTH 80

struct sim_info_t
{
    int file_format_version;
    char setup_call[400];
    char file_creation_time[MAX_STRING_LENGTH];
    char flash_version[MAX_STRING_LENGTH];
    char build_date[MAX_STRING_LENGTH];
    char build_dir[MAX_STRING_LENGTH];
    char build_machine[MAX_STRING_LENGTH];
    char cflags[400];
    char fflags[400];
    char setup_time_stamp[MAX_STRING_LENGTH];
    char build_time_stamp[MAX_STRING_LENGTH];
};

struct grid_t
{
    typedef std::array<char, 4> char4;
    typedef struct __attribute__((packed)) { double x, y, z; } vec3d;
    typedef struct { vec3d min, max; } aabbd;

    struct __attribute__((packed)) gid_t
    {
        int neighbors[6];
        int parent;
        int children[8];
    };

    std::vector<char4> unknown_names;
    std::vector<int> refine_level;
    std::vector<int> node_type; // node_type 1 ==> leaf
    std::vector<gid_t> gid;
    std::vector<vec3d> coordinates;
    std::vector<vec3d> block_size;
    std::vector<aabbd> bnd_box;
    std::vector<int> which_child;
};

struct variable_t
{
    size_t global_num_blocks;
    size_t nxb;
    size_t nyb;
    size_t nzb;

    std::vector<double> data;
};

void read_sim_info(sim_info_t& dest, H5::H5File const& file)
{
    H5::StrType str80(H5::PredType::C_S1, 80);
    H5::StrType str400(H5::PredType::C_S1, 400);

    H5::CompType ct(sizeof(sim_info_t));
    ct.insertMember("file_format_version", 0, H5::PredType::NATIVE_INT);
    ct.insertMember("setup_call", 4, str400);
    ct.insertMember("file_creation_time", 404, str80);
    ct.insertMember("flash_version", 484, str80);
    ct.insertMember("build_date", 564, str80);
    ct.insertMember("build_dir", 644, str80);
    ct.insertMember("build_machine", 724, str80);
    ct.insertMember("cflags", 804, str400);
    ct.insertMember("fflags", 1204, str400);
    ct.insertMember("setup_time_stamp", 1604, str80);
    ct.insertMember("build_time_stamp", 1684, str80);

    H5::DataSet dataset = file.openDataSet("sim info");

    dataset.read(&dest, ct);
}

void read_grid(grid_t& dest, H5::H5File const& file)
{
    H5::DataSet dataset;
    H5::DataSpace dataspace;

    {
        H5::StrType str4(H5::PredType::C_S1, 4);

        dataset = file.openDataSet("unknown names");
        dataspace = dataset.getSpace();
        dest.unknown_names.resize(dataspace.getSimpleExtentNpoints());
        dataset.read(dest.unknown_names.data(), str4, dataspace, dataspace);
    }

    {
        dataset = file.openDataSet("refine level");
        dataspace = dataset.getSpace();
        dest.refine_level.resize(dataspace.getSimpleExtentNpoints());
        dataset.read(dest.refine_level.data(), H5::PredType::NATIVE_INT, dataspace, dataspace);
    }

    {
        dataset = file.openDataSet("node type");
        dataspace = dataset.getSpace();
        dest.node_type.resize(dataspace.getSimpleExtentNpoints());
        dataset.read(dest.node_type.data(), H5::PredType::NATIVE_INT, dataspace, dataspace);
    }

    {
        dataset = file.openDataSet("gid");
        dataspace = dataset.getSpace();

        hsize_t dims[2];
        dataspace.getSimpleExtentDims(dims);
        dest.gid.resize(dims[0]);
        assert(dims[1] == 15);

        dataset.read(dest.gid.data(), H5::PredType::NATIVE_INT, dataspace, dataspace);
    }

    {
        dataset = file.openDataSet("coordinates");
        dataspace = dataset.getSpace();

        hsize_t dims[2];
        dataspace.getSimpleExtentDims(dims);
        dest.coordinates.resize(dims[0]);
        assert(dims[1] == 3);

        dataset.read(dest.coordinates.data(), H5::PredType::NATIVE_DOUBLE, dataspace, dataspace);
    }

    {
        dataset = file.openDataSet("block size");
        dataspace = dataset.getSpace();

        hsize_t dims[2];
        dataspace.getSimpleExtentDims(dims);
        dest.block_size.resize(dims[0]);
        assert(dims[1] == 3);

        dataset.read(dest.block_size.data(), H5::PredType::NATIVE_DOUBLE, dataspace, dataspace);
    }

    {
        dataset = file.openDataSet("bounding box");
        dataspace = dataset.getSpace();

        hsize_t dims[3];
        dataspace.getSimpleExtentDims(dims);
        dest.bnd_box.resize(dims[0] * 2);
        assert(dims[1] == 3);
        assert(dims[2] == 2);

        std::vector<double> temp(dims[0] * dims[1] * dims[2]);

        dataset.read(temp.data(), H5::PredType::NATIVE_DOUBLE, dataspace, dataspace);

        dest.bnd_box.resize(dims[0]);
        for (size_t i = 0; i < dims[0]; ++i)
        {
            dest.bnd_box[i].min.x = temp[i * 6];
            dest.bnd_box[i].max.x = temp[i * 6 + 1];
            dest.bnd_box[i].min.y = temp[i * 6 + 2];
            dest.bnd_box[i].max.y = temp[i * 6 + 3];
            dest.bnd_box[i].min.z = temp[i * 6 + 4];
            dest.bnd_box[i].max.z = temp[i * 6 + 5];
        //std::cout << dest.bnd_box[i].min.x << ' ' << dest.bnd_box[i].min.y << ' ' << dest.bnd_box[i].min.z << '\n';
        //std::cout << dest.bnd_box[i].max.x << ' ' << dest.bnd_box[i].max.y << ' ' << dest.bnd_box[i].max.z << '\n';
        }
    }

    {
        dataset = file.openDataSet("which child");
        dataspace = dataset.getSpace();
        dest.which_child.resize(dataspace.getSimpleExtentNpoints());
        dataset.read(dest.which_child.data(), H5::PredType::NATIVE_INT, dataspace, dataspace);
    }
}

void read_variable(variable_t& var, H5::H5File const& file, char const* varname)
{
    H5::DataSet dataset = file.openDataSet(varname);
    H5::DataSpace dataspace = dataset.getSpace();

    //std::cout << dataspace.getSimpleExtentNdims() << '\n';
    hsize_t dims[4];
    dataspace.getSimpleExtentDims(dims);
    var.global_num_blocks = dims[0];
    var.nxb = dims[1];
    var.nyb = dims[2];
    var.nzb = dims[3];
    var.data.resize(dims[0] * dims[1] * dims[2] * dims[3]);
    dataset.read(var.data.data(), H5::PredType::NATIVE_DOUBLE, dataspace, dataspace);
    //std::cout << dims[0] << ' ' << dims[1] << ' ' << dims[2] << ' ' << dims[3] << '\n';
}

void resample(grid_t const& grid, variable_t const& var, vvVolDesc& vd, int nx, int ny, int nz)
{
    vd.vox[0] = nx;
    vd.vox[1] = ny;
    vd.vox[2] = nz;
    vd.frames = 1;
    vd.setChan(1);
    vd.setDist(1.f,1.f,1.f);
    vd.bpc = 1;
    //vd.mapping(0) = virvo::vec2(0, UCHAR_MAX);
    vd.mapping(0) = virvo::vec2(-3.378636, 11.01426);
    uint8_t* raw = new uint8_t[vd.getFrameBytes()];
    vd.addFrame(raw, vvVolDesc::ARRAY_DELETE);

    std::cout << std::fixed;

    // Length of the sides of the bounding box
    double len_total[3] = {
        grid.bnd_box[0].max.x - grid.bnd_box[0].min.x,
        grid.bnd_box[0].max.y - grid.bnd_box[0].min.y,
        grid.bnd_box[0].max.z - grid.bnd_box[0].min.z
        };

    //std::cout << len_total[0] << ' ' << len_total[1] << ' ' << len_total[2] << '\n';

    int max_level = 0;
    double len[3] = { 0.0 };
    for (size_t i = 0; i < var.global_num_blocks; ++i)
    {
        if (grid.refine_level[i] > max_level)
        {
            max_level = grid.refine_level[i];
            len[0] = grid.bnd_box[i].max.x - grid.bnd_box[i].min.x;
            len[1] = grid.bnd_box[i].max.y - grid.bnd_box[i].min.y;
            len[2] = grid.bnd_box[i].max.z - grid.bnd_box[i].min.z;
        }
    }

    len[0] /= var.nxb;
    len[1] /= var.nyb;
    len[2] /= var.nzb;

    // This is the number of cells for the finest level (?)
    int vox[3];
    vox[0] = static_cast<int>(round(len_total[0] / len[0]));
    vox[1] = static_cast<int>(round(len_total[1] / len[1]));
    vox[2] = static_cast<int>(round(len_total[2] / len[2]));

    std::cout << vox[0] << ' ' << vox[1] << ' ' << vox[2] << '\n';

    //std::cout << grid.bnd_box[0].min.x << ' ' << grid.bnd_box[0].min.y << ' ' << grid.bnd_box[0].min.z << '\n';
    //std::cout << grid.bnd_box[0].max.x << ' ' << grid.bnd_box[0].max.y << ' ' << grid.bnd_box[0].max.z << '\n';
    float max_scalar = -FLT_MAX;
    float min_scalar =  FLT_MAX;
    for (size_t i = 0; i < var.global_num_blocks; ++i)
    {
        std::cout << "Block (" << (i+1) << '/' << var.global_num_blocks << ")\n";

        if (grid.node_type[i] == 1) // leaf!
        {
            // Project min on vox grid
            int level = max_level-grid.refine_level[i];
            int cellsize = 1<<level;

            int lower[3] = {
                static_cast<int>(round((grid.bnd_box[i].min.x - grid.bnd_box[0].min.x) / len_total[0] * vox[0])),
                static_cast<int>(round((grid.bnd_box[i].min.y - grid.bnd_box[0].min.y) / len_total[1] * vox[1])),
                static_cast<int>(round((grid.bnd_box[i].min.z - grid.bnd_box[0].min.z) / len_total[2] * vox[2]))
                };
            //std::cout << lower[0] << ' ' << lower[1] << ' ' << lower[2] << '\n';

            for (int z = 0; z < (int)var.nzb; ++z)
            {
                for (int y = 0; y < (int)var.nyb; ++y)
                {
                    for (int x = 0; x < (int)var.nxb; ++x)
                    {
                        double coord[3] = {
                            lower[0] / static_cast<double>(vox[0]) * len_total[0] + grid.bnd_box[0].min.x,
                            lower[1] / static_cast<double>(vox[1]) * len_total[1] + grid.bnd_box[1].min.y,
                            lower[2] / static_cast<double>(vox[2]) * len_total[2] + grid.bnd_box[2].min.z
                            };

                        // Clip out a region of interest
                        // Clip planes are specific to the SILCC molecular cloud data set
                        static const double XMIN = 3.5e20;
                        static const double XMAX = 6.2e20;
                        static const double YMIN = -4.9e20;
                        static const double YMAX = -2.2e20;
                        static const double ZMIN = -12.e19;
                        static const double ZMAX = 12.e19;
                        if (coord[0] <  XMIN || coord[0] > XMAX || coord[1] < YMIN || coord[1] > YMAX || coord[2] < ZMIN || coord[2] > ZMAX)
                            continue;

                        // data into var
                        size_t index = i * var.nxb * var.nyb * var.nzb
                                            + z * var.nyb * var.nxb
                                            + y * var.nxb
                                            + x;

                        double scalar = static_cast<double>(var.data[index]);
                        //scalar += 15429999982016076972032.000000;
                        scalar = scalar != .0 ? log(scalar) : scalar;
                        float scalarf = static_cast<float>(scalar);
                        max_scalar = std::max(max_scalar, scalarf);
                        min_scalar = std::min(min_scalar, scalarf);

                        for (int zz = lower[2] + z*cellsize; zz < lower[2] + z*cellsize + cellsize; ++zz)
                        {
                            for (int yy = lower[1] + y*cellsize; yy < lower[1] + y*cellsize + cellsize; ++yy)
                            {
                                for (int xx = lower[0] + x*cellsize; xx < lower[0] + x*cellsize + cellsize; ++xx)
                                {
                                    double coord_XXX[3] = {
                                        xx / static_cast<double>(vox[0]) * len_total[0] + grid.bnd_box[0].min.x,
                                        yy / static_cast<double>(vox[1]) * len_total[1] + grid.bnd_box[1].min.y,
                                        zz / static_cast<double>(vox[2]) * len_total[2] + grid.bnd_box[2].min.z
                                        };

                                    // volume coord
                                    int vx = (coord_XXX[0] - XMIN) / (XMAX - XMIN) * (nx - 1 + 0.999);
                                    int vy = (coord_XXX[1] - YMIN) / (YMAX - YMIN) * (ny - 1 + 0.999);
                                    int vz = (coord_XXX[2] - ZMIN) / (ZMAX - ZMIN) * (nz - 1 + 0.999);
                                    //std::cout << vx << ' ' << vy << ' ' << vz << '\n';

                                    vd.setChannelValue(scalarf, 0, vx, vy, vz, 0);
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    vd.range(0) = virvo::vec2(min_scalar, max_scalar);
    std::cout << min_scalar << ' ' << max_scalar << '\n';
}

namespace virvo
{
namespace flash
{

bool can_load(const vvVolDesc *vd)
{
    std::string filename(vd->getFilename());

    try
    {
        H5::H5File file(filename.c_str(), H5F_ACC_RDONLY);

        // Read simulation info
        sim_info_t sim_info;
        read_sim_info(sim_info, file);

        // Read grid data
        grid_t grid;
        read_grid(grid, file);

        return true;
    }
    catch (...)
    {
        return false;
    }
}

void load(vvVolDesc* vd)
{
    std::string filename(vd->getFilename());
    std::string var = "temp";

    try
    {
        H5::H5File file(filename.c_str(), H5F_ACC_RDONLY);

        // Read simulation info
        sim_info_t sim_info;
        read_sim_info(sim_info, file);

        // Read grid data
        grid_t grid;
        read_grid(grid, file);

        // Read data
        variable_t* density = new variable_t;
        read_variable(*density, file, var.c_str());

        resample(grid, *density, *vd, 512, 512, 512);
    }
    catch (H5::FileIException error)
    {
        error.printError();
        return;
    }
    catch (H5::DataSpaceIException error)
    {
        error.printError();
        return;
    }
    catch (H5::DataTypeIException error)
    {
        error.printError();
        return;
    }

    return;
}

} // flash
} // virvo
#endif // VV_HAVE_HDF5
