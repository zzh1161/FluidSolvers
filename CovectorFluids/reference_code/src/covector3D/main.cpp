#include <cmath>
#include "../include/array.h"
#include <iostream>
#include "GPU_Advection.h"
#include "CovectorSolver.h"
#include <boost/filesystem.hpp>

void makeCylinderUp(openvdb::FloatGrid::Ptr grid, float radius, float thickness, Vec3f& center, const openvdb::CoordBBox& indexBB, double h)
{
    typename openvdb::FloatGrid::Accessor accessor = grid->getAccessor();

    for (openvdb::Int32 i = indexBB.min().x(); i < indexBB.max().x(); ++i) {
        for (openvdb::Int32 j = indexBB.min().y(); j < indexBB.max().y(); ++j) {
            for (openvdb::Int32 k = indexBB.min().z(); k < indexBB.max().z(); ++k) {
                // transform point (i, j, k) of index space into world space
                openvdb::Vec3d p(i * h - center[0], j * h - center[1], k * h - center[2]);

                Vec2f d = Vec2f(sqrt(p.x() * p.x() + p.z() * p.z()), abs(p.y())) - Vec2f(radius, thickness);
                Vec2f d_max(max(d[0], 0.0f), max(d[1], 0.0f));
                float distance = min(max(d[0], d[1]), 0.0f) + sqrt(d_max[0] * d_max[0] + d_max[1] * d_max[1]);
                // compute level set function value
                // float distance = sqrt(p.z() * p.z() + p.y() * p.y()) - radius;

                accessor.setValue(openvdb::Coord(i, j, k), distance);
            }
        }
    }

    grid->setTransform(openvdb::math::Transform::createLinearTransform(h));
}

void makeCylinder(openvdb::FloatGrid::Ptr grid, float radius, float thickness, Vec3f& center, const openvdb::CoordBBox& indexBB, double h)
{
    typename openvdb::FloatGrid::Accessor accessor = grid->getAccessor();

    for (openvdb::Int32 i = indexBB.min().x(); i < indexBB.max().x(); ++i) {
        for (openvdb::Int32 j = indexBB.min().y(); j < indexBB.max().y(); ++j) {
            for (openvdb::Int32 k = indexBB.min().z(); k < indexBB.max().z(); ++k) {
                // transform point (i, j, k) of index space into world space
                openvdb::Vec3d p(i * h - center[0], j * h - center[1], k * h - center[2]);

                Vec2f d = Vec2f(sqrt(p.z() * p.z() + p.y() * p.y()), abs(p.x())) - Vec2f(radius, thickness);
                Vec2f d_max(max(d[0], 0.0f), max(d[1], 0.0f));
                float distance = min(max(d[0], d[1]), 0.0f) + sqrt(d_max[0]*d_max[0]+d_max[1]*d_max[1]);
                // compute level set function value
                // float distance = sqrt(p.z() * p.z() + p.y() * p.y()) - radius;

                accessor.setValue(openvdb::Coord(i, j, k), distance);
            }
        }
    }

    grid->setTransform(openvdb::math::Transform::createLinearTransform(h));
}

void makeSlab(openvdb::FloatGrid::Ptr grid, float radius, float innerRadius, float thickness, Vec3f& center, const openvdb::CoordBBox& indexBB, double h)
{
    typename openvdb::FloatGrid::Accessor accessor = grid->getAccessor();

    for (openvdb::Int32 i = indexBB.min().x(); i < indexBB.max().x(); ++i) {
        for (openvdb::Int32 j = indexBB.min().y(); j < indexBB.max().y(); ++j) {
            for (openvdb::Int32 k = indexBB.min().z(); k < indexBB.max().z(); ++k) {
                // transform point (i, j, k) of index space into world space
                openvdb::Vec3d p(i * h - center[0], j * h - center[1], k * h - center[2]);

                Vec2f d = Vec2f(max(abs(p.z()),abs(p.y())), abs(p.x())) - Vec2f(radius, thickness);
                Vec2f d2 = Vec2f(-1.0f);
                if (innerRadius > 0.0f)
                    d2 = Vec2f(innerRadius - sqrt(p.z() * p.z() + p.y() * p.y()), abs(p.x()) - thickness);
                
                Vec2f d_max = Vec2f(max(d[0], max(d2[0], 0.0f)),
                                    max(d[1], max(d2[1], 0.0f)));
                float distance = min(max(max(d[0], d[1]),d2[0]), 0.0f) + sqrt(d_max[0] * d_max[0] + d_max[1] * d_max[1]);
                // compute level set function value
                // float distance = sqrt(p.z() * p.z() + p.y() * p.y()) - radius;

                accessor.setValue(openvdb::Coord(i, j, k), distance);
            }
        }
    }

    grid->setTransform(openvdb::math::Transform::createLinearTransform(h));
}

int main(int argc, char** argv) {
    uint ni = 0;
    uint nj = 0;
    uint nk = 0;
    uint total_frame = 0;
    int _baseres = 0;
    float L = 0;
    float h = 0;
    float dt = 0;
    float mapping_blend_coeff = 1;
    float viscosity = 0;
    float half_width = 0;
    float smoke_rise = 0;
    float smoke_drop = 0;
    float framerate = 24.f;
    float substeps = 1.f;
    Scheme sim_scheme;
    Experiment sim_experiment;
    bool do_EC, do_EC_with_clamp, do_2nd_order, do_dmc, do_antialiasing, do_vel_advection_only;
    int delayed_reinit_num = 1;
    string filepath = "H:/BiMocq/Out_3D/";
    int sim_name = 0;
    int experiment_name;
    if (argc != 3)
    {
        std::cout << "Please specify correct parameters!" << std::endl;
        std::cout << "inputs: [Method] [Experiment]" << std::endl;
        std::cout << "Valid method numbers are [0-7] for 0: SF, 1: SF+R, 2: SCPF, 3: MC, 4: MC+R, 5: BiMocq, 6: CF, 7: CF+MCM" << std::endl;
        std::cout << "Valid experiment numbers are [0-6] for 0: trefoil knot, 1: leapfrogging, 2: smoke plume, 3: pyroclastic, 4: ink jet, 5: delta wing, 6: bunny meteor" << std::endl;
        exit(0);
    }
    if (sim_name >= 8)
    {
        std::cout << "Please enter valid method number!" << std::endl;
        std::cout << "inputs: [Method] [Experiment]" << std::endl;
        std::cout << "Valid method numbers are [0-7] for 0: SF, 1: SF+R, 2: SCPF, 3: MC, 4: MC+R, 5: BiMocq, 6: CF, 7: CF+MCM" << std::endl;
        std::cout << "Valid experiment numbers are [0-6] for 0: trefoil knot, 1: leapfrogging, 2: smoke plume, 3: pyroclastic, 4: ink jet, 5: delta wing, 6: bunny meteor" << std::endl;
        exit(0);
    }
    if (experiment_name >= 7)
    {
        std::cout << "Please enter valid experiment number!" << std::endl;
        std::cout << "inputs: [Method] [Experiment]" << std::endl;
        std::cout << "Valid method numbers are [0-7] for 0: SF, 1: SF+R, 2: SCPF, 3: MC, 4: MC+R, 5: BiMocq, 6: CF, 7: CF+MCM" << std::endl;
        std::cout << "Valid experiment numbers are [0-6] for 0: trefoil knot, 1: leapfrogging, 2: smoke plume, 3: pyroclastic, 4: ink jet, 5: delta wing, 6: bunny meteor" << std::endl;
        exit(0);
    }
    sim_name = atoi(argv[1]);
    experiment_name = atoi(argv[2]);
    sim_scheme = static_cast<Scheme>(sim_name);
    sim_experiment = static_cast<Experiment>(experiment_name);
    do_EC = true;
    do_EC_with_clamp = true;
    do_2nd_order = sim_scheme == Scheme::BIMOCQ ? false : true;
    do_dmc = sim_scheme == Scheme::BIMOCQ ? true : false;
    delayed_reinit_num = (sim_scheme == Scheme::BIMOCQ || 
                          sim_scheme == Scheme::COVECTOR_BIMOCQ) ? 5 : 1;
    do_antialiasing = true;
    do_vel_advection_only = (sim_experiment == Experiment::INK_JET || 
                             sim_experiment == Experiment::PYROCLASTIC ||
                             sim_experiment == Experiment::SMOKE_PLUME) ? false : true;
    _baseres = 128;
    framerate = 24;

    std::vector<Emitter> emitter_list;
    std::vector<Boundary> boundary_list;

    float _theta = M_PI_2, _phi = 0.f;
    bool do_empty_top = false;
    bool do_sides_solid = false;
    std::string filepathVelField = "", filepathDensityRhoField = "", filepathDensityTempField = "";
    int pp_repeat_count = 1;
    bool set_velocity_inflow = false;
    bool set_init_vel = false;
    bool set_init_vel_from_emitter = false;
    bool do_true_mid_vel_covector = true;
    bool is_fixed_domain = true;
    bool do_inflow_solid = false;
    float inflow_vel = 1.f;
    switch(experiment_name)
    {
        case 0:
        {
            // simulation resolution
            int baseres = _baseres;
            ni = baseres * 2;
            nj = baseres;
            nk = baseres;
            substeps = 2;
            total_frame = 135; total_frame *= substeps;//sim_name != 2 ? 2 : 1;
            // length in x direction
            L = 10.0f;
            // grid size for simulation
            h = L / ni;
            // time step
            float deltaT = 1.f / framerate;
            dt = deltaT / substeps;
            // smoke properties
            smoke_rise = 0.f;
            smoke_drop = 0.f;
            viscosity = 0.f;// 1.0 * 1e-6;
            // blend coefficient that will blend 1-level mapping result with 2-level mapping result
            // phi_t = blend_coeff * phi_curr + (1 - blend_coeff) * phi_prev
            mapping_blend_coeff = 1.f;
            // levelset half width, used when blending semi-lagrangian result near the boundary
            half_width = 3.f;
            filepathVelField = std::string("../modelData/TrefoilKnot/trefoilKnotStaggeredVelField_") + to_string(_baseres) + std::string("cubed.vdb");
            filepathDensityTempField = std::string("../modelData/TrefoilKnot/trefoilKnotDensityField_") + to_string(_baseres) + std::string("cubed.vdb");
            filepathDensityRhoField = std::string("../modelData/TrefoilKnot/trefoilKnotDensityField_") + to_string(_baseres) + std::string("cubed.vdb");
        }
        break;
        case 1:
        {
            // simulation resolution
            int baseres = _baseres;
            ni = baseres * 2;
            nj = baseres;
            nk = baseres;
            substeps = 2;
            total_frame = 1401; total_frame *= substeps;
            // length in x direction
            L = 10.0f;
            // grid size for simulation
            h = L / ni;
            // time step
            float deltaT = 1.f / framerate;
            dt = deltaT / substeps;
            // smoke properties
            smoke_rise = 0.f;
            smoke_drop = 0.f;
            viscosity = 0.f;// 1.0 * 1e-6;
            // blend coefficient that will blend 1-level mapping result with 2-level mapping result
            // phi_t = blend_coeff * phi_curr + (1 - blend_coeff) * phi_prev
            mapping_blend_coeff = 1.f;
            // levelset half width, used when blending semi-lagrangian result near the boundary
            half_width = 3.f;
            // 2.0f and 1.2f aim for 0.6f ratio which almostmatches MC+R settings
            float innerCircleRadius = 1.2f;//0.9f;
            float outerCircleRadius = 2.0f;//1.5f;
            float thickness = 3.0 * h;
            float offest = 0.5f * h;
            auto vel_func_cylinder = [&](Vec3f pos)
            {
                Vec3f center(0.0, L / 4.0f - offest, L / 4.0f - offest);
                Vec2f samplePos(pos[1] - center[1], pos[2] - center[2]);
                float radius = sqrt(samplePos[0] * samplePos[0] + samplePos[1] * samplePos[1]);
                float circulation = 0.28;
                float vel = circulation / thickness;
                float vel_x = radius <= innerCircleRadius ? (vel * 2.0f) : (radius <= outerCircleRadius ? vel : 0.0f);
                float vel_y = 0.f;
                float vel_z = 0.f;
                return Vec3f(vel_x, vel_y, vel_z);
            };

            openvdb::FloatGrid::Ptr slab_sdf = openvdb::FloatGrid::create(20.0);
            openvdb::FloatGrid::Ptr cylinder_sdf = openvdb::FloatGrid::create(20.0);
            openvdb::CoordBBox indexBB(openvdb::Coord(0, 0, 0), openvdb::Coord(ni, nj, nk));
            makeSlab(slab_sdf, L / 2.f, 0.f, 1.5*h, Vec3f(1.5f+h, L / 4.0f, L / 4.0f), indexBB, h);
            Emitter e_slab(0, 1.f, 0.f, Vec3f(0.0f), slab_sdf,
                [](float frame)->Vec3f {return Vec3f(0.f); }, vel_func_cylinder, true);
            emitter_list.push_back(e_slab);

            set_init_vel_from_emitter = true;
        }
        break;
        case 2:
        {
            // simulation resolution
            int baseres = _baseres;
            ni = baseres;
            nj = baseres * 2;
            nk = baseres;
            substeps = 8;
            total_frame = 141; total_frame *= substeps;
            // length in y direction
            L = 5.0f;
            // grid size for simulation
            h = L / ni;
            // time step
            float deltaT = 1.f / framerate;
            dt = deltaT / substeps;
            // smoke properties
            smoke_rise = 5.0f;
            smoke_drop = 0.f;
            viscosity = 0.f;// 1.0 * 1e-6;
            // blend coefficient that will blend 1-level mapping result with 2-level mapping result
            // phi_t = blend_coeff * phi_curr + (1 - blend_coeff) * phi_prev
            mapping_blend_coeff = 1.f;
            // levelset half width, used when blending semi-lagrangian result near the boundary
            half_width = 3.f;
            auto vel_func_a = [&](Vec3f pos)
            {
                return Vec3f(0.0);

                float r = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                r = r * 2.f - 1.f;
                r *= 3.0f;
                float vel_x = 0.01 * r;
                float vel_y = 0.06f + 0.01*r;
                float vel_z = 0.01 * r;
                return Vec3f(vel_x, vel_y, vel_z);
            };

            float radius = 10.f * L / 128.f;
            openvdb::FloatGrid::Ptr sphere_sdf_a = openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(radius, openvdb::Vec3f(0.f, 0.f, 0.f), h, half_width);
            Vec2f offset(-20.f * L / 128.f);
            float height = 25.f * L / 128.f;
            float density_to_add = 1.f / substeps;
            Emitter e_sphere_a(total_frame, density_to_add, density_to_add, Vec3f(L / 2.0f + offset[0], height, L / 2.0f + offset[1]), sphere_sdf_a, [](float frame)->Vec3f {return Vec3f(0.f, 0.f, 0.f); }, vel_func_a, false);
            emitter_list.push_back(e_sphere_a);

            do_empty_top = true;
            _theta = M_PI_2 * 0.9f;
            _phi = M_PI_2 * 0.6f;
        }
        break;
        case 3:
        {
            // simulation resolution
            int baseres = _baseres;
            ni = baseres;
            nj = baseres * 2;
            nk = baseres;
            substeps = 8;
            total_frame = 121; total_frame *= substeps;
            // length in y direction
            L = 5.0f;
            // grid size for simulation
            h = L / ni;
            // time step
            float deltaT = 1.f / framerate;
            dt = deltaT / substeps;
            // smoke properties
            smoke_rise = 0.5f;
            smoke_drop = 0.f;
            viscosity = 0.f;// 1.0 * 1e-6;
            // blend coefficient that will blend 1-level mapping result with 2-level mapping result
            // phi_t = blend_coeff * phi_curr + (1 - blend_coeff) * phi_prev
            mapping_blend_coeff = 1.f;
            // levelset half width, used when blending semi-lagrangian result near the boundary
            half_width = 3.f;
            auto vel_func_a = [&](Vec3f pos)
            {
                return Vec3f(0.f);
            };

            float radius = 30.f * L / 128.f; // was 30
            Vec2f offset(-5.f * L / 128.f);
            float height = 5.f * L / 128.f;
            float density_to_add = 1.f / substeps;

            openvdb::FloatGrid::Ptr cylinder_sdf = openvdb::FloatGrid::create(20.0);
            openvdb::CoordBBox indexBB(openvdb::Coord(0, 0, 0), openvdb::Coord(ni, nj, nk));
            makeCylinderUp(cylinder_sdf, radius, height, Vec3f(L / 2.0f + offset[0], height, L / 2.0f + offset[1]), indexBB, h);
            Emitter e_cylinder_a(total_frame, density_to_add, density_to_add, Vec3f(0.f), cylinder_sdf, [](float frame)->Vec3f {return Vec3f(0.f, 0.f, 0.f); }, vel_func_a, false, true);
            emitter_list.push_back(e_cylinder_a);

            do_empty_top = true;
            _theta = M_PI_2 * 0.9f;
            _phi = M_PI_2 * 0.6f;
        }
        break;
        case 4:
        {
            // simulation resolution
            int baseres = _baseres;
            ni = baseres;
            nj = baseres * 2;
            nk = baseres;
            substeps = 2;
            total_frame = 161; total_frame *= substeps;
            // length in y direction
            L = 5.0f;
            // grid size for simulation
            h = L / ni;
            // time step
            float deltaT = 1.f / framerate;
            dt = deltaT / substeps;
            // smoke properties
            smoke_rise = 0.f;
            smoke_drop = 0.1f;
            viscosity = 0.f;// 1.0 * 1e-6;
            // blend coefficient that will blend 1-level mapping result with 2-level mapping result
            // phi_t = blend_coeff * phi_curr + (1 - blend_coeff) * phi_prev
            mapping_blend_coeff = 1.f;
            // levelset half width, used when blending semi-lagrangian result near the boundary
            half_width = 3.f;

            do_empty_top = true;
            _theta = M_PI_2 * 0.9f;
            _phi = M_PI_2 * 0.6f;
            auto vel_func_a = [&](Vec3f pos)
            {
                float speed = 2.f;
                return Vec3f(-cos(_theta) * cos(_phi) * speed, -sin(_theta) * speed, -cos(_theta) * sin(_phi) * speed);
            };

            float radius = 5.f * L / 128.f;
            openvdb::FloatGrid::Ptr sphere_sdf_a = openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(radius, openvdb::Vec3f(0.f, 0.f, 0.f), h, half_width);
            Vec2f offset(50.f * L / 128.f);
            float height = 5.f * L / 128.f;
            float density_to_add = 1.f / 8.f;
            Emitter e_sphere_a(total_frame, density_to_add, density_to_add, Vec3f(L / 2.0f + offset[0], 2 * L - height, L / 2.0f + offset[1]), sphere_sdf_a, [](float frame)->Vec3f {return Vec3f(0.f, 0.f, 0.f); }, vel_func_a, true);
            emitter_list.push_back(e_sphere_a);
        }
        break;
        case 5:
        {
            // simulation resolution
            int baseres = _baseres;
            ni = baseres;
            nj = baseres;
            nk = baseres;
            substeps = 4;
            total_frame = 201; total_frame *= substeps;
            // length in y direction
            L = 5.0f;
            // grid size for simulation
            h = L / ni;
            // time step
            float deltaT = 1.f / framerate;
            dt = deltaT / substeps;
            // smoke properties
            smoke_rise = 0.0f;
            smoke_drop = 0.f;
            viscosity = 0.f;// 1.0 * 1e-6;
            // blend coefficient that will blend 1-level mapping result with 2-level mapping result
            // phi_t = blend_coeff * phi_curr + (1 - blend_coeff) * phi_prev
            mapping_blend_coeff = 1.f;
            // levelset half width, used when blending semi-lagrangian result near the boundary
            half_width = 3.f;
            auto vel_func = [&](Vec3f pos)
            {
                return Vec3f(inflow_vel,0.f,0.f);
            };

            openvdb::FloatGrid::Ptr slab_sdf = openvdb::FloatGrid::create(20.0);
            openvdb::CoordBBox indexBB(openvdb::Coord(0, 0, 0), openvdb::Coord(ni, nj, nk));
            makeSlab(slab_sdf, L, 0.f, h, Vec3f(0.0, L / 2.0f, L / 2.0f), indexBB, h);
            Emitter e_slab(total_frame, 0.f, 0.f, Vec3f(0.f, 0.0f, 0.0f), slab_sdf,
                [](float frame)->Vec3f {return Vec3f(0.f, 0.f, 0.f); }, vel_func, true);
            emitter_list.push_back(e_slab);

            openvdb::FloatGrid::Ptr obstacle_sdf;
            std::string path_to_sdf = std::string("../modelData/DeltaWing/DeltaWingSimpleSDF_") + to_string(_baseres) + std::string("cubed.vdb");
            readVDBSDF(path_to_sdf, obstacle_sdf, "sdf");
            Boundary bdy_obstacle(Vec3f(0.f), obstacle_sdf, [](float frame)->Vec3f {return Vec3f(0.f, 0.f, 0.f); });
            boundary_list.push_back(bdy_obstacle);

            do_sides_solid = true;
            do_inflow_solid = true;
            pp_repeat_count = 1;
            set_velocity_inflow = true;
            set_init_vel = true;
        }
        break;
        case 6:
        {
            // simulation resolution
            int baseres = _baseres;
            ni = baseres*2;
            nj = baseres;
            nk = baseres;
            substeps = 4;
            total_frame = 231; total_frame *= substeps;
            // length in y direction
            L = 10.0f;
            // grid size for simulation
            h = L / ni;
            // time step
            float deltaT = 1.f / framerate;
            dt = deltaT / substeps;
            // smoke properties
            smoke_rise = 0.0f;
            smoke_drop = 0.f;
            viscosity = 0.f;// 1.0 * 1e-6;
            // blend coefficient that will blend 1-level mapping result with 2-level mapping result
            // phi_t = blend_coeff * phi_curr + (1 - blend_coeff) * phi_prev
            mapping_blend_coeff = 1.f;
            // levelset half width, used when blending semi-lagrangian result near the boundary
            half_width = 3.f;
            auto vel_func = [&](Vec3f pos)
            {
                return Vec3f(inflow_vel, 0.f, 0.f);
            };

            openvdb::FloatGrid::Ptr slab_sdf = openvdb::FloatGrid::create(20.0);
            openvdb::CoordBBox indexBB(openvdb::Coord(0, 0, 0), openvdb::Coord(ni, nj, nk));
            makeSlab(slab_sdf, L, 0.f, h, Vec3f(0.0, L / 2.0f, L / 2.0f), indexBB, h);
            Emitter e_slab(total_frame, 0.f, 0.f, Vec3f(0.f, 0.0f, 0.0f), slab_sdf,
                [](float frame)->Vec3f {return Vec3f(0.f, 0.f, 0.f); }, vel_func, true);
            emitter_list.push_back(e_slab);

            openvdb::FloatGrid::Ptr obstacle_sdf;
            std::string path_to_sdf = std::string("../modelData/bunnyMeteor/bunnySDF_") + to_string(_baseres) + std::string("cubed.vdb");
            readVDBSDF(path_to_sdf, obstacle_sdf, "sdf");
            Boundary bdy_obstacle(Vec3f(0.f), obstacle_sdf, [](float frame)->Vec3f {return Vec3f(0.f, 0.f, 0.f); });
            boundary_list.push_back(bdy_obstacle);

            do_inflow_solid = true;
            pp_repeat_count = 1;
            set_velocity_inflow = true;
            set_init_vel = true;
        }
        break;
        default:
        {

        }
    }
    std::cout << "[Experiment setup complete]" << std::endl;
    
    filepath += enumToString(sim_experiment) + "/" + enumToString(sim_scheme) + "/";
    boost::filesystem::create_directories(filepath);

	auto *myGPUmapper = new gpuMapper(ni, nj, nk, h);
	CovectorSolver mysolver(ni, nj, nk, L, viscosity, mapping_blend_coeff, sim_scheme, myGPUmapper);
    mysolver.do_true_mid_vel_covector = delayed_reinit_num == 1 && do_true_mid_vel_covector;
    mysolver.do_EC = do_EC;
    mysolver.do_EC_with_clamp = do_EC_with_clamp;
    mysolver.do_2nd_order = do_2nd_order;
    mysolver.do_dmc = do_dmc;
    mysolver.delayed_reinit_num = delayed_reinit_num;
    mysolver.do_antialiasing = do_antialiasing;
    if (do_vel_advection_only && (smoke_drop != 0.0f || smoke_rise != 0.0f))
    {
        std::cout << "There is bouyancy in this experiment, so density fields must also be advected!!!" << std::endl;
        do_vel_advection_only = false;
    }
    mysolver.do_vel_advection_only = do_vel_advection_only;
    mysolver.do_empty_top = do_empty_top;
    mysolver.do_sides_solid = do_sides_solid;
    mysolver.do_inflow_solid = do_inflow_solid;
    mysolver.theta = _theta;
    mysolver.phi = _phi;
    mysolver.pp_repeat_count = pp_repeat_count;
    mysolver.set_velocity_inflow = set_velocity_inflow;
    mysolver.is_fixed_domain = is_fixed_domain;
	mysolver.setSmoke(smoke_drop, smoke_rise, emitter_list);
    mysolver.setBoundary(boundary_list);
    mysolver.setupFromVDBFiles(filepathVelField, filepathDensityRhoField, filepathDensityTempField);
    if (set_init_vel && set_velocity_inflow)
        mysolver.setInitialVelocity(inflow_vel);
    if (set_init_vel_from_emitter)
        mysolver.setVelocityFromEmitter();
    std::cout << "[Solver setup complete]" << std::endl;
    mysolver.updateBoundary(0, dt);
    mysolver.setupPressureProjection();
    mysolver.pressureProjectVelField();
    mysolver.outputResult(0, filepath);
    //mysolver.outputResult(0, filepath);
    //return 0;
    for (uint i = 1; i < total_frame; i++)
	{
        cout << "Iteration " << i << " Starts !!!" << std::endl;
	    mysolver.updateBoundary(i, dt);
		mysolver.advance(i, dt);
        if ((experiment_name == 0) || i % int(substeps) == 0)
        {
            int framenum = (experiment_name == 0) ? i : i / int(substeps);
            mysolver.outputResult(framenum, filepath);
            cout << "Frame " << framenum << " Done !!!" << std::endl;
        }
    }
	return 0;
}