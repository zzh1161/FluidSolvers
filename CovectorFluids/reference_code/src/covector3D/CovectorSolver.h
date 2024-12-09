#ifndef COVECTORSOLVER_H
#define COVECTORSOLVER_H
#include "../include/array.h"
#include "../include/tbb/tbb.h"
#include "../include/fluid_buffer3D.h"
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <cstdio>
#include <string>
#include "../include/vec.h"
#include "../utils/pcg_solver.h"
#include "../include/array3.h"
#include "../utils/GeometricLevelGen.h"
#include "GPU_Advection.h"
#include <chrono>
#include "../utils/color_macro.h"
#include "Mapping.h"
#include "../utils/AlgebraicMultigrid.h"
#include "../utils/volumeMeshTools.h"

using namespace std;

enum Scheme {SEMILAG, REFLECTION, SCPF, MACCORMACK, MAC_REFLECTION, BIMOCQ, COVECTOR, COVECTOR_BIMOCQ};

inline std::string enumToString(const Scheme& sim_scheme)
{
    switch (sim_scheme)
    {
    case SEMILAG:
        return std::string("SF");
    case REFLECTION:
        return std::string("SF+R");
    case SCPF:
        return std::string("SCPF");
    case MACCORMACK:
        return std::string("MC");
    case MAC_REFLECTION:
        return std::string("MC+R");
    case BIMOCQ:
        return std::string("BiMocq");
    case COVECTOR:
        return std::string("CF");
    case COVECTOR_BIMOCQ:
        return std::string("CF+MCM");
    }
}

enum Experiment { TREFOIL_KNOT, LEAPFROG, SMOKE_PLUME, PYROCLASTIC, INK_JET, PLANE_WING, BUNNY_METEOR };

inline std::string enumToString(const Experiment& sim_scheme)
{
    switch (sim_scheme)
    {
    case TREFOIL_KNOT:
        return std::string("TrefoilKnot");
    case LEAPFROG:
        return std::string("Leapfrog");
    case SMOKE_PLUME:
        return std::string("SmokePlume");
    case PYROCLASTIC:
        return std::string("Pyroclastic");
    case INK_JET:
        return std::string("InkJet");
    case PLANE_WING:
        return std::string("DeltaWing");
    case BUNNY_METEOR:
        return std::string("BunnyMeteor");
    }
}

class Emitter{
public:
    Emitter() : emitFrame(0), emit_density(0.f), emit_temperature(0.f), e_pos(Vec3f(0.f)), e_sdf(nullptr),
                vel_func([](float frame)->Vec3f{return Vec3f(0.f);}),
                emit_velocity([](Vec3f pos)->Vec3f{return Vec3f(0.f);}),
                do_set_velocities(true), do_randomize_density(false) {}
    Emitter(int frame, float density, float temperature, Vec3f position, openvdb::FloatGrid::Ptr sdf,
            std::function<Vec3f(float framenum)> func,
            std::function<Vec3f(Vec3f pos)> emit_velfunc, bool set_velocities, bool randomize_density=false)
        : emitFrame(frame), emit_density(density), emit_temperature(temperature), e_pos(position), e_sdf(sdf), vel_func(func), emit_velocity(emit_velfunc), do_set_velocities(set_velocities), do_randomize_density(randomize_density){}
    ~Emitter() = default;

    int emitFrame;
    float emit_density;
    float emit_temperature;
    Vec3f e_pos;
    openvdb::FloatGrid::Ptr e_sdf;
    std::function<Vec3f(float framenum)> vel_func;
    std::function<Vec3f(Vec3f pos)> emit_velocity;
    bool do_set_velocities;
    bool do_randomize_density;

    // update levelset position
    void update(float framenum, float voxel_size, float dt)
    {
        e_pos += vel_func(framenum)*dt;
        openvdb::math::Mat4f transMat;
        transMat.setToScale(openvdb::Vec3f(voxel_size));
        transMat.setTranslation(openvdb::Vec3f(e_pos[0], e_pos[1], e_pos[2]));
        e_sdf->setTransform(openvdb::math::Transform::createLinearTransform(transMat));
    }
};

class Boundary{
public:
    Boundary(){};
    Boundary(Vec3f position, openvdb::FloatGrid::Ptr sdf, std::function<Vec3f(float framenum)> func): b_pos(position), b_sdf(sdf), vel_func(func) {}
    ~Boundary() = default;

    Vec3f b_pos;
    openvdb::FloatGrid::Ptr b_sdf;
    std::function<Vec3f(float framenum)> vel_func;

    // update levelset position
    void update(float framenum, float voxel_size, float dt)
    {
        b_pos += vel_func(framenum)*dt;
        openvdb::math::Mat4f transMat;
        transMat.setToScale(openvdb::Vec3f(voxel_size));
        transMat.setTranslation(openvdb::Vec3f(b_pos[0], b_pos[1], b_pos[2]));
        b_sdf->setTransform(openvdb::math::Transform::createLinearTransform(transMat));
    }
};

class CovectorSolver {
public:
    CovectorSolver() = default;
    CovectorSolver(uint nx, uint ny, uint nz, float L, float vis_coeff, float blend_coeff, Scheme myscheme, gpuMapper *mymapper);
    ~CovectorSolver() = default;

    void advance(int framenum, float dt);
    void advanceBimocq(int framenum, float dt);
    void advanceSemilag(int framenum, float dt);
    void advanceMacCormack(int framenum, float dt);
    void advanceReflection(int framenum, float dt, bool do_SF=false);
    void fullAdvect(int framenum, float dt, bool do_full_advect);
    void advanceCovector(int framenum, float dt);
    void advanceSCPF(int framenum, float dt);
    void blendBoundary(buffer3Df &field, const buffer3Df &blend_field);
    void velocityReinitialize();
    void scalarReinitialize();
    void addBuoyancy(buffer3Df& u_to_change, buffer3Df& v_to_change, buffer3Df& w_to_change, float dt);
    void emitSmoke(int framenum, float dt);
    void setVelocityFromEmitter(bool do_only_x_dir_vel=false);
    void setSmoke(float drop, float raise, const std::vector<Emitter> &emitters);
    void outputResult(uint frame, string filepath);
    void setupFromVDBFiles(const std::string& filepathVelField,
                           const std::string& filepathDensityRhoField,
                           const std::string& filepathDensityTempField);
    void setBoundary(const std::vector<Boundary> &boundaries);
    void updateBoundary(int framenum, float dt);
    void projection();
    void semilagAdvect(float cfldt, float dt, bool only_vel_update=false);
    void clearBoundary(buffer3Df& field);
    float getCFL();
    void clampExtrema(float dt, buffer3Df & f_n, buffer3Df & f_np1);
    void diffuse_field(double dt, double nu, buffer3Df & field);
    void pressureProjectVelField();

    void setInitialVelocity(float inflow_vel);
    void setupPressureProjection();

    // smoke parameter
    float _alpha;
    float _beta;

    // AMGPCG solver data
    SparseMatrixd matrix;
    FixedSparseMatrixd fixed_matrix;
    vector<FixedSparseMatrixd*> A_L;
    vector<FixedSparseMatrixd*> R_L;
    vector<FixedSparseMatrixd*> P_L;
    vector<Vec3i> S_L;
    int total_level;
    levelGen<double> amg_levelGen;
    std::vector<double> rhs;
    std::vector<double> pressure;
    buffer3Dc _b_desc;

    // simulation data
    uint _nx, _ny, _nz;
    float max_v;
    float _h;
    float viscosity;
    buffer3Df _un, _vn, _wn;
    buffer3Df _uinit, _vinit, _winit;
    buffer3Df _uprev, _vprev, _wprev;
    buffer3Df _utemp, _vtemp, _wtemp;
    buffer3Df _duproj, _dvproj, _dwproj;
    buffer3Df _duextern, _dvextern, _dwextern;
    buffer3Df _rho, _rhotemp, _rhoinit, _rhoprev, _drhoextern;
    buffer3Df _T, _Ttemp, _Tinit, _Tprev, _dTextern;

    buffer3Df _usolid, _vsolid, _wsolid;
    Array3c u_valid, v_valid, w_valid;
    // initialize advector
    MapperBase VelocityAdvector;
    MapperBase ScalarAdvector;
    gpuMapper *gpuSolver;
    int vel_lastReinit = 0;
    int scalar_lastReinit = 0;
    Scheme sim_scheme;

    std::vector<Emitter> sim_emitter;
    std::vector<Boundary> sim_boundary;
    bool do_EC, do_EC_with_clamp, do_2nd_order, do_dmc, do_antialiasing, do_vel_advection_only;
    int delayed_reinit_num = 1;
    bool do_empty_top = false, do_sides_solid = false, do_inflow_solid = false;
    float theta=M_PI_2, phi=0.f;
    int _extraPad = 2;
    int pp_repeat_count = 1;
    bool set_velocity_inflow = false;
    bool do_true_mid_vel_covector = true;
    bool is_fixed_domain = true;
};


#endif //COVECTORSOLVER_H
