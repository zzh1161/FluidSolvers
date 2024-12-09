#ifndef COVECTORSOLVER2D_H
#define COVECTORSOLVER2D_H
#include <stdlib.h>
#include <fcntl.h>
#include <stdio.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include "tbb/tbb.h"
#include "../include/array2.h"
#include "../include/vec.h"
#include "../utils/AlgebraicMultigrid.h"
#include "../utils/GeometricLevelGen.h"
#include "../utils/writeBMP.h"
#include "../utils/visualize.h"
#include "../utils/color_macro.h"
#include <boost/filesystem.hpp>

enum Scheme {SEMILAG, REFLECTION, SCPF, MACCORMACK, MAC_REFLECTION, BIMOCQ, COVECTOR, COVECTOR_BIMOCQ };

inline std::string enumToString(const Scheme &sim_scheme)
{
    switch(sim_scheme)
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

class CovectorSolver2D {
public:
    void clampPos(Vec2f &pos)
    {
        pos[0] = min(max(0.0f*h, pos[0]),float(ni*h)-0.0f*h);
        pos[1] = min(max(0.0f*h, pos[1]),float(nj*h)-0.0f*h);
    }
    CovectorSolver2D(int nx, int ny, float L, float b_coeff, bool bc, Scheme s_scheme, bool use5PointStencil = true,
                   bool secondOrderCovector = false, int SFOrMCOrDMC = 2, bool doBFECCEC = false, bool secondOrderReflection = false);
    ~CovectorSolver2D() {};
    int ni, nj;
    inline	float lerp(float v0, float v1, float c);
    inline	float bilerp(float v00, float v01, float v10, float v11, float cx, float cy);
    void semiLagAdvect(const Array2f &src, Array2f & dst, float dt, int ni, int nj, float off_x, float off_y);
    void solveMaccormack(const Array2f &src, Array2f &dst, Array2f & aux, float dt, int ni, int nj, float offsetx, float offsety);
    void applyBuoyancyForce(Array2f &v, float dt);
    void calculateCurl(bool set_color_bar_with_max_curl=false);
    void projection(float tol, bool PURE_NEUMANN);
    void pressureProjectVelField();

    void resampleVelBuffer(float dt);
    void resampleRhoBuffer(float dt);

    void advance(float dt, int frame, int delayed_reinit_frequency=1);
    void advanceSemilag(float dt, int currentframe);
    void advanceReflection(float dt, int currentframe, bool do_SemiLag = false);
    void advanceMaccormack(float dt, int currentframe);
    void advanceBIMOCQ(float dt, int currentframe, int delayed_reinit_frequency=1);
    void advanceCovector(float dt, int currentframe, int delayed_reinit_frequency=1);
    void advanceSCPF(float dt, int currentframe);

    void clampExtrema2(int _ni, int _nj,Array2f &before, Array2f &after);
    void updateForward(float dt, Array2f &fwd_x, Array2f &fwd_y);
    void updateBackward(float dt, Array2f &back_x, Array2f &back_y);

    void advectVelocity();
    void correctVelocity();
    void correctScalars();
    void advectScalars();
    void accumulateVelocity(Array2f &u_change, Array2f &v_change, float proj_coeff, bool error_correction);
    void accumulateScalars(Array2f &rho_change, Array2f &T_change, bool error_correction);
    void advectQuantity();
    void correctQuantity();

    void fullAdvect(float dt, bool advect_full_dt, bool do_mid_velocity);
    float calcDPsiTRow(const Vec2f& pos, const Vec2f& offset, 
                       const Array2f& u_sample, const Vec2f& u_offset, 
                       const Array2f& v_sample, const Vec2f& v_offset, 
                       const Array2f& map_x, const Array2f& map_y, float dt, bool do_mid_velocity, bool do_trapezoidal_rule=false);

    void advectVelocityCovector(float dt, bool do_mid_velocity);
    void correctVelocityCovector(float dt, bool do_mid_velocity);
    void advectScalarsCovector();
    void correctScalarsCovector();
    void advectQuantityCovector(float dt, bool do_mid_velocity);
    void correctQuantityCovector(float dt, bool do_mid_velocity);
    void accumulateVelocityCovector(Array2f &u_change, Array2f &v_change, float dt, bool do_mid_velocity);

    void advectVelocitySCPF(float dt);

    void buildMultiGrid(bool PURE_NEUMANN);
    void diffuseField(float nu, float dt, Array2f &field);
    void applyVelocityBoundary(bool do_set_obstacle_vel = true);


    /// new scheme for SEMILAG advection
    Vec2f calculateA(Vec2f pos, float h);
    void semiLagAdvectDMC(const Array2f &src, Array2f & dst, float dt, int ni, int nj, float off_x, float off_y);
    inline Vec2f solveODEDMC(float dt, const Vec2f &pos);
    inline Vec2f traceDMC(float dt, const Vec2f &pos, Vec2f &a);

    void setInitReyleighTaylor(float layer_height);
    void setInitVelocity(float distance, bool do_taylor_green=false);
    void setInitLeapFrog(float dist1, float dist2, float rho_h, float rho_w);
    void setInitZalesak();
    void setInitInvertedZalesak();
    void setInitVortexBox();
    void setSmoke(float smoke_rise, float smoke_drop);
    void setBoundaryMask(std::function<float(Vec2f pos)> sdf=nullptr);
    void setKarmanVelocity();
    void initKarmanVelocity();
    void setKarmanDensity(bool do_init=false);

    float maxVel();
    float estimateDistortion(Array2f &back_x, Array2f &back_y, Array2f &fwd_x, Array2f &fwd_y);
	inline Vec2f traceFE(float dt, const Vec2f &pos, const Array2f& un, const Array2f& vn);
	inline Vec2f traceRK3(float dt, const Vec2f &pos, const Array2f& un, const Array2f& vn);
    inline Vec2f traceRK4(float dt, const Vec2f& pos, const Array2f& un, const Array2f& vn);
    inline Vec2f solveODE(float dt, const Vec2f &pos, const Array2f& un, const Array2f& vn);
    void emitSmoke();
    void outputDensity(std::string folder, std::string file, int i, bool color_density, bool do_tonemapping=false);
    void outputCovectorField(std::string folder, std::string file, int i);
    void outputVortVisualized(std::string folder, std::string file, int i);
    void outputVellVisualized(std::string folder, std::string file, int i, bool do_y_comp=false);
    void outputLevelset(std::string sdfFilename, int i);
    void outputEnergy(std::string filename, float curr_time);
    void outputVorticityIntegral(std::string filename, float curr_time);
    void outputError(std::string filename, float curr_time);
    void outputErrorVisualized(std::string folder, std::string file, int i);
    void initDensityFromFile(std::string filepath, int pixel_shift_up, int pixel_shift_right, int img_size);
    void clearBoundaries();
    
    color_bar cBar;
    int total_resampleCount = 0;
    int total_scalar_resample = 0;
    int resampleCount = 0;
    int frameCount = 0;
    void getCFL();
    Vec2f getVelocity(const Vec2f &pos, const Array2f& un, const Array2f& vn);
    float sampleField(const Vec2f pos, const Array2f &field);

    float h, lengthScale;
    float alpha, beta;
    Array2f u, v, u_temp, v_temp;
    Array2f rho, temperature, s_temp;
    Array2f curl;
    Array2c emitterMask;
    Array2c boundaryMask;
    std::vector<double> pressure;
    std::vector<double> rhs;

    //linear solver data
    SparseMatrixd matrix;
    FixedSparseMatrixd matrix_fix;
    std::vector<FixedSparseMatrixd *> A_L;
    std::vector<FixedSparseMatrixd *> R_L;
    std::vector<FixedSparseMatrixd *> P_L;
    std::vector<Vec2i>                S_L;
    int total_level;
    //solver
    levelGen<double> mgLevelGenerator;

    float _cfl;

    // BIMOCQ mapping buffers
    Array2f forward_x;
    Array2f forward_y;
    Array2f forward_scalar_x;
    Array2f forward_scalar_y;
    Array2f backward_x;
    Array2f backward_y;
    Array2f backward_xprev;
    Array2f backward_yprev;
    Array2f backward_scalar_x;
    Array2f backward_scalar_y;
    Array2f backward_scalar_xprev;
    Array2f backward_scalar_yprev;
    Array2f map_tempx;
    Array2f map_tempy;
    Array2f map_tempx2;
    Array2f map_tempy2;

    // fluid buffers
    Array2f u_init;
    Array2f v_init;
    Array2f u_origin;
    Array2f v_origin;
    Array2f du;
    Array2f dv;
    Array2f du_temp;
    Array2f dv_temp;
    Array2f du_proj;
    Array2f dv_proj;
    Array2f drho;
    Array2f drho_temp;
    Array2f drho_prev;
    Array2f dT;
    Array2f dT_temp;
    Array2f dT_prev;
    Array2f rho_init;
    Array2f rho_orig;
    Array2f T_init;
    Array2f T_orig;
    Array2f cv_u;
    Array2f cv_v;
    Array2f cv_u_init;
    Array2f cv_v_init;

    // for Maccormack
    Array2f u_first;
    Array2f v_first;
    Array2f u_sec;
    Array2f v_sec;

    Array2f du_prev;
    Array2f dv_prev;

    Array2f u_flow;
    Array2f v_flow;

    int lastremeshing = 0;
    int rho_lastremeshing = 0;
    float blend_coeff;
    bool use_neumann_boundary;
    Scheme sim_scheme;
    bool advect_levelset = false;
    bool advect_levelset_covector = false;

    bool use_5_point_stencil;
    float w[6] = { 0.125f,0.125f,0.125f,0.125f,0.5f,1.0f };
    int start_idx;
    int end_idx;
    std::vector<Vec2f> dir;

    bool second_order_covector;
    bool second_order_ref;

    int reinit_iters;
    int RK4_or_DMC;
    bool do_BFECC_EC;

    bool do_real_mid_vel = false;

    int projection_repeat_count = 1;
    bool do_karman_velocity_setup = false;
    float karman_velocity_value = 0.f;
    bool do_clear_boundaries = false;
    float T_boundary_value = 0.f;
    float rho_boundary_value = 0.f;

    Vec3f HSVtoRGB(float H, float S,float V)
    {
        H = clamp(H, 0.f, 360.f);
        S = clamp(S, 0.f, 100.f);
        V = clamp(V, 0.f, 100.f);

        float s = S/100;
        float v = V/100;
        float C = s*v;
        float X = C*(1-abs(fmod(H/60.0, 2)-1));
        float m = v-C;
        float r,g,b;
        if(H >= 0 && H < 60){
            r = C,g = X,b = 0;
        }
        else if(H >= 60 && H < 120){
            r = X,g = C,b = 0;
        }
        else if(H >= 120 && H < 180){
            r = 0,g = C,b = X;
        }
        else if(H >= 180 && H < 240){
            r = 0,g = X,b = C;
        }
        else if(H >= 240 && H < 300){
            r = X,g = 0,b = C;
        }
        else{
            r = C,g = 0,b = X;
        }
        float R = r+m;
        float G = g+m;
        float B = b+m;

        return Vec3f(R,G,B);
    }

private:
    float m_max_curl = 0.f;
};

#endif //COVECTORSOLVER2D_H
