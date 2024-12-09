#include <iostream>
#include <cstdlib>
#include "CovectorSolver2D.h"
#include <string>
#include "../utils/visualize.h"

int main(int argc, char** argv)
{
    // resolution
    int nx;
    int ny;
    // time step
    float dt;
    // simulation domain length in x-direction
    float L;
    // two level mapping blend coefficient
    float blend_coeff;
    int total_frame;
    float vorticity_distance;
    // smoke property
    float smoke_rise;
    float smoke_drop;
    // use Neumann boundary or not
    bool PURE_NEUMANN;
    Scheme sim_scheme;
    int sim_name = 0;
    int example = 0;
    if (argc != 3)
    {
        std::cout << "Please specify correct parameters!" << std::endl;
        exit(0);
    }
    sim_name = atoi(argv[1]);
    example = atoi(argv[2]);
    sim_scheme = static_cast<Scheme>(sim_name);
    int RK4_or_DMC = 0;
    int delayed_reinit_frequency = (sim_scheme == Scheme::BIMOCQ || sim_scheme == Scheme::COVECTOR_BIMOCQ) ? 5 : 1;

    std::string base_path = "../Out_2D";
    switch(example)
    {
        // Taylor-vortex
        case 0:
        {
            std::cout << GREEN << "Start running 2D Taylor Vortex example!!!" << RESET << std::endl;
            nx = 256;
            ny = 256;
            dt = 0.025;
            L = 2.f*M_PI;
            total_frame = 300;
            vorticity_distance = 0.81;
            smoke_rise = 0.f;
            smoke_drop = 0.f;
            blend_coeff = 1.f;
            PURE_NEUMANN = true;
            bool secondOrderReflection = true;
            bool secondOrderCovector = true;
            bool fiveStencil = false;
            bool do_BFECC_EC = true;
            string BFECC_string = do_BFECC_EC && (sim_scheme == Scheme::BIMOCQ || sim_scheme == Scheme::COVECTOR) ? "_BFECC_" : "";
            string resolutionString = nx == 256 ? "" : "_res" + to_string(nx);
            std::string filepath = base_path + "/2D_Taylor_vortex" + resolutionString + "/" + enumToString(sim_scheme) + BFECC_string + (((sim_scheme == Scheme::MAC_REFLECTION || sim_scheme == Scheme::REFLECTION) && secondOrderReflection) || ((sim_scheme == Scheme::COVECTOR || sim_scheme == Scheme::SCPF) && secondOrderCovector) ? "_2Order_" : "") + ((sim_scheme == Scheme::BIMOCQ || sim_scheme == Scheme::COVECTOR) && fiveStencil ? "_5stencil_" : "")  + "/";
            std::string filename = enumToString(sim_scheme) + BFECC_string + (((sim_scheme == Scheme::MAC_REFLECTION || sim_scheme == Scheme::REFLECTION) && secondOrderReflection) || ((sim_scheme == Scheme::COVECTOR || sim_scheme == Scheme::SCPF) && secondOrderCovector) ? "_2Order_" : "") + ((sim_scheme == Scheme::BIMOCQ || sim_scheme == Scheme::COVECTOR) && fiveStencil ? "_5stencil_" : "")  + "_dt_" + std::to_string(dt).substr(0,5) + "_dist_" + std::to_string(vorticity_distance).substr(0,4) +"_";
            CovectorSolver2D smokeSimulator(nx, ny, L, blend_coeff, PURE_NEUMANN, sim_scheme, fiveStencil, secondOrderCovector, RK4_or_DMC, do_BFECC_EC, secondOrderReflection);
            smokeSimulator.setSmoke(smoke_rise, smoke_drop);
			smokeSimulator.setBoundaryMask();
            smokeSimulator.buildMultiGrid(PURE_NEUMANN);
            smokeSimulator.setInitVelocity(vorticity_distance);
            smokeSimulator.calculateCurl();
            smokeSimulator.outputVorticityIntegral(filepath, 0.f);
            smokeSimulator.outputEnergy(filepath, 0.f);
            
            for (int i = 0; i < total_frame; i++)
            {
                smokeSimulator.advance(dt, i, delayed_reinit_frequency);
                smokeSimulator.calculateCurl();
                smokeSimulator.outputVortVisualized(filepath, filename, i);
                float curr_time = dt * float(i+1);
                smokeSimulator.outputVorticityIntegral(filepath, curr_time);
                smokeSimulator.outputEnergy(filepath, curr_time);
            }
        }
        break;
        // Vortex leapfrogging pairs
        case 1:
        {
            std::cout << GREEN << "Start running 2D Vortex Leapfrogging example!!!" << RESET << std::endl;
            int baseres = 256;
            nx = baseres;
            ny = baseres;
            dt = 0.025;
            L = 2.f*M_PI;
            total_frame = 2000;
            smoke_rise = 0.f;
            smoke_drop = 0.f;
            blend_coeff = 1.f;
            PURE_NEUMANN = false;
            bool secondOrderReflection = true;
            bool secondOrderCovector = true;
            bool fiveStencil = false;
            bool do_BFECC_EC = true;
            string BFECC_string = do_BFECC_EC && (sim_scheme == Scheme::BIMOCQ || sim_scheme == Scheme::COVECTOR) ? "_BFECC_" : "";
            string resolutionString = nx == 256 ? "" : "_res" + to_string(nx);
            std::string filepath = base_path + "/2D_Leapfrog" + resolutionString + "/" + enumToString(sim_scheme) + BFECC_string + (((sim_scheme == Scheme::MAC_REFLECTION || sim_scheme == Scheme::REFLECTION) && secondOrderReflection) || ((sim_scheme == Scheme::COVECTOR || sim_scheme == Scheme::SCPF) && secondOrderCovector) ? "_2Order_" : "") + ((sim_scheme == Scheme::BIMOCQ || sim_scheme == Scheme::COVECTOR) && fiveStencil ? "_5stencil_" : "")  + "/";
            std::string filename = enumToString(sim_scheme) + BFECC_string + (((sim_scheme == Scheme::MAC_REFLECTION || sim_scheme == Scheme::REFLECTION) && secondOrderReflection) || ((sim_scheme == Scheme::COVECTOR || sim_scheme == Scheme::SCPF) && secondOrderCovector) ? "_2Order_" : "") + ((sim_scheme == Scheme::BIMOCQ || sim_scheme == Scheme::COVECTOR) && fiveStencil ? "_5stencil_" : "")  + "_dt_" + std::to_string(dt).substr(0,5) +"_";
            CovectorSolver2D smokeSimulator(nx, ny, L, blend_coeff, PURE_NEUMANN, sim_scheme, fiveStencil, secondOrderCovector, RK4_or_DMC, do_BFECC_EC, secondOrderReflection);
            smokeSimulator.setSmoke(smoke_rise, smoke_drop);
			smokeSimulator.setBoundaryMask();
            smokeSimulator.buildMultiGrid(PURE_NEUMANN);
            smokeSimulator.setInitLeapFrog(1.5, 3.0, M_PI-1.6, 0.3);
            smokeSimulator.applyVelocityBoundary();
            smokeSimulator.projection_repeat_count = 2;

            for (int i = 0; i < total_frame; i++)
            {
                smokeSimulator.advance(dt, i, delayed_reinit_frequency);
                smokeSimulator.calculateCurl();
                smokeSimulator.outputVortVisualized(filepath, filename, i);
                smokeSimulator.outputDensity(filepath, "density", i, true);
            }
        }
        break;
        // Ink drop
        case 2:
        {
            std::cout << GREEN << "Start running 2D Ink drop example!!!" << RESET << std::endl;
            nx = 512;
            ny = 512;
            dt = 0.01;
            L = 0.2;
            total_frame = 80;
            smoke_rise = -0.05f;
            smoke_drop = 0.8f;
            blend_coeff = 1.0f;
            PURE_NEUMANN = false;
            bool secondOrderReflection = true;
            bool secondOrderCovector = true;
            bool fiveStencil = false;
            bool do_BFECC_EC = true;
            string BFECC_string = do_BFECC_EC && (sim_scheme == Scheme::BIMOCQ || sim_scheme == Scheme::COVECTOR) ? "_BFECC_" : "";
            string resolutionString = nx == 256 ? "" : "_res" + to_string(nx);
            std::string filepath = base_path + "/2D_RayleighTaylor" + resolutionString + "/" + enumToString(sim_scheme) + BFECC_string + (((sim_scheme == Scheme::MAC_REFLECTION || sim_scheme == Scheme::REFLECTION) && secondOrderReflection) || ((sim_scheme == Scheme::COVECTOR || sim_scheme == Scheme::SCPF) && secondOrderCovector) ? "_2Order_" : "") + ((sim_scheme == Scheme::BIMOCQ || sim_scheme == Scheme::COVECTOR) && fiveStencil ? "_5stencil_" : "")  + "/";
            std::string filename = enumToString(sim_scheme) + BFECC_string + (((sim_scheme == Scheme::MAC_REFLECTION || sim_scheme == Scheme::REFLECTION) && secondOrderReflection) || ((sim_scheme == Scheme::COVECTOR || sim_scheme == Scheme::SCPF) && secondOrderCovector) ? "_2Order_" : "") + ((sim_scheme == Scheme::BIMOCQ || sim_scheme == Scheme::COVECTOR) && fiveStencil ? "_5stencil_" : "")  + "_dt_" + std::to_string(dt).substr(0,5);
            CovectorSolver2D smokeSimulator(nx, ny, L, blend_coeff, PURE_NEUMANN, sim_scheme, fiveStencil, secondOrderCovector, RK4_or_DMC, do_BFECC_EC, secondOrderReflection);
            smokeSimulator.setSmoke(smoke_rise, smoke_drop);
			smokeSimulator.setBoundaryMask();
            smokeSimulator.buildMultiGrid(PURE_NEUMANN);
            smokeSimulator.projection_repeat_count = 2;
            smokeSimulator.setInitReyleighTaylor(0.75f * L * ny / nx);
            smokeSimulator.do_clear_boundaries = true;
            smokeSimulator.T_boundary_value = 1.f;
            smokeSimulator.rho_boundary_value = 0.f;
            for (int i = 0; i < total_frame; i++)
            {
                smokeSimulator.advance(dt, i, delayed_reinit_frequency);
                smokeSimulator.outputDensity(filepath, "density", i, true);
            }
        }
        break;
        // Ink drop SIGGRAPH logo
        case 3:
        {
            std::cout << GREEN << "Start running 2D ink drop siggraph logo example!!!" << RESET << std::endl;
            nx = 420;
            ny = 520;
            dt = 0.01;
            L = 0.2;
            total_frame = 200;
            smoke_rise = -0.05f;
            smoke_drop = 0.8f;
            blend_coeff = 1.0f;
            PURE_NEUMANN = false;
            bool secondOrderReflection = true;
            bool secondOrderCovector = true;
            bool fiveStencil = false;
            bool do_BFECC_EC = true;
            string BFECC_string = do_BFECC_EC && (sim_scheme == Scheme::BIMOCQ || sim_scheme == Scheme::COVECTOR) ? "_BFECC_" : "";
            string resolutionString = nx == 256 ? "" : "_res" + to_string(nx);
            std::string filepath = base_path + "/2D_RayleighTaylor" + resolutionString + "/" + enumToString(sim_scheme) + BFECC_string + (((sim_scheme == Scheme::MAC_REFLECTION || sim_scheme == Scheme::REFLECTION) && secondOrderReflection) || ((sim_scheme == Scheme::COVECTOR || sim_scheme == Scheme::SCPF) && secondOrderCovector) ? "_2Order_" : "") + ((sim_scheme == Scheme::BIMOCQ || sim_scheme == Scheme::COVECTOR) && fiveStencil ? "_5stencil_" : "")  + "/";
            std::string filename = enumToString(sim_scheme) + BFECC_string + (((sim_scheme == Scheme::MAC_REFLECTION || sim_scheme == Scheme::REFLECTION) && secondOrderReflection) || ((sim_scheme == Scheme::COVECTOR || sim_scheme == Scheme::SCPF) && secondOrderCovector) ? "_2Order_" : "") + ((sim_scheme == Scheme::BIMOCQ || sim_scheme == Scheme::COVECTOR) && fiveStencil ? "_5stencil_" : "")  + "_dt_" + std::to_string(dt).substr(0,5);
            CovectorSolver2D smokeSimulator(nx, ny, L, blend_coeff, PURE_NEUMANN, sim_scheme, fiveStencil, secondOrderCovector, RK4_or_DMC, do_BFECC_EC, secondOrderReflection);
            smokeSimulator.setSmoke(smoke_rise, smoke_drop);
			smokeSimulator.setBoundaryMask();
            smokeSimulator.buildMultiGrid(PURE_NEUMANN);
            smokeSimulator.projection_repeat_count = 2;
            smokeSimulator.initDensityFromFile("../modelData/sigg_logo.txt", 170, -20, 400);
            smokeSimulator.do_clear_boundaries = true;
            smokeSimulator.T_boundary_value = 1.f;
            smokeSimulator.rho_boundary_value = 0.f;
            for (int i = 0; i < total_frame; i++)
            {
                smokeSimulator.advance(dt, i, delayed_reinit_frequency);
                smokeSimulator.outputDensity(filepath, "density", i, true);
            }
        }
        break;
        // Covector transport with Zalesak's disk
        case 4:
        {
            std::cout << GREEN << "Start running 2D Inverted Zalesak's Disk example!!!" << RESET << std::endl;
            nx = 200;
            ny = 200;
            L = 1;
            dt = 2;
            total_frame = 315;
            smoke_rise = 0.f;
            smoke_drop = 0.f;
            blend_coeff = 1.f;
            PURE_NEUMANN = true;
            bool secondOrderReflection = false;
            bool secondOrderCovector = false;
            bool fiveStencil = false;
            bool do_BFECC_EC = true;
            string BFECC_string = do_BFECC_EC && (sim_scheme == Scheme::BIMOCQ || sim_scheme == Scheme::COVECTOR) ? "_BFECC_" : "";
            string resolutionString = nx == 200 ? "" : "_res" + to_string(nx);
            std::string filepath = base_path + "/2D_InvertedZalesak" + resolutionString + "/" + enumToString(sim_scheme) + BFECC_string + (((sim_scheme == Scheme::MAC_REFLECTION || sim_scheme == Scheme::REFLECTION) && secondOrderReflection) || ((sim_scheme == Scheme::COVECTOR || sim_scheme == Scheme::SCPF) && secondOrderCovector) ? "_2Order_" : "") + ((sim_scheme == Scheme::BIMOCQ || sim_scheme == Scheme::COVECTOR) && fiveStencil ? "_5stencil_" : "")  + "/";
            std::string filename = enumToString(sim_scheme) + BFECC_string + (((sim_scheme == Scheme::MAC_REFLECTION || sim_scheme == Scheme::REFLECTION) && secondOrderReflection) || ((sim_scheme == Scheme::COVECTOR || sim_scheme == Scheme::SCPF) && secondOrderCovector) ? "_2Order_" : "") + ((sim_scheme == Scheme::BIMOCQ || sim_scheme == Scheme::COVECTOR) && fiveStencil ? "_5stencil_" : "")  + "_";
            CovectorSolver2D smokeSimulator(nx, ny, L, blend_coeff, PURE_NEUMANN, sim_scheme, fiveStencil, secondOrderCovector, RK4_or_DMC, do_BFECC_EC, secondOrderReflection);
            smokeSimulator.advect_levelset = true;
            smokeSimulator.advect_levelset_covector = true;
            smokeSimulator.setSmoke(smoke_rise, smoke_drop);
			smokeSimulator.setBoundaryMask();
            smokeSimulator.buildMultiGrid(PURE_NEUMANN);
            smokeSimulator.setInitInvertedZalesak();
            smokeSimulator.outputCovectorField(filepath, "grad_density", 0);
            for (int i = 1; i < total_frame; i++)
            {
                smokeSimulator.advance(dt, i, delayed_reinit_frequency);
                smokeSimulator.outputCovectorField(filepath, "grad_density", i);
            }
        }
        break;
        // von Kármán vortex Street example
        case 5:
        {
            std::cout << GREEN << "Start running Kármán vortex Street example!!!" << RESET << std::endl;
            nx = 512;
            ny = 256;
            dt = 0.1;
            L = 2.f*M_PI;
            total_frame = 300;
            smoke_rise = 0.f;
            smoke_drop = 0.f;
            blend_coeff = 1.f;
            PURE_NEUMANN = false;
            bool secondOrderReflection = true;
            bool secondOrderCovector = true;
            bool fiveStencil = false;
            bool do_BFECC_EC = true;
            string BFECC_string = do_BFECC_EC && (sim_scheme == Scheme::BIMOCQ || sim_scheme == Scheme::COVECTOR) ? "_BFECC_" : "";
            string resolutionString = nx == 512 ? "" : "_res" + to_string(nx);
            std::string filepath = base_path + "/Karman_vortex" + resolutionString + "/" + enumToString(sim_scheme) + BFECC_string + (((sim_scheme == Scheme::MAC_REFLECTION || sim_scheme == Scheme::REFLECTION) && secondOrderReflection) || ((sim_scheme == Scheme::COVECTOR || sim_scheme == Scheme::SCPF) && secondOrderCovector) ? "_2Order_" : "") + ((sim_scheme == Scheme::BIMOCQ || sim_scheme == Scheme::COVECTOR) && fiveStencil ? "_5stencil_" : "")  + "/";
            std::string filename = enumToString(sim_scheme) + BFECC_string + (((sim_scheme == Scheme::MAC_REFLECTION || sim_scheme == Scheme::REFLECTION) && secondOrderReflection) || ((sim_scheme == Scheme::COVECTOR || sim_scheme == Scheme::SCPF) && secondOrderCovector) ? "_2Order_" : "") + ((sim_scheme == Scheme::BIMOCQ || sim_scheme == Scheme::COVECTOR) && fiveStencil ? "_5stencil_" : "")  + "_dt_" + std::to_string(dt).substr(0,5) + "_dist_" + std::to_string(vorticity_distance).substr(0,4) +"_";
            
            CovectorSolver2D smokeSimulator(nx, ny, L, blend_coeff, PURE_NEUMANN, sim_scheme, fiveStencil, secondOrderCovector, RK4_or_DMC, do_BFECC_EC, secondOrderReflection);
            smokeSimulator.setSmoke(smoke_rise, smoke_drop);

            auto sphere_obstacle_boundary_func = [&](Vec2f pos)
            {
                Vec2f center(L/8.f, L/4.f+0.01f);//L/8.f);
                float radius = L/30.f;
                float dist = mag(Vec2f(pos[0] - center[0], pos[1] - center[1])) - radius;

                return dist;
            };
            smokeSimulator.do_karman_velocity_setup = true;
            smokeSimulator.karman_velocity_value = 0.5f;
            smokeSimulator.projection_repeat_count = 2;
            smokeSimulator.initKarmanVelocity();
            smokeSimulator.setBoundaryMask(sphere_obstacle_boundary_func);
            smokeSimulator.buildMultiGrid(PURE_NEUMANN);
            smokeSimulator.pressureProjectVelField();
            smokeSimulator.calculateCurl(true);
            smokeSimulator.outputVortVisualized(filepath, filename, 0);
            smokeSimulator.outputVellVisualized(filepath, filename, 0);
            smokeSimulator.outputVellVisualized(filepath, filename, 0, true);
            smokeSimulator.setKarmanDensity(true);
            smokeSimulator.outputDensity(filepath, "density", 0, true, true);
            for (int i = 0; i < total_frame; i++)
            {
                smokeSimulator.advance(dt, i, delayed_reinit_frequency);
                smokeSimulator.calculateCurl();
                smokeSimulator.outputVortVisualized(filepath, filename, i);
                smokeSimulator.outputVellVisualized(filepath, filename, i);
                smokeSimulator.outputVellVisualized(filepath, filename, i, true);
                smokeSimulator.outputDensity(filepath, "density", i, true, true);
            }
        }
        break;
    }

    return 0;
}