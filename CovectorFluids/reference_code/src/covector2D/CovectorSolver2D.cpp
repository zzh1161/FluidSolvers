#include "CovectorSolver2D.h"

inline Vec2f CovectorSolver2D::traceFE(float dt, const Vec2f &pos, const Array2f& un, const Array2f& vn)
{
	Vec2f input = pos;
	Vec2f velocity = getVelocity(input, un, vn);
	input = input + dt*velocity;
	clampPos(input);
	return input;
}

inline Vec2f CovectorSolver2D::traceRK3(float dt, const Vec2f &pos, const Array2f& un, const Array2f& vn)
{
	float c1 = 2.0 / 9.0*dt, c2 = 3.0 / 9.0 * dt, c3 = 4.0 / 9.0 * dt;
	Vec2f input = pos;
	Vec2f velocity1 = getVelocity(input, un, vn);
	Vec2f midp1 = input + ((float)(0.5*dt))*velocity1;
	Vec2f velocity2 = getVelocity(midp1, un, vn);
	Vec2f midp2 = input + ((float)(0.75*dt))*velocity2;
	Vec2f velocity3 = getVelocity(midp2, un, vn);
	input = input + c1*velocity1 + c2*velocity2 + c3*velocity3;
	clampPos(input);
	return input;
}

inline Vec2f CovectorSolver2D::traceRK4(float dt, const Vec2f& pos, const Array2f& un, const Array2f& vn)
{
    float c1 = 1.0 / 6.0 * dt, c2 = 1.0 / 3.0 * dt, c3 = 1.0 / 3.0 * dt, c4 = 1.0 / 6.0 * dt;
    Vec2f input = pos;
    Vec2f velocity1 = getVelocity(input, un, vn);
    Vec2f midp1 = input + ((float)(0.5 * dt)) * velocity1;
    Vec2f velocity2 = getVelocity(midp1, un, vn);
    Vec2f midp2 = input + ((float)(0.5 * dt)) * velocity2;
    Vec2f velocity3 = getVelocity(midp2, un, vn);
    Vec2f midp3 = input + ((float)(dt)) * velocity3;
    Vec2f velocity4 = getVelocity(midp3, un, vn);
    input = input + c1 * velocity1 + c2 * velocity2 + c3 * velocity3 + c4 * velocity4;
    clampPos(input);
    return input;
}

inline Vec2f CovectorSolver2D::solveODE(float dt, const Vec2f &pos, const Array2f& un, const Array2f& vn)
{
    float ddt = dt;
    Vec2f pos1 = traceRK4(ddt, pos, un, vn);
    ddt/=2.0;
    int substeps = 2;
    Vec2f pos2 = traceRK4(ddt, pos, un, vn);pos2 = traceRK4(ddt, pos2, un, vn);
    int iter = 0;

    while(dist(pos2,pos1)>0.0001*h && iter<6)
    {
        pos1 = pos2;
        ddt/=2.0;
        substeps *= 2;
        pos2 = pos;
        for(int j=0;j<substeps;j++)
        {
            pos2 = traceRK4(ddt, pos2, un, vn);
        }
        iter++;
    }
    return pos2;
}

inline Vec2f CovectorSolver2D::solveODEDMC(float dt, const Vec2f &pos)
{
    Vec2f a = calculateA(pos, h);
    Vec2f opos=pos;
    opos = traceDMC(dt, opos, a);
    return opos;
}

void CovectorSolver2D::getCFL()
{
    _cfl = h / fabs(maxVel());
}

inline Vec2f CovectorSolver2D::traceDMC(float dt, const Vec2f &pos, Vec2f &a)
{
    Vec2f vel = getVelocity(pos, u, v);
    float new_x = pos[0] - dt*vel[0];
    float new_y = pos[1] - dt*vel[1];
    if (fabs(a[0]) >1e-4) new_x = pos[0] - (1-exp(-a[0]*dt))*vel[0]/(a[0]);
    else new_x = solveODE(-dt, pos, u, v)[0];

    if (fabs(a[1]) >1e-4) new_y = pos[1] - (1-exp(-a[1]*dt))*vel[1]/(a[1]);
    else new_y = solveODE(-dt, pos, u, v)[1];
    return Vec2f(new_x, new_y);
}

inline float CovectorSolver2D::lerp(float v0, float v1, float c)
{
    return (1-c)*v0+c*v1;
}

inline float CovectorSolver2D::bilerp(float v00, float v01, float v10, float v11, float cx, float cy)
{
    return lerp(lerp(v00,v01,cx), lerp(v10,v11,cx),cy);
}

Vec2f CovectorSolver2D::calculateA(Vec2f pos, float h)
{
    Vec2f vel = getVelocity(pos, u, v);
    float new_x = (vel[0] > 0)? pos[0]-h : pos[0]+h;
    float new_y = (vel[1] > 0)? pos[1]-h : pos[1]+h;
    Vec2f new_pos = Vec2f(new_x, new_y);
    Vec2f new_vel = getVelocity(new_pos, u, v);
    float a_x = (vel[0] - new_vel[0]) / (pos[0] - new_pos[0]);
    float a_y = (vel[1] - new_vel[1]) / (pos[1] - new_pos[1]);
    return Vec2f(a_x, a_y);
}

void CovectorSolver2D::semiLagAdvectDMC(const Array2f &src, Array2f & dst, float dt, int ni, int nj, float off_x, float off_y)
{
    tbb::parallel_for((int)0,
                      (int)(ni*nj),
                      (int)1,
                      [&](int tId)
                      {
                          int j = tId / ni;
                          int i = tId % ni;
                          Vec2f pos = h*Vec2f(i, j) + h*Vec2f(off_x, off_y);
                          Vec2f back_pos = solveODEDMC(dt, pos);
                          dst(i, j) = sampleField(back_pos - h*Vec2f(off_x, off_y), src);
                      });
}


void CovectorSolver2D::semiLagAdvect(const Array2f & src, Array2f & dst, float dt, int ni, int nj, float off_x, float off_y)
{
    tbb::parallel_for((int)0,
                      (int)(ni*nj),
                      (int)1,
                      [&](int tId)
                      {
                          int j = tId / ni;
                          int i = tId % ni;
                          Vec2f pos = h*Vec2f(i, j) + h*Vec2f(off_x, off_y);
						  Vec2f back_pos = solveODE(-dt, pos, u, v);
                          dst(i, j) = sampleField(back_pos - h*Vec2f(off_x, off_y), src);
                      });
}

void CovectorSolver2D::advance(float dt, int frame, int delayed_reinit_frequency)
{
    switch(sim_scheme)
    {
        case SEMILAG:
            advanceSemilag(dt, frame);
            break;
        case REFLECTION:
            advanceReflection(dt, frame, true);
            break;
        case SCPF:
            advanceSCPF(dt, frame);
            break;
        case MACCORMACK:
            advanceMaccormack(dt, frame);
            break;
        case MAC_REFLECTION:
            advanceReflection(dt, frame);
            break;
        case BIMOCQ:
            advanceBIMOCQ(dt, frame, delayed_reinit_frequency);
            break;
        case COVECTOR:
            advanceCovector(dt, frame);
            break;
        case COVECTOR_BIMOCQ:
            advanceCovector(dt, frame, delayed_reinit_frequency);
            break;
        default:
            break;
    }
}

CovectorSolver2D::CovectorSolver2D(int nx, int ny, float L, float b_coeff, bool bc, Scheme s_scheme,
                               bool use5PointStencil, bool secondOrderCovector, int SFOrMCOrDMC,
                               bool doBFECCEC, bool secondOrderReflection)
{
    blend_coeff = b_coeff;
    use_neumann_boundary = bc;
    lengthScale = L;
    h = L / (float)nx;
    ni = nx;
    nj = ny;
    u.resize(nx + 1, ny, 0);
    v.resize(nx, ny + 1, 0);
    u_temp.resize(nx + 1, ny, 0);
    v_temp.resize(nx, ny + 1, 0);
    rho.resize(nx, ny, 0);
    temperature.resize(nx, ny, 0);
    s_temp.resize(nx, ny, 0);
    pressure.resize(nx*ny);
    std::fill(pressure.begin(), pressure.end(), 0);
    rhs.resize(nx*ny);
    std::fill(rhs.begin(), rhs.end(), 0);
    emitterMask.resize(nx, ny, 0);
    boundaryMask.resize(nx,ny, 0);
    curl.resize(nx+1, ny+1, 0);
    cv_u.resize(nx+1, ny, 0);
    cv_v.resize(nx, ny+1, 0);
    cv_u_init.resize(nx+1, ny, 0);
    cv_v_init.resize(nx, ny+1, 0);

    u_flow.resize(nx + 1, ny, 0);
    v_flow.resize(nx, ny + 1, 0);

    // for maccormack
    u_first.resize(ni+1, nj, 0);
    v_first.resize(ni, nj+1, 0);
    u_sec.resize(ni+1, nj, 0);
    v_sec.resize(ni, nj+1, 0);

    // init BIMOCQ
    du.resize(nx+1, ny, 0);
    du_temp.resize(nx+1, ny, 0);
    du_proj.resize(nx+1, ny, 0);
    dv.resize(nx, ny+1, 0);
    dv_temp.resize(nx, ny+1, 0);
    dv_proj.resize(nx, ny+1, 0);
    drho.resize(nx, ny, 0);
    drho_temp.resize(nx, ny, 0);
    drho_prev.resize(nx, ny, 0);

    dT.resize(nx, ny, 0);
    dT_temp.resize(nx, ny, 0);
    dT_prev.resize(nx, ny, 0);

    u_init.resize(nx+1, ny, 0);
    v_init.resize(nx, ny+1, 0);
    rho_init.resize(nx, ny, 0);
    rho_orig.resize(nx, ny, 0);
    T_init.resize(nx, ny, 0);
    T_orig.resize(nx, ny, 0);

    u_origin.resize(ni+1,nj, 0);

    v_origin.resize(ni,nj+1, 0);

    du_prev.resize(ni+1,nj, 0);

    dv_prev.resize(ni,nj+1, 0);

    // mapping buffers
    forward_x.resize(nx, ny, 0);
    forward_y.resize(nx, ny, 0);
    backward_x.resize(nx, ny, 0);
    backward_y.resize(nx, ny, 0);
    forward_scalar_x.resize(nx, ny, 0);
    forward_scalar_y.resize(nx, ny, 0);
    backward_scalar_x.resize(nx, ny, 0);
    backward_scalar_y.resize(nx, ny, 0);
    backward_xprev.resize(nx, ny, 0);
    backward_yprev.resize(nx, ny, 0);
    backward_scalar_xprev.resize(nx, ny, 0);
    backward_scalar_yprev.resize(nx, ny, 0);
    map_tempx.resize(nx, ny, 0);
    map_tempy.resize(nx, ny, 0);
    map_tempx2.resize(ni, nj, 0.0f);
    map_tempy2.resize(ni, nj, 0.0f);

    tbb::parallel_for((int)0, ni*nj, 1, [&](int tIdx)
    {
        int i = tIdx%ni; int j = tIdx / ni;
        forward_x(i, j) = h*((float)i + 0.5);
        backward_x(i, j) = h*((float)i + 0.5);
        forward_y(i, j) = h*((float)j + 0.5);
        backward_y(i, j) = h*((float)j + 0.5);
    });
    backward_scalar_x = backward_x;
    backward_scalar_y = backward_y;
    forward_scalar_x = forward_x;
    forward_scalar_y = forward_y;
    backward_xprev = backward_x;
    backward_yprev = backward_y;
    backward_scalar_xprev = backward_x;
    backward_scalar_yprev = backward_y;

    sim_scheme = s_scheme;

    use_5_point_stencil = use5PointStencil;
    start_idx = use_5_point_stencil ? 0 : 5;
    end_idx = use_5_point_stencil ? 5 : 6;
    dir.push_back(Vec2f(-0.25, -0.25));
    dir.push_back(Vec2f(0.25, -0.25));
    dir.push_back(Vec2f(-0.25, 0.25));
    dir.push_back(Vec2f(0.25, 0.25));
    dir.push_back(Vec2f(0.0, 0.0));
    dir.push_back(Vec2f(0.0, 0.0));

    second_order_covector = secondOrderCovector;
    second_order_ref = secondOrderReflection;

    RK4_or_DMC = SFOrMCOrDMC;
    do_BFECC_EC = doBFECCEC;
}

void CovectorSolver2D::solveMaccormack(const Array2f &src, Array2f &dst, Array2f & aux, float dt, int ni, int nj, float offsetx, float offsety)
{
    semiLagAdvect(src, dst, dt, ni, nj, offsetx, offsety);
    Array2f dst_copy(dst);
    semiLagAdvect(dst, aux, -dt, ni, nj, offsetx, offsety);
    // clamp extrema
    tbb::parallel_for((int)0, (dst.ni)*dst.nj, 1, [&](int tIdx) {
        int i = tIdx%(dst.ni);
        int j = tIdx / (dst.ni);
        dst(i, j) = dst(i,j) + 0.5*(src(i,j) - aux(i,j));
    });
    clampExtrema2(ni, nj, dst_copy, dst);
}

void CovectorSolver2D::applyBuoyancyForce(Array2f &_v, float dt)
{
    /// NOTE: this function is used for Rayleigh-Taylor example, where rho and temperature represent two kinds of fluid
    /// with different density, so both rho and temperature act like drop force
    /// for smoke, you may want temperature acts like rising force, which change the - beta*temperature(i,j) to be + beta*temperature(i,j)
    tbb::parallel_for((int)0, ni*nj, 1, [&](int tIdx)
    {
        int i = tIdx%ni;
        int j = tIdx / ni;
        _v(i, j) += 0.5*dt*(-alpha*rho(i, j) + beta*temperature(i, j));
    });
    tbb::parallel_for((int)0, ni*nj, 1, [&](int tIdx)
    {
        int i = tIdx%ni;
        int j = tIdx / ni;
        _v(i, j + 1) += 0.5*dt*(-alpha*rho(i, j) + beta*temperature(i, j));
    });
}
void CovectorSolver2D::pressureProjectVelField()
{
    projection(1e-6, use_neumann_boundary);
    u_init = u;
    v_init = v;
    u_origin = u;
    v_origin = v;
}

void CovectorSolver2D::projection(float tol, bool PURE_NEUMANN)
{
    if (do_karman_velocity_setup)
        setKarmanDensity();
    for (int count = 0; count < projection_repeat_count; count++)
    {
        applyVelocityBoundary();
        if (do_karman_velocity_setup)
            setKarmanVelocity();

        rhs.assign(ni*nj, 0.0);
        pressure.assign(ni*nj, 0.0);
        //build rhs;
        tbb::parallel_for((int)0, ni*nj, 1, [&](int tIdx)
        {
            int i = tIdx%ni;
            int j = tIdx / ni;
            rhs[tIdx] = -(u(i+1,j) - u(i,j) + v(i,j+1) - v(i,j)) / h;
        });
        double res_out; int iter_out;
        bool converged = AMGPCGSolvePrebuilt2D(matrix_fix,rhs,pressure,A_L,R_L,P_L,S_L,total_level,(double)tol,500,res_out,iter_out,ni,nj, PURE_NEUMANN);

        if (converged)
            std::cout << "pressure solver converged in " << iter_out << " iterations, with residual " << res_out << std::endl;
        else
            std::cout << "warning! solver didn't reach convergence with maximum iterations!" << std::endl;
        //subtract gradient
        tbb::parallel_for((int)0, ni*nj, 1, [&](int tIdx)
        {
            int i = tIdx%ni;
            int j = tIdx / ni;
            u(i, j) -= pressure[tIdx] / h;
            v(i, j) -= pressure[tIdx] / h;
        });
        tbb::parallel_for((int)0, ni*nj, 1, [&](int tIdx)
        {
            int i = tIdx%ni;
            int j = tIdx / ni;
            u(i + 1, j) += pressure[tIdx] / h;
            v(i, j + 1) += pressure[tIdx] / h;
        });

        applyVelocityBoundary();
        if (do_karman_velocity_setup)
            setKarmanVelocity();
    }
}

void CovectorSolver2D::advanceBIMOCQ(float dt, int currentframe, int delayed_reinit_frequency)
{
    std::cout << BLUE <<  "BIMOCQ scheme frame " << currentframe << " starts !" << RESET << std::endl;

    float proj_coeff = 2.0;
    getCFL();
    if (currentframe != 0 && !advect_levelset)
    {
        u = u_temp;
        v = v_temp;
    }
    frameCount++;
    resampleCount++;

    /// update Backward & Forward mapping
    if (!advect_levelset)
    {
        updateForward(dt, forward_x, forward_y);
        updateBackward(dt, backward_x, backward_y);
    }
    updateForward(dt, forward_scalar_x, forward_scalar_y);
    updateBackward(dt, backward_scalar_x, backward_scalar_y);

    if (!advect_levelset)
    {
        advectVelocity();
        if (do_BFECC_EC) correctVelocity();
    }
    /// advect Rho, T
    advectScalars();
    if (!advect_levelset && do_BFECC_EC) correctScalars();
    if (do_clear_boundaries)
        clearBoundaries();

    Array2f u_save;
    Array2f v_save;
    Array2f rho_save;
    Array2f T_save;

    u_save = u;
    v_save = v;
    rho_save = rho;
    T_save = temperature;

    applyBuoyancyForce(v, dt);

    du_temp = u; du_temp -= u_save;
    dv_temp = v; dv_temp -= v_save;
    u_save = u;
    v_save = v;

    if (!advect_levelset) projection(1e-6, use_neumann_boundary);
    float d_vel = estimateDistortion(backward_x, backward_y, forward_x, forward_y);
    float d_scalar = estimateDistortion(backward_scalar_x, backward_scalar_y, forward_scalar_x, forward_scalar_y);
    float vel = maxVel();
    std::cout << "Velocity remapping condition:" << d_vel / (vel * dt) << std::endl;
    std::cout << "Scalars remapping condition:" << d_scalar / (vel * dt) << std::endl;

    int vel_reinit_iters = delayed_reinit_frequency;
    int rho_reinit_iters = delayed_reinit_frequency;
    bool vel_remapping = ((d_vel / (vel * dt))>0.2 ||(currentframe-lastremeshing) >= vel_reinit_iters);
    bool rho_remapping = ((d_scalar / (vel * dt))>0.2 ||(currentframe-rho_lastremeshing) >= rho_reinit_iters);

    if (vel_remapping)
        proj_coeff = 1.0;

    if (!advect_levelset)
    {
        // calculate the field difference
        du_proj = u; du_proj -= u_save;
        dv_proj = v; dv_proj -= v_save;
        drho_temp = rho; drho_temp -= rho_save;
        dT_temp = temperature; dT_temp -= T_save;

        /// cumulate du, dv
        accumulateVelocity(du_temp, dv_temp, 1.0, false);
        accumulateVelocity(du_proj, dv_proj, proj_coeff, false);
        accumulateScalars(drho_temp, dT_temp, false);
    }

    if (vel_remapping && !advect_levelset)
    {
        lastremeshing = currentframe;
        resampleVelBuffer(dt);
        // if (currentframe%2==0)
            accumulateVelocity(du_proj, dv_proj, proj_coeff, false);
    }
    if (rho_remapping)
    {
        rho_lastremeshing = currentframe;
        resampleRhoBuffer(dt);
    }

    u_temp = u;
    v_temp = v;
}

void CovectorSolver2D::fullAdvect( float dt, bool advect_full_dt, bool do_mid_velocity )
{
    getCFL();
    
    u_flow = u;
    v_flow = v;
    /// update Backward & Forward mapping
    if (!advect_levelset)
    {
        updateBackward(dt, backward_x, backward_y);
        updateForward(dt, forward_x, forward_y);
    }
    if (advect_full_dt)
    {
        updateBackward(dt, backward_scalar_x, backward_scalar_y);
        updateForward(dt, forward_scalar_x, forward_scalar_y);
    }

    /// advect U,V
    if (!advect_levelset)
    {
        advectVelocityCovector(dt, do_mid_velocity);
        if (do_BFECC_EC)
        {
            correctVelocityCovector(dt, do_mid_velocity);
        }
    }
    /// advect Rho, T
    if (advect_full_dt)
    {
        advectScalarsCovector();
        if (advect_levelset_covector)
            advectQuantityCovector(dt, do_mid_velocity);
        if (do_BFECC_EC)
        {
            correctScalarsCovector();
            if (advect_levelset_covector) 
                correctQuantityCovector(dt, do_mid_velocity);
        }
    }

    if (do_clear_boundaries)
        clearBoundaries();

    Array2f u_save = u;
    Array2f v_save = v;

    applyBuoyancyForce(v, dt);

    du_temp = u; du_temp -= u_save;
    dv_temp = v; dv_temp -= v_save;

    if (!advect_levelset) projection(1e-6, use_neumann_boundary);

    if (!advect_levelset && advect_full_dt)
    {
        accumulateVelocityCovector(du_temp, dv_temp, dt, do_mid_velocity);
    }
}

void CovectorSolver2D::advanceCovector(float dt, int currentframe, int delayed_reinit_frequency)
{
    std::cout << BLUE << "Covector scheme frame " << currentframe << " starts !" << RESET << std::endl;

    if (currentframe != 0 && !advect_levelset)
    {
        u = u_temp;
        v = v_temp;
    }
    frameCount++;
    resampleCount++;

    if (second_order_covector)
    {
        // update u and v by half time-step
        Array2f backward_temp_x(backward_x), backward_temp_y(backward_y), forward_temp_x(forward_x), forward_temp_y(forward_y);
        fullAdvect(dt * 0.5f, false, delayed_reinit_frequency == 1);
        backward_x = backward_temp_x; backward_y = backward_temp_y; forward_x = forward_temp_x; forward_y = forward_temp_y;
    }
    fullAdvect(dt, true, delayed_reinit_frequency == 1);

    float d_vel = estimateDistortion(backward_x, backward_y, forward_x, forward_y);
    float d_scalar = estimateDistortion(backward_scalar_x, backward_scalar_y, forward_scalar_x, forward_scalar_y);
    float vel = maxVel();
    std::cout << "Velocity remapping condition:" << d_vel / (vel * dt) << std::endl;
    std::cout << "Scalars remapping condition:" << d_scalar / (vel * dt) << std::endl;

    int vel_reinit_iters = delayed_reinit_frequency;
    int rho_reinit_iters = delayed_reinit_frequency;
    bool vel_remapping = ((d_vel / (vel * dt))>0.1 ||(currentframe-lastremeshing) >= vel_reinit_iters);
    bool rho_remapping = ((d_scalar / (vel * dt))>0.1 ||(currentframe-rho_lastremeshing) >= rho_reinit_iters);

    cout << "REMESHING NOW!" << endl;

    if (vel_remapping && !advect_levelset)
    {
        lastremeshing = currentframe;
        resampleVelBuffer(dt);
    }
    if (rho_remapping)
    {
        rho_lastremeshing = currentframe;
        resampleRhoBuffer(dt);
    }

    u_temp = u;
    v_temp = v;

}

void CovectorSolver2D::advanceSCPF(float dt, int currentframe)
{
    std::cout << BLUE << "SCPF scheme frame " << currentframe << " starts !" << RESET << std::endl;

    getCFL();
    if (currentframe != 0 && !advect_levelset)
    {
        u = u_temp;
        v = v_temp;
    }
    frameCount++;
    resampleCount++;

    /// update Backward & Forward mapping
    if (!advect_levelset)
    {
        updateBackward(dt, backward_x, backward_y);
    }
    updateBackward(dt, backward_scalar_x, backward_scalar_y);

    /// advect U,V
    if (!advect_levelset)
    {
        u_flow = u;
        v_flow = v;
        advectVelocitySCPF(dt);
    }
    /// advect Rho, T
    advectScalarsCovector();
    if (do_clear_boundaries)
        clearBoundaries();

    if (advect_levelset_covector) {
        advectQuantityCovector(dt, false);
    }
    

    Array2f u_save = u;
    Array2f v_save = v;

    applyBuoyancyForce(v, dt);

    if (!advect_levelset) projection(1e-6, use_neumann_boundary);

    cout << "REMESHING NOW!" << endl;

    lastremeshing = currentframe;
    resampleVelBuffer(dt);

    rho_lastremeshing = currentframe;
    resampleRhoBuffer(dt);

    u_temp = u;
    v_temp = v;

}

void CovectorSolver2D::advanceSemilag(float dt, int currentframe)
{
    std::cout << BLUE <<  "Semi-Lagrangian scheme frame " << currentframe << " starts !" << RESET << std::endl;
    // Semi-Lagrangian advect density
    s_temp.assign(ni, nj, 0.0f);
    semiLagAdvect(rho, s_temp, dt, ni, nj, 0.5, 0.5);
    rho.assign(ni, nj, s_temp.a.data);
    if (do_clear_boundaries)
        clearBoundaries();

    if (!advect_levelset)
    {
        // Semi-Lagrangian advect temperature
        s_temp.assign(ni, nj, 0.0f);
        semiLagAdvect(temperature, s_temp, dt, ni, nj, 0.5, 0.5);
        temperature.assign(ni, nj, s_temp.a.data);

        // Semi-Lagrangian advect velocity
        u_temp.assign(ni + 1, nj, 0.0f);
        v_temp.assign(ni, nj + 1, 0.0f);
        semiLagAdvect(u, u_temp, dt, ni + 1, nj, 0.0, 0.5);
        semiLagAdvect(v, v_temp, dt, ni, nj + 1, 0.5, 0.0);
        u.assign(ni + 1, nj, u_temp.a.data);
        v.assign(ni, nj + 1, v_temp.a.data);

        applyBuoyancyForce(v, dt);
        projection(1e-6,use_neumann_boundary);
    }
}

void CovectorSolver2D::advanceReflection(float dt, int currentframe, bool do_SemiLag)
{
    std::cout << BLUE << "Reflection scheme frame " << currentframe << " starts !" << RESET << std::endl;

    // advect rho
    Array2f rho_first;
    Array2f rho_sec;
    rho_first.assign(ni, nj, 0.0);
    if (do_SemiLag)
    {
        semiLagAdvect(rho, rho_first, dt, ni, nj, 0.5, 0.5);
    }
    else
    {
        rho_sec.assign(ni, nj, 0.0);
        solveMaccormack(rho, rho_first, rho_sec, dt, ni, nj, 0.5, 0.5);
    }
    rho = rho_first;
    if (do_clear_boundaries)
        clearBoundaries();

    if (!advect_levelset)
    {
        // advect temperature
        Array2f T_first;
        Array2f T_sec;
        T_first.assign(ni, nj, 0.0);
        if (do_SemiLag)
        {
            semiLagAdvect(temperature, T_first, dt, ni, nj, 0.5, 0.5);
        }
        else
        {
            T_sec.assign(ni, nj, 0.0);
            solveMaccormack(temperature, T_first, T_sec, dt, ni, nj, 0.5, 0.5);
        }
        temperature = T_first;

        Array2f u_save;
        Array2f v_save;
        Array2f u_save2;
        Array2f v_save2;

        if (second_order_ref)
        {
            u_save2 = u;
            v_save2 = v;
        }
        // step 1
        u_first.assign(ni+1, nj, 0.0);
        v_first.assign(ni, nj+1, 0.0);
        if (do_SemiLag)
        {
            semiLagAdvect(u, u_first, 0.5 * dt, ni + 1, nj, 0.0, 0.5);
            semiLagAdvect(v, v_first, 0.5 * dt, ni, nj + 1, 0.5, 0.0);
        }
        else
        {
            u_sec.assign(ni + 1, nj, 0.0);
            v_sec.assign(ni, nj + 1, 0.0);
            solveMaccormack(u, u_first, u_sec, 0.5*dt, ni+1, nj, 0.0, 0.5);
            solveMaccormack(v, v_first, v_sec, 0.5*dt, ni, nj+1, 0.5, 0.0);
        }

        u = u_first;
        v = v_first;

        applyBuoyancyForce(v, 0.5f*dt);
        u_save = u;
        v_save = v;
        // step 2
        projection(1e-6, use_neumann_boundary);

        // step 3
        tbb::parallel_for((int)0, (ni+1)*nj, 1, [&](int tIdx) {
            int i = tIdx%(ni+1);
            int j = tIdx / (ni+1);
            u_temp(i, j) = 2.0*u(i,j) - u_save(i,j);
        });
        tbb::parallel_for((int)0, ni*(nj+1), 1, [&](int tIdx) {
            int i = tIdx%ni;
            int j = tIdx / ni;
            v_temp(i, j) = 2.0*v(i,j) - v_save(i,j);
        });
        if (second_order_ref)
        {
            tbb::parallel_for((int)0, (ni + 1) * nj, 1, [&](int tIdx) {
                int i = tIdx % (ni + 1);
                int j = tIdx / (ni + 1);
                u(i, j) = 2.0 * u(i, j) - u_save2(i, j);
                });
            tbb::parallel_for((int)0, ni * (nj + 1), 1, [&](int tIdx) {
                int i = tIdx % ni;
                int j = tIdx / ni;
                v(i, j) = 2.0 * v(i, j) - v_save2(i, j);
                });
        }
        if (second_order_ref)
        {
            applyBuoyancyForce(v_temp, 0.5f*dt);
        }

        // step 4
        u_first.assign(ni+1, nj, 0.0);
        v_first.assign(ni, nj+1, 0.0);
        if (do_SemiLag)
        {
            semiLagAdvect(u_temp, u_first, 0.5 * dt, ni + 1, nj, 0.0, 0.5);
            semiLagAdvect(v_temp, v_first, 0.5 * dt, ni, nj + 1, 0.5, 0.0);
        }
        else
        {
            u_sec.assign(ni + 1, nj, 0.0);
            v_sec.assign(ni, nj + 1, 0.0);
            solveMaccormack(u_temp, u_first, u_sec, 0.5 * dt, ni + 1, nj, 0.0, 0.5);
            solveMaccormack(v_temp, v_first, v_sec, 0.5 * dt, ni, nj + 1, 0.5, 0.0);
        }
        u = u_first;
        v = v_first;

        if (!second_order_ref)
        {
            applyBuoyancyForce(v, 0.5f*dt);
        }
        // step 5
        projection(1e-6, use_neumann_boundary);
    }
}

float CovectorSolver2D::estimateDistortion(Array2f &back_x, Array2f &back_y, Array2f &fwd_x, Array2f &fwd_y)
{
    float d = 0;
    for(int tIdx=0;tIdx<ni*nj;tIdx++)
    {
        int i = tIdx % ni;
        int j = tIdx / ni;
        if(i>2&&i<ni-3 && j>2&&j<nj-3) {
            Vec2f init_pos = h*(Vec2f(i,j) + Vec2f(0.5, 0.5));
            Vec2f fpos0 = Vec2f(fwd_x(i, j), fwd_y(i, j));
            Vec2f bpos = Vec2f(sampleField(fpos0 - Vec2f(0.5) * h, back_x),
                               sampleField(fpos0 - Vec2f(0.5) * h, back_y));
            float new_dist = dist(bpos, init_pos);
            if ( new_dist > d) d = new_dist;
        }
    }

    for(int tIdx=0;tIdx<ni*nj;tIdx++)
    {
        int i = tIdx % ni;
        int j = tIdx / ni;
        if(i>2&&i<ni-3 && j>2&&j<nj-3) {
            Vec2f init_pos = h*(Vec2f(i,j) + Vec2f(0.5, 0.5));
            Vec2f bpos0 = Vec2f(back_x(i, j), back_y(i, j));
            Vec2f fpos = Vec2f(sampleField(bpos0 - Vec2f(0.5) * h, fwd_x),
                               sampleField(bpos0 - Vec2f(0.5) * h, fwd_y));
            float new_dist = dist(fpos, init_pos);
            if (new_dist > d) d = new_dist;
        }
    }
    return d;
}

float CovectorSolver2D::maxVel()
{
    float vel=0;
    int idx_i = 0;
    int idx_j = 0;
    for(int i=0;i<u.a.n;i++)
    {
        if (u.a[i] > vel)
        {
            vel = u.a[i];
            idx_i = i / (ni+1);
            idx_j = i % (ni+1);
        }
    }
    for(int j=0;j<v.a.n;j++)
    {
        if (v.a[j] > vel)
        {
            vel = v.a[j];
            idx_i = j / ni;
            idx_j = j % nj;
        }
    }
    return vel + 1e-5;
}

void CovectorSolver2D::correctQuantityCovector(float dt, bool do_mid_velocity)
{
    Array2f u_curr = cv_u;
    Array2f v_curr = cv_v;
    Array2f u_error, v_error;
    u_error.resize(ni + 1, nj, 0.0);
    v_error.resize(ni, nj + 1, 0.0);
    tbb::parallel_for((int)0, (ni + 1) * nj, 1, [&](int tIdx) {
        int i = tIdx % (ni + 1);
        int j = tIdx / (ni + 1);
        if (i != 0 && i != ni)
        {
            for (int k = start_idx; k < end_idx; k++) {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.0, 0.5) + h * dir[k];

                u_error(i, j) += w[k] * calcDPsiTRow(pos, Vec2f(0.5,0.0), cv_u, Vec2f(0.0,0.5), cv_v, Vec2f(0.5,0.0), forward_scalar_x, forward_scalar_y, dt, do_mid_velocity);
            }
        }
        });
    tbb::parallel_for((int)0, ni * (nj + 1), 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        if (j != 0 && j != nj)
        {
            for (int k = start_idx; k < end_idx; k++) {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.0) + h * dir[k];

                v_error(i, j) += w[k] * calcDPsiTRow(pos, Vec2f(0.0,0.5), cv_u, Vec2f(0.0,0.5), cv_v, Vec2f(0.5,0.0), forward_scalar_x, forward_scalar_y, dt, do_mid_velocity);
            }
        }
        });
    u_error -= cv_u_init;
    u_error *= 0.5f;
    v_error -= cv_v_init;
    v_error *= 0.5f;
    tbb::parallel_for((int)0, (ni + 1)* nj, 1, [&](int tIdx) {
        int i = tIdx % (ni + 1);
        int j = tIdx / (ni + 1);
        if (i != 0 && i != ni)
        {
            for (int k = start_idx; k < end_idx; k++) {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.0, 0.5) + h * dir[k];

                cv_u(i, j) -= w[k] * calcDPsiTRow(pos, Vec2f(0.5,0.0), u_error, Vec2f(0.0,0.5), v_error, Vec2f(0.5,0.0), backward_scalar_x, backward_scalar_y, -dt, do_mid_velocity);
            }
        }
        });
    clampExtrema2(ni+1, nj, u_curr, cv_u);
    tbb::parallel_for((int)0, ni * (nj + 1), 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        if (j != 0 && j != nj)
        {
            for (int k = start_idx; k < end_idx; k++) {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.0) + h * dir[k];

                cv_v(i, j) -= w[k] * calcDPsiTRow(pos, Vec2f(0.0,0.5), u_error, Vec2f(0.0,0.5), v_error, Vec2f(0.5,0.0), backward_scalar_x, backward_scalar_y, -dt, do_mid_velocity);
            }
        }
        });
    clampExtrema2(ni, nj+1, v_curr, cv_v);
}

void CovectorSolver2D::correctScalarsCovector()
{
    Array2f rho_curr = rho;
    Array2f T_curr = temperature;
    Array2f rho_error;
    Array2f T_error;
    rho_error.resize(ni, nj, 0.0);
    T_error.resize(ni, nj, 0.0);
    tbb::parallel_for((int)0, ni * nj, 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        for (int k = start_idx; k < end_idx; k++)
        {
            Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.5) + h * dir[k];
            float x_init = sampleField(pos - h * Vec2f(0.5), forward_scalar_x);
            float y_init = sampleField(pos - h * Vec2f(0.5), forward_scalar_y);
            Vec2f samplePos = Vec2f(x_init, y_init);
            rho_error(i, j) += w[k] * sampleField(samplePos - h * Vec2f(0.5, 0.5), rho);
        }
        });
    rho_error -= rho_init;
    rho_error *= 0.5f;
    tbb::parallel_for((int)0, ni * nj, 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        for (int k = start_idx; k < end_idx; k++)
        {
            Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.5) + h * dir[k];
            float x_init = sampleField(pos - h * Vec2f(0.5), backward_scalar_x);
            float y_init = sampleField(pos - h * Vec2f(0.5), backward_scalar_y);
            Vec2f samplePos = Vec2f(x_init, y_init);
            rho(i, j) -= w[k] * sampleField(samplePos - h * Vec2f(0.5, 0.5), rho_error);
        }
        });
    clampExtrema2(ni, nj,rho_curr,rho);
    /// T
    tbb::parallel_for((int)0, ni * nj, 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        for (int k = start_idx; k < end_idx; k++)
        {
            Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.5) + h * dir[k];
            float x_init = sampleField(pos - h * Vec2f(0.5), forward_scalar_x);
            float y_init = sampleField(pos - h * Vec2f(0.5), forward_scalar_y);
            Vec2f samplePos = Vec2f(x_init, y_init);
            T_error(i, j) += w[k] * sampleField(samplePos - h * Vec2f(0.5, 0.5), temperature);
        }
        });
    T_error -= T_init;
    T_error *= 0.5f;
    tbb::parallel_for((int)0, ni * nj, 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        for (int k = start_idx; k < end_idx; k++)
        {
            Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.5) + h * dir[k];
            float x_init = sampleField(pos - h * Vec2f(0.5), backward_scalar_x);
            float y_init = sampleField(pos - h * Vec2f(0.5), backward_scalar_y);
            Vec2f samplePos = Vec2f(x_init, y_init);
            temperature(i, j) -= w[k] * sampleField(samplePos - h * Vec2f(0.5, 0.5), T_error);
        }
        });
    clampExtrema2(ni,nj,T_curr,temperature);
}

void CovectorSolver2D::correctScalars()
{
    Array2f rho_curr = rho;
    Array2f T_curr = temperature;
    Array2f temp_rho;
    Array2f temp_T;
    temp_rho.resize(ni,nj,0.0);
    temp_T.resize(ni,nj,0.0);
    tbb::parallel_for((int) 0, ni * nj, 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        for(int k = start_idx; k < end_idx; k++)
        {
            Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.5) + h * dir[k];
            float x_init = sampleField(pos - h * Vec2f(0.5), forward_scalar_x);
            float y_init = sampleField(pos - h * Vec2f(0.5), forward_scalar_y);
            Vec2f samplePos = Vec2f(x_init, y_init);
            temp_rho(i, j) += w[k]*(sampleField(samplePos - h * Vec2f(0.5, 0.5), rho) - drho(i,j));
        }
    });
    temp_rho -= rho_init;
    temp_rho *= 0.5f;
    tbb::parallel_for((int) 0, ni * nj, 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        for(int k = start_idx; k < end_idx; k++)
        {
            Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.5) + h * dir[k];
            float x_init = sampleField(pos - h * Vec2f(0.5), backward_scalar_x);
            float y_init = sampleField(pos - h * Vec2f(0.5), backward_scalar_y);
            Vec2f samplePos = Vec2f(x_init, y_init);
            rho(i, j) -= w[k]*sampleField(samplePos - h * Vec2f(0.5, 0.5), temp_rho);
        }
    });
    clampExtrema2(ni, nj,rho_curr,rho);
    /// T
    tbb::parallel_for((int) 0, ni * nj, 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        for(int k = start_idx; k < end_idx; k++)
        {
            Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.5) + h * dir[k];
            float x_init = sampleField(pos - h * Vec2f(0.5), forward_scalar_x);
            float y_init = sampleField(pos - h * Vec2f(0.5), forward_scalar_y);
            Vec2f samplePos = Vec2f(x_init, y_init);
            temp_T(i, j) += w[k]*(sampleField(samplePos - h * Vec2f(0.5, 0.5), temperature) - dT(i,j));
        }
    });
    temp_T -= T_init;
    temp_T *= 0.5f;
    tbb::parallel_for((int) 0, ni * nj, 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        for(int k = start_idx; k < end_idx; k++)
        {
            Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.5) + h * dir[k];
            float x_init = sampleField(pos - h * Vec2f(0.5), backward_scalar_x);
            float y_init = sampleField(pos - h * Vec2f(0.5), backward_scalar_y);
            Vec2f samplePos = Vec2f(x_init, y_init);
            temperature(i, j) -= w[k]*sampleField(samplePos - h * Vec2f(0.5, 0.5), temp_T);
        }
    });
    clampExtrema2(ni,nj,T_curr,temperature);
}

void CovectorSolver2D::correctVelocityCovector(float dt, bool do_mid_velocity)
{
    Array2f u_curr = u;
    Array2f v_curr = v;
    Array2f u_error, v_error;
    u_error.resize(ni + 1, nj, 0.0);
    v_error.resize(ni, nj + 1, 0.0);
    tbb::parallel_for((int)0, (ni + 1) * nj, 1, [&](int tIdx) {
        int i = tIdx % (ni + 1);
        int j = tIdx / (ni + 1);
        if (i != 0 && i != ni)
        {
            for (int k = start_idx; k < end_idx; k++) {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.0, 0.5) + h * dir[k];

                u_error(i, j) += w[k] * (calcDPsiTRow(pos, Vec2f(0.5,0.0), u, Vec2f(0.0,0.5), v, Vec2f(0.5,0.0), forward_x, forward_y, dt, do_mid_velocity) - du(i,j));
            }
        }
        });
    tbb::parallel_for((int)0, ni * (nj + 1), 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        if (j != 0 && j != nj)
        {
            for (int k = start_idx; k < end_idx; k++) {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.0) + h * dir[k];

                v_error(i, j) += w[k] * (calcDPsiTRow(pos, Vec2f(0.0,0.5), u, Vec2f(0.0,0.5), v, Vec2f(0.5,0.0), forward_x, forward_y, dt, do_mid_velocity) - dv(i,j));
            }
        }
        });
    u_error -= u_init;
    u_error *= 0.5f;
    v_error -= v_init;
    v_error *= 0.5f;
    tbb::parallel_for((int)0, (ni + 1)* nj, 1, [&](int tIdx) {
        int i = tIdx % (ni + 1);
        int j = tIdx / (ni + 1);
        if (i != 0 && i != ni)
        {
            for (int k = start_idx; k < end_idx; k++) {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.0, 0.5) + h * dir[k];

                u(i, j) -= w[k] * calcDPsiTRow(pos, Vec2f(0.5,0.0), u_error, Vec2f(0.0,0.5), v_error, Vec2f(0.5,0.0), backward_x, backward_y, -dt, do_mid_velocity);
            }
        }
        });
    clampExtrema2(ni+1, nj, u_curr, u);
    tbb::parallel_for((int)0, ni * (nj + 1), 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        if (j != 0 && j != nj)
        {
            for (int k = start_idx; k < end_idx; k++) {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.0) + h * dir[k];

                v(i, j) -= w[k] * calcDPsiTRow(pos, Vec2f(0.0,0.5), u_error, Vec2f(0.0,0.5), v_error, Vec2f(0.5,0.0), backward_x, backward_y, -dt, do_mid_velocity);
            }
        }
        });
    clampExtrema2(ni, nj+1, v_curr, v);
}

void CovectorSolver2D::correctVelocity()
{
    Array2f u_curr = u;
    Array2f v_curr = v;
    Array2f temp_u, temp_v;
    temp_u.resize(ni+1, nj, 0.0);
    temp_v.resize(ni, nj+1, 0.0);
    tbb::parallel_for((int) 0, (ni + 1) * nj, 1, [&](int tIdx) {
        int i = tIdx % (ni + 1);
        int j = tIdx / (ni + 1);
        if (i != 0 && i != ni)
        {
            for (int k = start_idx; k < end_idx; k++) {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.0, 0.5) + h * dir[k];
                float x_init = sampleField(pos - h * Vec2f(0.5), forward_x);
                float y_init = sampleField(pos - h * Vec2f(0.5), forward_y);
                Vec2f pos1 = Vec2f(x_init,y_init);
                temp_u(i, j) +=  w[k] * (sampleField(pos1 - h * Vec2f(0.0, 0.5), u) - du(i,j));
            }
        }
    });
    temp_u -= u_init;
    temp_u *= 0.5f;
    tbb::parallel_for((int) 0, (ni + 1) * nj, 1, [&](int tIdx) {
        int i = tIdx % (ni + 1);
        int j = tIdx / (ni + 1);
        if (i != 0 && i != ni)
        {
            for (int k = start_idx; k < end_idx; k++) {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.0, 0.5) + h * dir[k];
                float x_init = sampleField(pos - h * Vec2f(0.5), backward_x);
                float y_init = sampleField(pos - h * Vec2f(0.5), backward_y);
                Vec2f pos1 = Vec2f(x_init,y_init);
                u(i, j) -=  w[k] * sampleField(pos1 - h * Vec2f(0.0, 0.5), temp_u);
            }
        }
    });
    clampExtrema2(ni+1, nj, u_curr, u);
    tbb::parallel_for((int) 0, ni * (nj + 1), 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        if (j != 0 && j != nj)
        {
            for (int k = start_idx; k < end_idx; k++) {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.0) + h * dir[k];
                float x_init = sampleField(pos - h * Vec2f(0.5), forward_x);
                float y_init = sampleField(pos - h * Vec2f(0.5), forward_y);
                Vec2f pos1 = Vec2f(x_init,y_init);
                temp_v(i, j) += w[k] * (sampleField(pos1 - h * Vec2f(0.5, 0.0), v) - dv(i,j));
            }
        }
    });
    temp_v -= v_init;
    temp_v *= 0.5f;
    tbb::parallel_for((int) 0, ni * (nj + 1), 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        if (j != 0 && j != nj)
        {
            for (int k = start_idx; k < end_idx; k++) {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.0) + h * dir[k];
                float x_init = sampleField(pos - h * Vec2f(0.5), backward_x);
                float y_init = sampleField(pos - h * Vec2f(0.5), backward_y);
                Vec2f pos1 = Vec2f(x_init,y_init);
                v(i, j) -=  w[k] * (sampleField(pos1 - h * Vec2f(0.5, 0.0), temp_v));
            }
        }
    });
    clampExtrema2(ni, nj+1, v_curr, v);
}

void CovectorSolver2D::advectVelocitySCPF(float dt)
{
    /// u
    tbb::parallel_for((int)0, (ni + 1) * nj, 1, [&](int tIdx) {
        int i = tIdx % (ni + 1);
        int j = tIdx / (ni + 1);
        u(i, j) = 0;
        if (i != 0 && i != ni)
        {
            for (int k = start_idx; k < end_idx; k++) {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.0, 0.5) + h * dir[k];
                u(i, j) += w[k] * calcDPsiTRow(pos, Vec2f(0.5,0.0), u_init, Vec2f(0.0,0.5), v_init, Vec2f(0.5,0.0), backward_x, backward_y, -dt, false, true);
            }
        }
    });
    /// v
    tbb::parallel_for((int)0, ni * (nj + 1), 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        v(i, j) = 0.0;
        if (j != 0 && j != nj)
        {
            for (int k = start_idx; k < end_idx; k++)
            {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.0) + h * dir[k];
                v(i, j) += w[k] * calcDPsiTRow(pos, Vec2f(0.0,0.5), u_init, Vec2f(0.0,0.5), v_init, Vec2f(0.5,0.0), backward_x, backward_y, -dt, false, true);
            }
        }
    });
}

float CovectorSolver2D::calcDPsiTRow(const Vec2f& pos, const Vec2f& offset, 
                                   const Array2f& u_sample, const Vec2f& u_offset, 
                                   const Array2f& v_sample, const Vec2f& v_offset, 
                                   const Array2f& map_x, const Array2f& map_y, float dt, bool do_mid_velocity, bool do_trapezoidal_rule)
{

    Vec2f pos1;
    if (do_mid_velocity)
    {
        pos1 = solveODE(dt, pos, u_flow, v_flow);
    }
    else
    {
        float x_init = sampleField(pos - h * Vec2f(0.5), map_x);
        float y_init = sampleField(pos - h * Vec2f(0.5), map_y);
        pos1 = Vec2f(x_init, y_init);
    }
    Vec2f pos_front = pos + h * offset;
    float x_front_init = sampleField(pos_front - h * Vec2f(0.5), map_x);
    float y_front_init = sampleField(pos_front - h * Vec2f(0.5), map_y);
    Vec2f pos1_front = Vec2f(x_front_init, y_front_init);
    Vec2f pos_back  = pos - h * offset;
    float x_back_init = sampleField(pos_back - h * Vec2f(0.5), map_x);
    float y_back_init = sampleField(pos_back - h * Vec2f(0.5), map_y);
    Vec2f pos1_back = Vec2f(x_back_init, y_back_init);
    Vec2f diff = -pos1_back+pos1_front;
    float distance = dist(pos_back,pos_front);

    if (do_trapezoidal_rule)
    {
        Vec2f vel_back(sampleField(pos1_back - h * u_offset, u_sample),sampleField(pos1_back - h * v_offset, v_sample));
        Vec2f vel_front(sampleField(pos1_front - h * u_offset, u_sample),sampleField(pos1_front - h * v_offset, v_sample)); 
        Vec2f avg_vel = (vel_back + vel_front)/2.f;
        return dot(diff,avg_vel) / distance;
    }
    else
    {
        Vec2f vel(sampleField(pos1 - h * u_offset, u_sample),sampleField(pos1 - h * v_offset, v_sample));
        return dot(diff,vel) / distance;
    }
}

void CovectorSolver2D::advectVelocityCovector(float dt, bool do_mid_velocity)
{
    Array2f u_sample(u_init);
    Array2f v_sample(v_init);
    tbb::parallel_for((int)0, (ni + 1) * nj, 1, [&](int tIdx) {
        int i = tIdx % (ni + 1);
        int j = tIdx / (ni + 1);
        u_sample(i,j) += du(i,j);
    });
    tbb::parallel_for((int)0, ni * (nj+1), 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        v_sample(i,j) += dv(i,j);
    });
    /// u
    tbb::parallel_for((int)0, (ni + 1) * nj, 1, [&](int tIdx) {
        int i = tIdx % (ni + 1);
        int j = tIdx / (ni + 1);
        u(i, j) = 0;
        if (i != 0 && i != ni)
        {
            for (int k = start_idx; k < end_idx; k++) {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.0, 0.5) + h * dir[k];

                u(i, j) += w[k] * calcDPsiTRow(pos, Vec2f(0.5,0.0), u_sample, Vec2f(0.0,0.5), v_sample, Vec2f(0.5,0.0), backward_x, backward_y, -dt, do_mid_velocity);
            }
        }
        });
    /// v
    tbb::parallel_for((int)0, ni * (nj + 1), 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        v(i, j) = 0.0;
        if (j != 0 && j != nj)
        {
            for (int k = start_idx; k < end_idx; k++)
            {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.0) + h * dir[k];
                v(i, j) += w[k] * calcDPsiTRow(pos, Vec2f(0.0,0.5), u_sample, Vec2f(0.0,0.5), v_sample, Vec2f(0.5, 0.0), backward_x, backward_y, -dt, do_mid_velocity);
            }
        }
        });
}

void CovectorSolver2D::advectVelocity()
{
    /// u
    tbb::parallel_for((int) 0, (ni + 1) * nj, 1, [&](int tIdx) {
        int i = tIdx % (ni + 1);
        int j = tIdx / (ni + 1);
        u(i,j) = 0;
        if (i != 0 && i != ni)
        {
            for (int k = start_idx; k < end_idx; k++) {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.0, 0.5) + h * dir[k];
                float x_init = sampleField(pos - h * Vec2f(0.5), backward_x);
                float y_init = sampleField(pos - h * Vec2f(0.5), backward_y);
                Vec2f pos1 = Vec2f(x_init, y_init);
                float x_origin = sampleField(pos1 - h * Vec2f(0.5), backward_xprev);
                float y_origin = sampleField(pos1 - h * Vec2f(0.5), backward_yprev);
                Vec2f pos2 = Vec2f(x_origin, y_origin);
                u(i, j) += (1.f - blend_coeff) * w[k] * (sampleField(pos2 - h * Vec2f(0.0, 0.5), u_origin) +
                                            sampleField(pos1 - h * Vec2f(0.0, 0.5), du) +
                                            sampleField(pos2 - h* Vec2f(0,0.5), du_prev)
                );
                u(i, j) += blend_coeff * w[k] * (sampleField(pos1 - h * Vec2f(0.0, 0.5), u_init) +
                                            sampleField(pos1 - h * Vec2f(0.0, 0.5), du));
            }
        }
    });
    /// v
    tbb::parallel_for((int) 0, ni * (nj + 1), 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        v(i, j) = 0.0;
        if (j != 0 && j != nj)
        {
            for (int k = start_idx; k < end_idx; k++)
            {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.0) + h * dir[k];
                float x_init = sampleField(pos - h * Vec2f(0.5), backward_x);
                float y_init = sampleField(pos - h * Vec2f(0.5), backward_y);
                Vec2f pos1 = Vec2f(x_init, y_init);
                float x_origin = sampleField(pos1 - h * Vec2f(0.5), backward_xprev);
                float y_origin = sampleField(pos1 - h * Vec2f(0.5), backward_yprev);
                Vec2f pos2 = Vec2f(x_origin, y_origin);
                v(i, j) += (1.f - blend_coeff) * w[k] * (sampleField(pos2 - h * Vec2f(0.5, 0.0), v_origin) +
                                            sampleField(pos1 - h * Vec2f(0.5, 0.0), dv) +
                                            sampleField(pos2 - h * Vec2f(0.5,0.0), dv_prev));
                v(i, j) += blend_coeff * w[k] * (sampleField(pos1 - h * Vec2f(0.5, 0.0), v_init) +
                                            sampleField(pos1 - h * Vec2f(0.5, 0.0), dv));
            }
        }
    });
}

void CovectorSolver2D::advectQuantityCovector(float dt, bool do_mid_velocity)
{
    tbb::parallel_for((int)0, (ni + 1) * nj, 1, [&](int tIdx) {
        int i = tIdx % (ni + 1);
        int j = tIdx / (ni + 1);
        cv_u(i, j) = 0;
        if (i != 0 && i != ni)
        {
            for (int k = start_idx; k < end_idx; k++) {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.0, 0.5) + h * dir[k];
                cv_u(i, j) += w[k] * calcDPsiTRow(pos, Vec2f(0.5,0.0), cv_u_init, Vec2f(0.0,0.5), cv_v_init, Vec2f(0.5,0.0), backward_scalar_x, backward_scalar_y, -dt, do_mid_velocity);
            }
        }
        });
    /// v
    tbb::parallel_for((int)0, ni * (nj + 1), 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        cv_v(i, j) = 0.0;
        if (j != 0 && j != nj)
        {
            for (int k = start_idx; k < end_idx; k++)
            {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.0) + h * dir[k];
                cv_v(i, j) += w[k] * calcDPsiTRow(pos, Vec2f(0.0,0.5), cv_u_init, Vec2f(0.0,0.5), cv_v_init, Vec2f(0.5, 0.0), backward_scalar_x, backward_scalar_y, -dt, do_mid_velocity);
            }
        }
        });
}

void CovectorSolver2D::advectScalarsCovector()
{
    tbb::parallel_for((int)0, ni * nj, 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        rho(i, j) = 0.f;
        for (int k = start_idx; k < end_idx; k++) {
            Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.5) + h * dir[k];
            float x_init = sampleField(pos - h * Vec2f(0.5), backward_scalar_x);
            float y_init = sampleField(pos - h * Vec2f(0.5), backward_scalar_y);
            Vec2f pos1 = Vec2f(x_init, y_init);
            float x_origin = sampleField(pos1 - h * Vec2f(0.5), backward_scalar_xprev);
            float y_origin = sampleField(pos1 - h * Vec2f(0.5), backward_scalar_yprev);
            Vec2f pos2 = Vec2f(x_origin, y_origin);
            rho(i, j) += (1.f - blend_coeff) * w[k] * sampleField(pos2 - h * Vec2f(0.5, 0.5), rho_orig);
            rho(i, j) += blend_coeff * w[k] * sampleField(pos1 - h * Vec2f(0.5, 0.5), rho_init);
        }
    });
    tbb::parallel_for((int)0, ni * nj, 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        temperature(i, j) = 0.f;
        for (int k = start_idx; k < end_idx; k++) {
            Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.5) + h * dir[k];
            float x_init = sampleField(pos - h * Vec2f(0.5), backward_scalar_x);
            float y_init = sampleField(pos - h * Vec2f(0.5), backward_scalar_y);
            Vec2f pos1 = Vec2f(x_init, y_init);
            float x_origin = sampleField(pos1 - h * Vec2f(0.5), backward_scalar_xprev);
            float y_origin = sampleField(pos1 - h * Vec2f(0.5), backward_scalar_yprev);
            Vec2f pos2 = Vec2f(x_origin, y_origin);
            temperature(i, j) += (1.f - blend_coeff) * w[k] * sampleField(pos2 - h * Vec2f(0.5, 0.5), T_orig);
            temperature(i, j) += blend_coeff * w[k] * sampleField(pos1 - h * Vec2f(0.5, 0.5), T_init);
        }
    });
}

void CovectorSolver2D::advectScalars()
{
    tbb::parallel_for((int) 0, ni * nj, 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        rho(i,j) = 0.f;
        for (int k = start_idx; k < end_idx; k++) {
            Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.5) + h * dir[k];
            float x_init = sampleField(pos - h * Vec2f(0.5), backward_scalar_x);
            float y_init = sampleField(pos - h * Vec2f(0.5), backward_scalar_y);
            Vec2f pos1 = Vec2f(x_init,y_init);
            float x_origin = sampleField(pos1 - h*Vec2f(0.5), backward_scalar_xprev);
            float y_origin = sampleField(pos1 - h*Vec2f(0.5), backward_scalar_yprev);
            Vec2f pos2 = Vec2f(x_origin,y_origin);
            rho(i, j) += (1.f - blend_coeff) * w[k] * (sampleField(pos2 - h * Vec2f(0.5, 0.5), rho_orig) +
                                            sampleField(pos1 - h * Vec2f(0.5, 0.5), drho) +
                                            sampleField(pos2 - h * Vec2f(0.5, 0.5), drho_prev));
            rho(i, j) += blend_coeff * w[k] * (sampleField(pos1 - h * Vec2f(0.5, 0.5), rho_init) +
                                            sampleField(pos1 - h * Vec2f(0.5, 0.5), drho));
        }
    });
    tbb::parallel_for((int) 0, ni * nj, 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        temperature(i,j) = 0.f;
        for (int k = start_idx; k < end_idx; k++) {
            Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.5) + h * dir[k];
            float x_init = sampleField(pos - h * Vec2f(0.5), backward_scalar_x);
            float y_init = sampleField(pos - h * Vec2f(0.5), backward_scalar_y);
            Vec2f pos1 = Vec2f(x_init, y_init);
            float x_origin = sampleField(pos1 - h * Vec2f(0.5), backward_scalar_xprev);
            float y_origin = sampleField(pos1 - h * Vec2f(0.5), backward_scalar_yprev);
            Vec2f pos2 = Vec2f(x_origin, y_origin);
            temperature(i, j) += (1.f - blend_coeff) * w[k] * (sampleField(pos2 - h * Vec2f(0.5, 0.5), T_orig) +
                                                    sampleField(pos1 - h * Vec2f(0.5, 0.5), dT) +
                                                    sampleField(pos2 - h * Vec2f(0.5, 0.5), dT_prev));
            temperature(i, j) += blend_coeff * w[k] * (sampleField(pos1 - h * Vec2f(0.5, 0.5), T_init) +
                                            sampleField(pos1 - h * Vec2f(0.5, 0.5), dT));
        }
    });
}

void CovectorSolver2D::accumulateVelocityCovector(Array2f &u_change, Array2f &v_change, float dt, bool do_mid_velocity)
{
    // sample du, dv
    tbb::parallel_for((int) 0, (ni + 1) * nj, 1, [&](int tIdx) {
        int i = tIdx % (ni + 1);
        int j = tIdx / (ni + 1);
        for(int k = start_idx; k < end_idx; k++)
        {
            Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.0, 0.5) + h * dir[k];
            du(i, j) += w[k] * calcDPsiTRow(pos, Vec2f(0.5,0.0), u_change, Vec2f(0.0,0.5), v_change, Vec2f(0.5,0.0), forward_x, forward_y, dt, do_mid_velocity);
        }
    });
    tbb::parallel_for((int) 0, ni * (nj + 1), 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        for (int k = start_idx; k < end_idx; k++)
        {
            Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.0) + h * dir[k];
            dv(i, j) += w[k] * calcDPsiTRow(pos, Vec2f(0.0,0.5), u_change, Vec2f(0.0,0.5), v_change, Vec2f(0.5,0.0), forward_x, forward_y, dt, do_mid_velocity);
        }
    });
}

void CovectorSolver2D::accumulateVelocity(Array2f &u_change, Array2f &v_change, float proj_coeff, bool error_correction)
{
    Array2f test_du, test_du_star;
    Array2f test_dv, test_dv_star;
    test_du.resize(ni+1, nj, 0.0);
    test_du_star.resize(ni+1, nj, 0.0);
    test_dv.resize(ni, nj+1, 0.0);
    test_dv_star.resize(ni, nj+1, 0.0);
    // step 7
    // sample du, dv, drho, dT
    tbb::parallel_for((int) 0, (ni + 1) * nj, 1, [&](int tIdx) {
        int i = tIdx % (ni + 1);
        int j = tIdx / (ni + 1);
        for(int k = start_idx; k < end_idx; k++)
        {
            Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.0, 0.5) + h * dir[k];
            float x_init = sampleField(pos - h * Vec2f(0.5), forward_x);
            float y_init = sampleField(pos - h * Vec2f(0.5), forward_y);
            Vec2f samplePos = Vec2f(x_init, y_init);
            //clampPos(samplePos);
            test_du(i, j) += w[k]*sampleField(samplePos - h * Vec2f(0.0, 0.5), u_change);
        }
    });
    if (error_correction)
    {
        tbb::parallel_for((int) 0, (ni + 1) * nj, 1, [&](int tIdx) {
            int i = tIdx % (ni + 1);
            int j = tIdx / (ni + 1);
            for(int k = start_idx; k < end_idx; k++)
            {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.0, 0.5) + h * dir[k];
                float x_init = sampleField(pos - h * Vec2f(0.5), backward_x);
                float y_init = sampleField(pos - h * Vec2f(0.5), backward_y);
                Vec2f samplePos = Vec2f(x_init, y_init);
                test_du_star(i, j) += w[k]*sampleField(samplePos - h * Vec2f(0.0, 0.5), test_du);
            }
        });
        test_du_star -= u_change;
        test_du_star *= 0.5;
    }
    tbb::parallel_for((int) 0, (ni + 1) * nj, 1, [&](int tIdx) {
        int i = tIdx % (ni + 1);
        int j = tIdx / (ni + 1);
        for(int k = start_idx; k < end_idx; k++)
        {
            Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.0, 0.5) + h * dir[k];
            float x_init = sampleField(pos - h * Vec2f(0.5), forward_x);
            float y_init = sampleField(pos - h * Vec2f(0.5), forward_y);
            Vec2f samplePos = Vec2f(x_init, y_init);
            if (error_correction){
                du(i,j) += w[k]* proj_coeff *(sampleField(samplePos - h * Vec2f(0.0, 0.5), u_change) -
                                    sampleField(samplePos - h * Vec2f(0.0, 0.5), test_du_star));
            }
            else{
                du(i,j) += w[k]* proj_coeff *sampleField(samplePos - h * Vec2f(0.0, 0.5), u_change);
            }
        }
    });
    tbb::parallel_for((int) 0, ni * (nj + 1), 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        for (int k = start_idx; k < end_idx; k++)
        {
            Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.0) + h * dir[k];
            float x_init = sampleField(pos - h * Vec2f(0.5), forward_x);
            float y_init = sampleField(pos - h * Vec2f(0.5), forward_y);
            Vec2f samplePos = Vec2f(x_init, y_init);
            test_dv(i, j) += w[k] * sampleField(samplePos - h * Vec2f(0.5, 0.0), v_change);
        }
    });
    if (error_correction)
    {
        tbb::parallel_for((int) 0, ni * (nj + 1), 1, [&](int tIdx) {
            int i = tIdx % ni;
            int j = tIdx / ni;
            for (int k = start_idx; k < end_idx; k++)
            {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.0) + h * dir[k];
                float x_init = sampleField(pos - h * Vec2f(0.5), backward_x);
                float y_init = sampleField(pos - h * Vec2f(0.5), backward_y);
                Vec2f samplePos = Vec2f(x_init, y_init);
                test_dv_star(i, j) += w[k] * sampleField(samplePos - h * Vec2f(0.5, 0.0), test_dv);
            }
        });
        test_dv_star -= v_change;
        test_dv_star *= 0.5;
    }
    tbb::parallel_for((int) 0, ni * (nj + 1), 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        for (int k = start_idx; k < end_idx; k++)
        {
            Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.0) + h * dir[k];
            float x_init = sampleField(pos - h * Vec2f(0.5), forward_x);
            float y_init = sampleField(pos - h * Vec2f(0.5), forward_y);
            Vec2f samplePos = Vec2f(x_init, y_init);
            if (error_correction){
                dv(i,j) += w[k] * proj_coeff * (sampleField(samplePos - h * Vec2f(0.5, 0.0), v_change) -
                                        sampleField(samplePos - h * Vec2f(0.5, 0.0), test_dv_star));
            }
            else{
                dv(i,j) += w[k] * proj_coeff * (sampleField(samplePos - h * Vec2f(0.5, 0.0), v_change));
            }
        }
    });
}

void CovectorSolver2D::updateForward(float dt, Array2f &fwd_x, Array2f &fwd_y)
{
    // forward mapping
    tbb::parallel_for((int) 0, ni * nj, 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        Vec2f pos = Vec2f(fwd_x(i, j), fwd_y(i, j));
        Vec2f posNew = solveODE(dt, pos, u, v);
        fwd_x(i, j) = posNew[0];
        fwd_y(i, j) = posNew[1];
    });
}

void CovectorSolver2D::updateBackward(float dt, Array2f &back_x, Array2f &back_y)
{
    // backward mapping
    float substep = _cfl;
    float T = dt;
    float t = 0;
    while(t < T)
    {
        if (t + substep > T) substep = T - t;
        map_tempx.assign(ni, nj, 0.0f);
        map_tempy.assign(ni, nj, 0.0f);
        if (RK4_or_DMC == 0) { //SF
            semiLagAdvect(back_x, map_tempx, substep, ni, nj, 0.5, 0.5);
            semiLagAdvect(back_y, map_tempy, substep, ni, nj, 0.5, 0.5);
        }
        else // RK4_or_DMC == 1 //DMC
        {
            semiLagAdvectDMC(back_x, map_tempx, substep, ni, nj, 0.5, 0.5);
            semiLagAdvectDMC(back_y, map_tempy, substep, ni, nj, 0.5, 0.5);
        }
        back_x = map_tempx;
        back_y = map_tempy;
        t += substep;
    }
}

void CovectorSolver2D::clampExtrema2(int _ni, int _nj, Array2f &before, Array2f &after)
{
    tbb::parallel_for((int) 0, _ni * _nj, 1, [&](int tIdx) {
        int i = tIdx % _ni;
        int j = tIdx / _ni;
        float min_v=1e+6f, max_v=0.f;

        for(int jj=j-1;jj<=j+1;jj++)
        {
            for(int ii=i-1;ii<=i+1;ii++)
            {
                if (ii >= 0 && ii <= _ni-1 && jj >= 0 && jj <= _nj-1)
                {
                    max_v = std::max(max_v, before.at(ii,jj));
                    min_v = std::min(min_v, before.at(ii,jj));
                }
            }
        }
        after(i,j) = std::min(std::max(after.at(i,j),min_v),max_v);
    });
}

void CovectorSolver2D::accumulateScalars(Array2f &rho_change, Array2f &T_change, bool error_correction)
{
    Array2f temp_scalar;
    Array2f T_scalar;
    Array2f temp_scalar_star;
    Array2f T_scalar_star;
    temp_scalar.resize(ni, nj, 0.0);
    T_scalar.resize(ni, nj, 0.0);
    temp_scalar_star.resize(ni, nj, 0.0);
    T_scalar_star.resize(ni, nj, 0.0);
    tbb::parallel_for((int) 0, ni * nj, 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        for(int k = start_idx; k < end_idx; k++)
        {
            Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.5) + h * dir[k];
            float x_init = sampleField(pos - h * Vec2f(0.5), forward_scalar_x);
            float y_init = sampleField(pos - h * Vec2f(0.5), forward_scalar_y);
            Vec2f samplePos = Vec2f(x_init, y_init);
            temp_scalar(i, j) += w[k]*sampleField(samplePos - h * Vec2f(0.5, 0.5), rho_change);
        }
    });
    if (error_correction)
    {
        tbb::parallel_for((int) 0, ni * nj, 1, [&](int tIdx) {
            int i = tIdx % ni;
            int j = tIdx / ni;
            for(int k = start_idx; k < end_idx; k++)
            {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.5) + h * dir[k];
                float x_init = sampleField(pos - h * Vec2f(0.5), backward_scalar_x);
                float y_init = sampleField(pos - h * Vec2f(0.5), backward_scalar_y);
                Vec2f samplePos = Vec2f(x_init, y_init);
                temp_scalar_star(i, j) += w[k]*sampleField(samplePos - h * Vec2f(0.5, 0.5), temp_scalar);
            }
        });
        temp_scalar_star -= rho_change;
        temp_scalar_star *= 0.5f;
    }
    tbb::parallel_for((int) 0, ni * nj, 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        for(int k = start_idx; k < end_idx; k++)
        {
            Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.5) + h * dir[k];
            float x_init = sampleField(pos - h * Vec2f(0.5), forward_scalar_x);
            float y_init = sampleField(pos - h * Vec2f(0.5), forward_scalar_y);
            Vec2f samplePos = Vec2f(x_init, y_init);
            if (error_correction){
                drho(i,j) += w[k]*(sampleField(samplePos - h * Vec2f(0.5, 0.5), rho_change) -
                                    sampleField(samplePos - h * Vec2f(0.5, 0.5), temp_scalar_star));
            }
            else{
                drho(i,j) += w[k]*(sampleField(samplePos - h * Vec2f(0.5, 0.5), rho_change));
            }
        }
    });
    /// T
    tbb::parallel_for((int) 0, ni * nj, 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        for(int k = start_idx; k < end_idx; k++)
        {
            Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.5) + h * dir[k];
            float x_init = sampleField(pos - h * Vec2f(0.5), forward_scalar_x);
            float y_init = sampleField(pos - h * Vec2f(0.5), forward_scalar_y);
            Vec2f samplePos = Vec2f(x_init, y_init);
            T_scalar(i, j) += w[k]*sampleField(samplePos - h * Vec2f(0.5, 0.5), T_change);
        }
    });
    if (error_correction)
    {
        tbb::parallel_for((int) 0, ni * nj, 1, [&](int tIdx) {
            int i = tIdx % ni;
            int j = tIdx / ni;
            for(int k = start_idx; k < end_idx; k++)
            {
                Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.5) + h * dir[k];
                float x_init = sampleField(pos - h * Vec2f(0.5), backward_scalar_x);
                float y_init = sampleField(pos - h * Vec2f(0.5), backward_scalar_y);
                Vec2f samplePos = Vec2f(x_init, y_init);
                T_scalar_star(i, j) += w[k]*sampleField(samplePos - h * Vec2f(0.5, 0.5), T_scalar);
            }
        });
        T_scalar_star -= T_change;
        T_scalar_star *= 0.5f;
    }
    tbb::parallel_for((int) 0, ni * nj, 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        for(int k = start_idx; k < end_idx; k++)
        {
            Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.5) + h * dir[k];
            float x_init = sampleField(pos - h * Vec2f(0.5), forward_scalar_x);
            float y_init = sampleField(pos - h * Vec2f(0.5), forward_scalar_y);
            Vec2f samplePos = Vec2f(x_init, y_init);
            if (error_correction){
                dT(i,j) += w[k]*(sampleField(samplePos - h * Vec2f(0.5, 0.5), T_change) -
                                    sampleField(samplePos - h * Vec2f(0.5, 0.5), T_scalar_star));
            }
            else{
                dT(i,j) += w[k]*(sampleField(samplePos - h * Vec2f(0.5, 0.5), T_change));
            }
        }
    });
}

void CovectorSolver2D::resampleVelBuffer(float dt)
{
    std::cout<< RED << "velocity remeshing!\n" << RESET;
    total_resampleCount ++;
    u_origin = u_init;
    v_origin = v_init;
    u_init = u;
    v_init = v;
    du_prev = du;
    dv_prev = dv;
    du.assign(0);
    dv.assign(0);

    backward_xprev = backward_x;
    backward_yprev = backward_y;
    tbb::parallel_for((int)0, ni*nj, 1, [&](int tIdx)
    {
        int i = tIdx%ni; int j = tIdx / ni;
        forward_x(i, j) = h*((float)i + 0.5);
        backward_x(i, j) = h*((float)i + 0.5);
        forward_y(i, j) = h*((float)j + 0.5);
        backward_y(i, j) = h*((float)j + 0.5);
    });
}

void CovectorSolver2D::resampleRhoBuffer(float dt)
{
    std::cout<< RED << "rho remeshing!\n" << RESET;
    total_scalar_resample ++;
    rho_orig = rho_init;
    rho_init = rho;
    T_orig = T_init;
    T_init = temperature;
    drho_prev = drho;
    dT_prev = dT;
    cv_u_init = cv_u;
    cv_v_init = cv_v;
    drho.assign(0);
    dT.assign(0);

    backward_scalar_xprev = backward_scalar_x;
    backward_scalar_yprev = backward_scalar_y;
    tbb::parallel_for((int)0, ni*nj, 1, [&](int tIdx)
    {
        int i = tIdx%ni; int j = tIdx / ni;
        forward_scalar_x(i, j) = h*((float)i + 0.5);
        backward_scalar_x(i, j) = h*((float)i + 0.5);
        forward_scalar_y(i, j) = h*((float)j + 0.5);
        backward_scalar_y(i, j) = h*((float)j + 0.5);
    });
}

void CovectorSolver2D::setSmoke(float smoke_rise, float smoke_drop)
{
    alpha = smoke_drop;
    beta = smoke_rise;
}

void CovectorSolver2D::diffuseField(float nu, float dt, Array2f &field)
{
    Array2f field_temp;
    field_temp = field;
    double coef = nu*(dt/(h*h));
    int n = field.ni*field.nj;
    for(int iter=0;iter<20;iter++)
    {
        tbb::parallel_for(0,n,1,[&](int tid)
        {
            int i = tid%field.ni;
            int j = tid/field.ni;
            if((i+j)%2==0)
            {
                float b_ij = field(i,j);
                float x_l  = ((i-1)>=0)?field_temp(i-1,j):0;
                float x_r  = ((i+1)<field.ni)?field_temp(i+1,j):0;
                float x_u  = ((j+1)<field.nj)?field_temp(i,j+1):0;
                float x_d  = ((j-1)>=0)?field_temp(i,j-1):0;

                field_temp(i,j) = (b_ij + coef*(x_l + x_r + x_u + x_d))/(1.0+4.0*coef);
            }
        });
        tbb::parallel_for(0,n,1,[&](int tid)
        {
            int i = tid%field.ni;
            int j = tid/field.ni;
            if((i+j)%2==1)
            {
                float b_ij = field(i,j);
                float x_l  = ((i-1)>=0)?field_temp(i-1,j):0;
                float x_r  = ((i+1)<field.ni)?field_temp(i+1,j):0;
                float x_u  = ((j+1)<field.nj)?field_temp(i,j+1):0;
                float x_d  = ((j-1)>=0)?field_temp(i,j-1):0;

                field_temp(i,j) = (b_ij + coef*(x_l + x_r + x_u + x_d))/(1.0+4.0*coef);
            }
        });
    }
    field = field_temp;
}

void CovectorSolver2D::advanceMaccormack(float dt, int currentframe)
{
    std::cout << BLUE <<  "MacCormack scheme frame " << currentframe << " starts !" << RESET << std::endl;

    // advect rho
    Array2f rho_first;
    Array2f rho_sec;
    rho_first.resize(ni, nj, 0.0);
    rho_sec.resize(ni, nj, 0.0);
    solveMaccormack(rho, rho_first, rho_sec, dt, ni, nj, 0.5, 0.5);
    rho = rho_first;
    if (do_clear_boundaries)
        clearBoundaries();

    if (advect_levelset_covector)
    {
        u_first.assign(ni+1, nj, 0.0);
        v_first.assign(ni, nj+1, 0.0);
        u_sec.assign(ni+1, nj, 0.0);
        v_sec.assign(ni, nj+1, 0.0);
        solveMaccormack(cv_u, u_first, u_sec, dt, ni+1, nj, 0.0, 0.5);
        solveMaccormack(cv_v, v_first, v_sec, dt, ni, nj+1, 0.5, 0.0);
        cv_u = u_first;
        cv_v = v_first;
    }

    if (!advect_levelset)
    {
        // advect temperature
        Array2f T_first;
        Array2f T_sec;
        T_first.resize(ni, nj, 0.0);
        T_sec.resize(ni, nj, 0.0);
        solveMaccormack(temperature, T_first, T_sec, dt, ni, nj, 0.5, 0.5);
        temperature = T_first;

        // advect velocity
        u_first.assign(ni+1, nj, 0.0);
        v_first.assign(ni, nj+1, 0.0);
        u_sec.assign(ni+1, nj, 0.0);
        v_sec.assign(ni, nj+1, 0.0);
        solveMaccormack(u, u_first, u_sec, dt, ni+1, nj, 0.0, 0.5);
        solveMaccormack(v, v_first, v_sec, dt, ni, nj+1, 0.5, 0.0);
        u = u_first;
        v = v_first;

        applyBuoyancyForce(v, dt);
        projection(1e-6,use_neumann_boundary);
    }
}

void CovectorSolver2D::setInitVelocity(float distance, bool do_taylor_green)
{
    if (do_taylor_green)
    {
        tbb::parallel_for((int)0, (ni+1)*(nj), 1 , [&](int thread_Idx) {
            int j = thread_Idx/(ni+1);
            int i = thread_Idx%(ni+1);
            Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.0, 0.5);
            u(i,j) = sin(pos[0])*cos(pos[1]);
            u_init(i,j) = u(i,j);
            u_origin(i,j) = u(i,j);
        });
        tbb::parallel_for((int)0, (ni)*(nj+1), 1 , [&](int thread_Idx) {
            int j = thread_Idx/(ni);
            int i = thread_Idx%(ni);
            Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.0);
            v(i,j) = -cos(pos[0])*sin(pos[1]);
            v_init(i,j) = v(i,j);
            v_origin(i,j) = v(i,j);
        });

        return;     
    }

    SparseMatrixd M;
    int n = ni*nj;
    M.resize(n);
    tbb::parallel_for((int)0, ni*nj, 1, [&](int thread_idx)
    {
        int j = thread_idx / ni;
        int i = thread_idx % ni;
        //if in fluid domain
        if (i >= 0 && j >= 0 && i < ni&&j < nj)
        {
            {
                if (i-1>=0 ){
                    M.add_to_element(thread_idx, thread_idx - 1, -1 / (h * h));
                    M.add_to_element(thread_idx, thread_idx, 1 / (h * h));
                }
                else
                {
                    M.add_to_element(thread_idx, thread_idx, 1 / (h * h));
                }

                if (i+1<ni ){
                    M.add_to_element(thread_idx, thread_idx + 1, -1 / (h * h));
                    M.add_to_element(thread_idx, thread_idx, 1 / (h * h));
                }
                else
                {
                    M.add_to_element(thread_idx, thread_idx, 1 / (h * h));
                }

                if (j-1>=0 ){
                    M.add_to_element(thread_idx, thread_idx - ni, -1 / (h * h));
                    M.add_to_element(thread_idx, thread_idx, 1 / (h * h));
                }
                else
                {
                    M.add_to_element(thread_idx, thread_idx, 1 / (h * h));
                }

                if (j+1<nj ){
                    M.add_to_element(thread_idx, thread_idx + ni, -1 / (h * h));
                    M.add_to_element(thread_idx, thread_idx, 1 / (h * h));
                }
                else
                {
                    M.add_to_element(thread_idx, thread_idx, 1 / (h * h));
                }
            }
        }
    });
    FixedSparseMatrixd M_fix;
    M_fix.construct_from_matrix(M);
    std::vector<FixedSparseMatrixd *> AA_L;
    std::vector<FixedSparseMatrixd *> RR_L;
    std::vector<FixedSparseMatrixd *> PP_L;
    std::vector<Vec2i>                SS_L;
    int ttotal_level;
    mgLevelGenerator.generateLevelsGalerkinCoarsening2D(AA_L, RR_L, PP_L, SS_L, ttotal_level, M_fix, ni, nj);

    //initialize curl;
    float max_curl=0;
    tbb::parallel_for((int)0, (ni+1)*(nj+1), 1 , [&](int thread_Idx)
    {
        int j = thread_Idx/(ni+1);
        int i = thread_Idx%(ni+1);
        Vec2f pos = h*Vec2f(i, j) - Vec2f(M_PI);
        Vec2f vort_pos0 = Vec2f(-0.5*distance,0);
        Vec2f vort_pos1 = Vec2f(+0.5*distance,0);
        double r_sqr0 = dist2(pos, vort_pos0);
        double r_sqr1 = dist2(pos, vort_pos1);
        curl(i,j) = +1.0/0.3*(2.0 - r_sqr0/0.09)*exp(0.5*(1.0 - r_sqr0/0.09));
        curl(i,j) += 1.0/0.3*(2.0 - r_sqr1/0.09)*exp(0.5*(1.0 - r_sqr1/0.09));
        max_curl = std::max(fabs(curl(i,j)), max_curl);
    }
    );
    rhs.assign(ni*nj,0);
    pressure.resize(ni*nj);
    //compute stream function
    tbb::parallel_for((int)0, (ni)*(nj), 1 , [&](int thread_Idx) {
        int j = thread_Idx/(ni);
        int i = thread_Idx%(ni);
        rhs[j*ni + i] = curl(i,j);
        pressure[j*ni+i] = 0;
    }
    );
    double res_out; int iter_out;
    bool converged = AMGPCGSolvePrebuilt2D(M_fix,rhs,pressure,AA_L,RR_L,PP_L,SS_L,ttotal_level,1e-6,500,res_out,iter_out,ni,nj, false);
    if (converged)
        std::cout << "pressure solver converged in " << iter_out << " iterations, with residual " << res_out << std::endl;
    else
        std::cout << "warning! solver didn't reach convergence with maximum iterations!" << std::endl;

    curl.assign(0);
    //compute u = curl psi
    tbb::parallel_for((int)0, (ni)*(nj), 1 , [&](int thread_Idx) {
        int j = thread_Idx/(ni);
        int i = thread_Idx%(ni);
        curl(i,j) = pressure[j*ni+i];
    }
    );

    tbb::parallel_for((int)0, (ni+1)*(nj), 1 , [&](int thread_Idx) {
        int j = thread_Idx/(ni+1);
        int i = thread_Idx%(ni+1);
        u(i,j) = (curl(i, j+1) - curl(i,j))/h;
        u_init(i,j) = (curl(i, j+1) - curl(i,j))/h;
        u_origin(i,j) = (curl(i, j+1) - curl(i,j))/h;
    });
    tbb::parallel_for((int)0, (ni)*(nj+1), 1 , [&](int thread_Idx) {
        int j = thread_Idx/(ni);
        int i = thread_Idx%(ni);
        v(i,j) = -(curl(i+1, j) - curl(i,j))/h;
        v_init(i,j) = -(curl(i+1, j) - curl(i,j))/h;
        v_origin(i,j) = -(curl(i+1, j) - curl(i,j))/h;
    });
    cBar = color_bar(max_curl);
}

void CovectorSolver2D::setInitReyleighTaylor(float layer_height)
{
    tbb::parallel_for((int)0, ni*nj, 1, [&](int tIdx)
    {
        int i = tIdx%ni;
        int j = tIdx / ni;
        Vec2f pos = h*(Vec2f(i,j) + Vec2f(0.5, 0.5));
        Vec2f center(0.1,layer_height);
        float radius = 0.04;
        if (dist(center,pos) < radius)
        {
            rho(i,j) = 1.f;
            rho_init(i,j) = 1.f;
            rho_orig(i,j) = 1.f;
        }
        else{
            temperature(i,j) = 1.f;
            T_init(i,j) = 1.f;
            T_orig(i,j) = 1.f;
        }
    });
}

void CovectorSolver2D::clearBoundaries()
{
    tbb::parallel_for((int)0, ni*nj, 1, [&](int tIdx)
    {
        int i = tIdx%ni;
        int j = tIdx / ni;
        if (i == 0 || i == ni-1 || j == 0 || j == nj-1)
        {
            rho(i,j) = rho_boundary_value;
            temperature(i,j) = T_boundary_value;
        }
    });   
}

void CovectorSolver2D::setInitLeapFrog(float dist_a, float dist_b, float rho_h, float rho_w)
{
    //initialize curl;
    float max_curl=0;
    float a = 0.02f;
    tbb::parallel_for((int)0, (ni+1)*(nj+1), 1 , [&](int thread_Idx)
      {
          int j = thread_Idx/(ni+1);
          int i = thread_Idx%(ni+1);
          Vec2f pos = h*Vec2f(i, j) - Vec2f(M_PI);
          Vec2f vort_pos0 = Vec2f(-0.5*dist_a,-2.0f);
          Vec2f vort_pos1 = Vec2f(+0.5*dist_a,-2.0f);
          Vec2f vort_pos2 = Vec2f(-0.5*dist_b,-2.0f);
          Vec2f vort_pos3 = Vec2f(+0.5*dist_b,-2.0f);
          double r_sqr0 = dist2(pos, vort_pos0);
          double r_sqr1 = dist2(pos, vort_pos1);
          double r_sqr2 = dist2(pos, vort_pos2);
          double r_sqr3 = dist2(pos, vort_pos3);
          float c_a = 1000.0/(2.0*3.1415926)*exp(-0.5*(r_sqr0)/a/a);
          float c_b = -1000.0/(2.0*3.1415926)*exp(-0.5*(r_sqr1)/a/a);
          float c_c = 1000.0/(2.0*3.1415926)*exp(-0.5*(r_sqr2)/a/a);
          float c_d = -1000.0/(2.0*3.1415926)*exp(-0.5*(r_sqr3)/a/a);
          curl(i,j) += c_a;
          curl(i,j) += c_b;
          curl(i,j) += c_c;
          curl(i,j) += c_d;
          max_curl = std::max(fabs(curl(i,j)), max_curl);
      }
    );
    rhs.assign(ni*nj,0);
    pressure.resize(ni*nj);
    //compute stream function
    tbb::parallel_for((int)0, (ni)*(nj), 1 , [&](int thread_Idx) {
                          int j = thread_Idx/(ni);
                          int i = thread_Idx%(ni);
                          rhs[j*ni + i] = curl(i,j);
                          pressure[j*ni+i] = 0;
                      }
    );
    double res_out; int iter_out;
    bool converged = AMGPCGSolvePrebuilt2D(matrix_fix,rhs,pressure,A_L,R_L,P_L,S_L,total_level,1e-6,500,res_out,iter_out,ni,nj, false);
    if (converged)
        std::cout << "pressure solver converged in " << iter_out << " iterations, with residual " << res_out << std::endl;
    else
        std::cout << "warning! solver didn't reach convergence with maximum iterations!" << std::endl;

    curl.assign(0);
    //compute u = curl psi
    tbb::parallel_for((int)0, (ni)*(nj), 1 , [&](int thread_Idx) {
                          int j = thread_Idx/(ni);
                          int i = thread_Idx%(ni);
                          curl(i,j) = pressure[j*ni+i];
                      }
    );

    tbb::parallel_for((int)0, (ni+1)*(nj), 1 , [&](int thread_Idx) {
        int j = thread_Idx/(ni+1);
        int i = thread_Idx%(ni+1);
        u(i,j) = (curl(i, j+1) - curl(i,j))/h;
        u_init(i,j) = (curl(i, j+1) - curl(i,j))/h;
        u_origin(i,j) = (curl(i, j+1) - curl(i,j))/h;
    });
    tbb::parallel_for((int)0, (ni)*(nj+1), 1 , [&](int thread_Idx) {
        int j = thread_Idx/(ni);
        int i = thread_Idx%(ni);
        v(i,j) = -(curl(i+1, j) - curl(i,j))/h;
        v_init(i,j) = -(curl(i+1, j) - curl(i,j))/h;
        v_origin(i,j) = -(curl(i+1, j) - curl(i,j))/h;
    });
    cBar = color_bar(max_curl);

    tbb::parallel_for((int)0, ni*nj, 1, [&](int tIdx)
    {
        int i = tIdx%ni;
        int j = tIdx / ni;
        Vec2f pos = h*(Vec2f(i,j) + Vec2f(0.5, 0.5));
        if (rho_h - rho_w < pos[1] && pos[1] < rho_h + rho_w && pos[0] > rho_w && pos[0] < 2*M_PI - rho_w)
        {
            rho(i,j) = 1.f;
            rho_init(i,j) = 1.f;
            rho_orig(i,j) = 1.f;
        }
        else {
            temperature(i,j) = 1.f;
            T_init(i,j) = 1.f;
            T_orig(i,j) = 1.f;
        }
    });

}

void CovectorSolver2D::setInitZalesak()
{
    // for circle
    float r = 0.1*ni*h;
    float center_x = 0.5*ni*h;
    float center_y = 0.65*ni*h;
    // for rectangle
    float width = 0.04*ni*h;
    float height = 0.20*ni*h;
    float rec_x = 0.5*ni*h;
    float rec_y = 0.6*ni*h;
    tbb::parallel_for((int)0, ni*nj, 1, [&](int tIdx)
    {
        int i = tIdx%ni;
        int j = tIdx / ni;
        float pos_x = h*(i + 0.5);
        float pos_y = h*(j + 0.5);
        float circle = sqrt((pos_x-center_x)*(pos_x-center_x) + (pos_y-center_y)*(pos_y-center_y)) - r;
        Vec2f p = Vec2f(pos_x, pos_y) - Vec2f(rec_x, rec_y);
        Vec2f d = Vec2f(abs(p[0]), abs(p[1])) - .5f*Vec2f(width, height);
        Vec2f maxv = max_union(d,Vec2f(0));
        float rec = dist(maxv, Vec2f(0.0)) + min(max(d[0], d[1]), 0.f);
        rho(i,j) = max(circle,-rec);
        rho_init(i,j) = max(circle,-rec);
        rho_orig(i,j) = max(circle,-rec);
    });
    // init velocity field
    Vec2f center = Vec2f(0.5*ni*h, 0.5*ni*h);
    tbb::parallel_for((int) 0, (ni + 1) * nj, 1, [&](int tIdx) {
        int i = tIdx % (ni + 1);
        int j = tIdx / (ni + 1);
        Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.0, 0.5);
        u(i, j) = M_PI*(0.5*ni*h - pos.v[1]) / 314.f;
        u_init(i, j) = M_PI*(0.5*ni*h - pos.v[1]) / 314.f;
        u_origin(i, j) = M_PI*(0.5*ni*h - pos.v[1]) / 314.f;
    });
    tbb::parallel_for((int) 0, ni * (nj + 1), 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.0);
        v(i, j) = M_PI*(pos.v[0] - 0.5*ni*h) / 314.f;
        v_init(i, j) = M_PI*(pos.v[0] - 0.5*ni*h) / 314.f;
        v_origin(i, j) = M_PI*(pos.v[0] - 0.5*ni*h) / 314.f;
    });
}

void CovectorSolver2D::setInitInvertedZalesak()
{
    // for circle
    float r = 0.1*ni*h;
    float center_x = 0.5*ni*h;
    float center_y = 0.65*ni*h;
    // for rectangle
    float width = 0.04*ni*h;
    float height = 0.20*ni*h;
    float rec_x = 0.5*ni*h;
    float rec_y = 0.6*ni*h;
    tbb::parallel_for((int)0, ni*nj, 1, [&](int tIdx)
    {
        int i = tIdx%ni;
        int j = tIdx / ni;
        float pos_x = h*(i + 0.5);
        float pos_y = h*(j + 0.5);
        float circle = sqrt((pos_x-center_x)*(pos_x-center_x) + (pos_y-center_y)*(pos_y-center_y)) - r;
        Vec2f p = Vec2f(pos_x, pos_y) - Vec2f(rec_x, rec_y);
        Vec2f d = Vec2f(abs(p[0]), abs(p[1])) - .5f*Vec2f(width, height);
        Vec2f maxv = max_union(d,Vec2f(0));
        float rec = dist(maxv, Vec2f(0.0)) + min(max(d[0], d[1]), 0.f);
        rho(i,j) = max(circle,-rec);
        rho_init(i,j) = max(circle,-rec);
        rho_orig(i,j) = max(circle,-rec);
    });

    cv_u.assign(0.0f);
    cv_v.assign(0.0f);
    tbb::parallel_for((int)0, (ni+1) * nj, 1, [&](int tIdx) {
        int i = tIdx % (ni+1);
        int j = tIdx / (ni+1);
        if (i>0 && i<ni)
        {
            float sdf_front = rho(i,j);
            float sdf_back = rho(i-1,j);
            float value_front = sdf_front <= 0 ? 1.0f : 0.0f;
            float value_back = sdf_back <= 0 ? 1.0f : 0.0f;
            cv_u(i, j) = (value_back * abs(sdf_front) + value_front * abs(sdf_back)) / (abs(sdf_front) + abs(sdf_back));
        }
    });
    tbb::parallel_for((int)0, ni * (nj+1), 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        if (j>0 && j<nj)
        {
            Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.0);
            if (dist(pos,Vec2f(0.5*ni*h, 0.5*ni*h)) < 0.35)
            {
            float sdf_front = rho(i,j);
            float sdf_back = rho(i,j-1);
            float value_front = sdf_front <= 0 ? 0.0f : 1.0f;
            float value_back = sdf_back <= 0 ? 0.0f : 1.0f;
            cv_v(i, j) = (value_back * abs(sdf_front) + value_front * abs(sdf_back)) / (abs(sdf_front) + abs(sdf_back));
            }
        }
    });
    cv_u_init = cv_u;
    cv_v_init = cv_v;

    // init velocity field
    Vec2f center = Vec2f(0.5*ni*h, 0.5*ni*h);
    tbb::parallel_for((int) 0, (ni + 1) * nj, 1, [&](int tIdx) {
        int i = tIdx % (ni + 1);
        int j = tIdx / (ni + 1);
        Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.0, 0.5);
        u(i, j) = M_PI*(center.v[1] - pos.v[1]) / 314.f;
        u_init(i, j) = M_PI*(center.v[1] - pos.v[1]) / 314.f;
        u_origin(i, j) = M_PI*(center.v[1] - pos.v[1]) / 314.f;
    });
    tbb::parallel_for((int) 0, ni * (nj + 1), 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.0);
        v(i, j) = M_PI*(pos.v[0] - center.v[0]) / 314.f;
        v_init(i, j) = M_PI*(pos.v[0] - center.v[0]) / 314.f;
        v_origin(i, j) = M_PI*(pos.v[0] - center.v[0]) / 314.f;
    });
}

void CovectorSolver2D::setInitVortexBox()
{
    // for circle
    float r = 0.15*ni*h;
    float center_x = 0.5*ni*h;
    float center_y = 0.75*ni*h;
    float normalize = 0.f;
    tbb::parallel_for((int)0, ni*nj, 1, [&](int tIdx)
    {
        int i = tIdx%ni;
        int j = tIdx / ni;
        float pos_x = h*(i + 0.5);
        float pos_y = h*(j + 0.5);
        float circle = sqrt((pos_x-center_x)*(pos_x-center_x) + (pos_y-center_y)*(pos_y-center_y)) - r;
        rho(i,j) = circle;
        rho_init(i,j) = circle;
        rho_orig(i,j) = circle;
        pos_x /= ni*h;
        pos_y /= nj*h;
        float tmp_x = -2.f*sin(M_PI*pos_x)*sin(M_PI*pos_x)*sin(M_PI*pos_y)*cos(M_PI*pos_y);
        float tmp_y = 2.f*sin(M_PI*pos_x)*cos(M_PI*pos_x)*sin(M_PI*pos_y)*sin(M_PI*pos_y);
        float mag = sqrt(tmp_x*tmp_x+tmp_y*tmp_y);
        if (mag > normalize) normalize = mag;
    });
    tbb::parallel_for((int) 0, (ni + 1) * nj, 1, [&](int tIdx) {
        int i = tIdx % (ni + 1);
        int j = tIdx / (ni + 1);
        Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.0, 0.5);
        float pos_x = pos[0] / (ni*h);
        float pos_y = pos[1] / (nj*h);
        u(i, j) = -2.f*sin(M_PI*pos_x)*sin(M_PI*pos_x)*sin(M_PI*pos_y)*cos(M_PI*pos_y) / normalize;
    });
    tbb::parallel_for((int) 0, ni * (nj + 1), 1, [&](int tIdx) {
        int i = tIdx % ni;
        int j = tIdx / ni;
        Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.0);
        float pos_x = pos[0] / (ni*h);
        float pos_y = pos[1] / (nj*h);
        v(i, j) = 2.f*sin(M_PI*pos_x)*cos(M_PI*pos_x)*sin(M_PI*pos_y)*sin(M_PI*pos_y) / normalize;
    });
}

void CovectorSolver2D::setBoundaryMask(std::function<float(Vec2f pos)> sdf)
{
    tbb::parallel_for((int)0, ni*nj, 1, [&](int thread_idx)
    {
        int j = thread_idx / ni;
        int i = thread_idx % ni;

        //fluid=0, air=1, boundary=2, obstacle=3
        boundaryMask(i,j) = 0;

        Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5); 
        if (sdf && sdf(pos) <= 0) boundaryMask(i,j) = 3;
    });
}

void CovectorSolver2D::initKarmanVelocity()
{
    // init background velocity to (1,0)
    tbb::parallel_for((int)0, (ni+1)*nj, 1, [&](int thread_idx)
    {
        int j = thread_idx / (ni+1);
        int i = thread_idx % (ni+1);

        u(i,j) = karman_velocity_value;
    });
    tbb::parallel_for((int)0, ni*(nj+1), 1, [&](int thread_idx)
    {
        int j = thread_idx / ni;
        int i = thread_idx % ni;
        
        v(i,j)=0;
    });  
}

void CovectorSolver2D::setKarmanDensity(bool do_init)
{

    tbb::parallel_for((int)0, ni*nj, 1, [&](int tIdx)
    {
        int i = tIdx%ni;
        int j = tIdx / ni;

        if (boundaryMask(i,j) == 3 || boundaryMask(i,j-1) == 3  || boundaryMask(i,j+1) == 3)
        {
            rho(i,j) = 10.f;
            rho_init(i,j) = 10.f;
            rho_orig(i,j) = 10.f;
        }
        else if (do_init){
            temperature(i,j) = 1.f;
            T_init(i,j) = 1.f;
            T_orig(i,j) = 1.f;
        }
    });   
}

void CovectorSolver2D::setKarmanVelocity()
{
    // set in-let velocity to (1,0)
    // left hand boundary is pure neumann but with velocity
    // being (1,0) instead of zero
    tbb::parallel_for((int)0, (ni+1)*nj, 1, [&](int thread_idx)
    {
        int j = thread_idx / (ni+1);
        int i = thread_idx % (ni+1);

        // if (i>=0 && i<=3)
        if (i==0)
        {
            u(i,j)=karman_velocity_value;
        }
    });
    // make sure top and bottom velocities are pure neumann even though
    // right hand boundary is open.
    tbb::parallel_for((int)0, ni*nj, 1, [&](int tIdx) {
        int i = tIdx%ni;
        int j = tIdx / ni;

        if(j==0)
        {
            v(i, j) = 0;
        }
        if(j==nj-1)
        {
            v(i, j + 1) = 0;
        }
    });
}

void CovectorSolver2D::buildMultiGrid(bool PURE_NEUMANN)
{
    //build the matrix
    //we are assuming a a whole fluid domain
    int n = ni*nj;
    matrix.resize(n);
    tbb::parallel_for((int)0, ni*nj, 1, [&](int thread_idx)
    {
        int j = thread_idx / ni;
        int i = thread_idx % ni;
        //if in fluid domain
        if (i >= 0 && j >= 0 && i < ni&&j < nj)
        {
            if (i-1>=0 && boundaryMask(i-1,j) != 3){
                matrix.add_to_element(thread_idx, thread_idx - 1, -1 / (h * h));
                matrix.add_to_element(thread_idx, thread_idx, 1 / (h * h));
            }
            else
            {
                if (!PURE_NEUMANN && !do_karman_velocity_setup) matrix.add_to_element(thread_idx, thread_idx, 1 / (h * h));
            }

            if (i+1<ni && boundaryMask(i+1,j) != 3){
                matrix.add_to_element(thread_idx, thread_idx + 1, -1 / (h * h));
                matrix.add_to_element(thread_idx, thread_idx, 1 / (h * h));
            }
            else
            {
                if (!PURE_NEUMANN) matrix.add_to_element(thread_idx, thread_idx, 1 / (h * h));
            }

            if (j-1>=0 && boundaryMask(i,j-1) != 3){
                matrix.add_to_element(thread_idx, thread_idx - ni, -1 / (h * h));
                matrix.add_to_element(thread_idx, thread_idx, 1 / (h * h));
            }
            else
            {
                if (!PURE_NEUMANN && !do_karman_velocity_setup) matrix.add_to_element(thread_idx, thread_idx, 1 / (h * h));
            }

            if (j+1<nj && boundaryMask(i,j+1) != 3){
                matrix.add_to_element(thread_idx, thread_idx + ni, -1 / (h * h));
                matrix.add_to_element(thread_idx, thread_idx, 1 / (h * h));
            }
            else
            {
                if (!PURE_NEUMANN && !do_karman_velocity_setup) matrix.add_to_element(thread_idx, thread_idx, 1 / (h * h));
            }
        }
    });
    matrix_fix.construct_from_matrix(matrix);
    mgLevelGenerator.generateLevelsGalerkinCoarsening2D(A_L, R_L, P_L, S_L, total_level, matrix_fix, ni, nj);
}

void CovectorSolver2D::applyVelocityBoundary(bool do_set_obstacle_vel)
{
    tbb::parallel_for((int)0, ni*nj, 1, [&](int tIdx) {
        int i = tIdx%ni;
        int j = tIdx / ni;
        if (use_neumann_boundary)
        {
            if(i==0)
            {
                u(i,j) = 0;
            }
            if(j==0)
            {
                v(i, j) = 0;
            }
            if(i==ni-1)
            {
                u(i + 1, j) = 0;
            }
            if(j==nj-1)
            {
                v(i, j + 1) = 0;
            }
        }

        if (do_set_obstacle_vel && boundaryMask(i,j) == 3)
        {
            u(i,j) = 0;
            u(i+1,j) = 0;
            v(i, j) = 0;
            v(i, j+1) = 0;
        }
    });
}

void CovectorSolver2D::calculateCurl(bool set_color_bar_with_max_curl) {
    curl.assign(0);
    tbb::parallel_for((int)0, (ni+1)*(nj+1), 1, [&](int tIdx)
    {
        int i = tIdx%(ni+1);
        int j = tIdx/(ni+1);
        if(i>0&&i<ni&&j>0&&j<nj)
        {
            curl(i,j) = -(u(i,j) - u(i,j-1) + v(i-1,j) - v(i,j))/h;
            m_max_curl = max(m_max_curl, abs(curl(i,j)));
        }
    });
    if (set_color_bar_with_max_curl)
        cBar = color_bar(m_max_curl);
}

void CovectorSolver2D::emitSmoke()
{
    tbb::parallel_for((int)0, ni*nj, 1, [&](int tIdx)
    {
        int i = tIdx%ni; int j = tIdx / ni;
        if (emitterMask(i, j) == 1)
        {
            rho(i, j) = 1.0;
            temperature(i, j) = 1.0;
            if(rho_init.a.size()>0)
            {
                rho_init(i, j) = 1.0;
            }
            if(T_init.a.size()>0)
            {
                T_init(i,j) = 1.0;
            }
        }
    });
}

Vec2f CovectorSolver2D::getVelocity(const Vec2f & pos, const Array2f& un, const Array2f& vn)
{
    float u_sample, v_sample;
    //offset of u, we are in a staggered grid
    Vec2f upos = pos - Vec2f(0.0f, 0.5*h);
    u_sample = sampleField(upos, un);

    //offset of v, we are in a staggered grid
    Vec2f vpos = pos - Vec2f(0.5*h, 0.0f);
    v_sample = sampleField(vpos, vn);

    return Vec2f(u_sample, v_sample);
}


float CovectorSolver2D::sampleField(const Vec2f pos, const Array2f &field)
{
    Vec2f spos = pos;
    int i = floor(spos.v[0] / h), j = floor(spos.v[1] / h);
    return bilerp(field.boundedAt(i, j), field.boundedAt(i + 1, j),
                  field.boundedAt(i, j + 1), field.boundedAt(i + 1, j + 1), spos.v[0] / h - (float)i, spos.v[1] / h - (float)j);
}

void CovectorSolver2D::initDensityFromFile(std::string filepath, int pixel_shift_up, int pixel_shift_right, int img_size)
{
    std::string filename(filepath);

    std::ifstream input_file(filename);
    if (!input_file.is_open())
    {
        cerr << "Could not open the file - '" << filename << "'" << endl;
        return;
    }

    int index = 0;
    temperature.assign(1);
    T_init.assign(1);
    T_orig.assign(1);
    std::string line;
    while (std::getline(input_file, line))
    {
        std::stringstream text_stream(line);
        std::string item;
        while (std::getline(text_stream, item, ' ')) {
            int read_value = std::stoi(item);
            int i = index%img_size;
            int j = index/img_size;
            j+=pixel_shift_up;
            i+=pixel_shift_right;
            if (read_value)
            {
                temperature(i,j) = 1;
                T_init(i,j) = 1.f;
                T_orig(i,j) = 1.f;
            }
            else
            {
                temperature(i,j) = 0;
                T_init(i,j) = 0;
                T_orig(i,j) = 0;
                rho(i,j) = 1;
                rho_init(i,j) = 1.f;
                rho_orig(i,j) = 1.f;
            }
            index++;
        }
    }

    input_file.close();
    return;
}

void CovectorSolver2D::outputDensity(std::string folder, std::string file, int i, bool color_density, bool do_tonemapping)
{
    boost::filesystem::create_directories(folder);
    std::string filestr;
    filestr = folder + file + std::string("_\%04d.bmp");
    char filename[1024];
    sprintf(filename, filestr.c_str(), i);
    if (color_density)
    {
        std::vector<Vec3uc> color;
        color.resize(ni*nj);
        cBar = color_bar(0,1);
        tbb::parallel_for((int)0, (int)(ni*nj), 1, [&](int tIdx)
        {
            int i = tIdx%ni;
            int j = tIdx/ni;
            
            if (boundaryMask(i,j) == 3)
            {
                color[j*ni + i] = Vec3uc(0);
            }
            else
            {
                float value = rho(i,j) - temperature(i,j);
                value = (value + 1.f)*0.5f;
                if (do_tonemapping)
                {
                    value = max(value, 0.f);
                    value = sqrt(value)*(1.f+sqrt(value)/pow(2.2f,2.f))/(1.f+sqrt(value));
                }
                color[j*ni + i] = cBar.toRGB(1.f-value);
            }
        });
        wrtieBMPuc3(filename, ni, nj, (unsigned char*)(&(color[0])));
    }
    else
        writeBMP(filename, ni, nj, rho.a.data);
}

void CovectorSolver2D::outputCovectorField(std::string folder, std::string file, int i)
{
    boost::filesystem::create_directories(folder);
    std::string filestr;
    filestr = folder + file + std::string("_\%04d.bmp");
    char filename[1024];
    sprintf(filename, filestr.c_str(), i);
    Array2f cell_centered_cv_u;
    cell_centered_cv_u.resize(ni+1,nj+1,0);
    Array2f cell_centered_cv_v;
    cell_centered_cv_v.resize(ni+1,nj+1,0);
    tbb::parallel_for((int)0, (ni+1)*(nj+1), 1, [&](int tIdx) {
        int i = tIdx%(ni+1);
        int j = tIdx/(ni+1);
        if(i>0&&i<ni&&j>0&&j<nj)
        {
            cell_centered_cv_u(i,j) = (cv_u(i,j) + cv_u(i+1,j))/2.0f;
            cell_centered_cv_v(i,j) = (cv_v(i,j) + cv_v(i,j+1))/2.0f;
        }
    });  
    Array2f color_field;
    color_field.resize(ni*3,nj,0);
    tbb::parallel_for((int)0, ni*nj, 1, [&](int tIdx) {
        int i = tIdx%ni;
        int j = tIdx/ni;
        float angle = atan2(cell_centered_cv_v(i,j),cell_centered_cv_u(i,j));
        float r = mag(Vec2f(cell_centered_cv_u(i,j),cell_centered_cv_v(i,j)));
        float hue = angle/(2*M_PI);
        Vec3f RGB_values((cell_centered_cv_u(i,j)+1)/2.f, (cell_centered_cv_v(i,j)+1)/2.f, rho(i,j)+0.1);
        color_field(3*i+0,j) = RGB_values[0];
        color_field(3*i+1,j) = RGB_values[1];
        color_field(3*i+2,j) = RGB_values[2];
    });
    writeBMPColorVector(filename, ni, nj, color_field.a.data); 
}

void CovectorSolver2D::outputVortVisualized(std::string folder, std::string file, int i)
{
    boost::filesystem::create_directories(folder);
    std::string filestr;
    filestr = folder + file + std::string("\%04d.bmp");
    char filename[1024];
    sprintf(filename, filestr.c_str(), i);
    std::vector<Vec3uc> color;
    color.resize(ni*nj);
    cBar = color_bar(0,0);
    tbb::parallel_for((int)0, (int)(ni*nj), 1, [&](int tIdx)
    {
        int i = tIdx%ni;
        int j = tIdx/ni;

        float vort = 0.25*(curl(i,j)+curl(i+1,j)+curl(i,j+1)+curl(i+1,j+1));
        color[j*ni + i] = cBar.toRGB(fabs(vort), 10);
    });
    wrtieBMPuc3(filename, ni, nj, (unsigned char*)(&(color[0])));
}

void CovectorSolver2D::outputVellVisualized(std::string folder, std::string file, int i, bool do_y_comp)
{
    boost::filesystem::create_directories(folder);
    std::string filestr;
    filestr = folder + file + "VELL" + (do_y_comp? "_y_" : "_x_") + std::string("\%04d.bmp");
    char filename[1024];
    sprintf(filename, filestr.c_str(), i);
    std::vector<Vec3uc> color;
    color.resize(ni*nj);
    cBar = color_bar(0,0);
    tbb::parallel_for((int)0, (int)(ni*nj), 1, [&](int tIdx)
    {
        int i = tIdx%ni;
        int j = tIdx/ni;

        float vel = 0;
        if (do_y_comp)
            vel = fabs(v(i,j)+v(i+1,j))*0.5;
        else
            vel = fabs(u(i,j)+u(i+1,j))*0.5;

        color[j*ni + i] = cBar.toRGB(vel, karman_velocity_value);
    });
    wrtieBMPuc3(filename, ni, nj, (unsigned char*)(&(color[0])));
}

void CovectorSolver2D::outputLevelset(std::string sdfFilename, int i)
{
    boost::filesystem::create_directories(sdfFilename);
    std::ofstream foutU;
    std::string old_string = std::to_string(i);
    std::string new_string = std::string(4 - old_string.length(), '0') + old_string;
    std::string filenameU = sdfFilename + std::string("levelset_") + new_string + std::string(".txt");
    foutU.open(filenameU);
    for (int i = 0; i<ni; i++)
    {
        for (int j = 0; j<nj; j++)
        {
            foutU << rho(i,j) << " ";
        }
        if (i != ni-1) foutU << std::endl;
    }
    foutU.close();
}

void CovectorSolver2D::outputEnergy(std::string filename, float curr_time)
{
    boost::filesystem::create_directories(filename);
    std::ofstream foutU;
    std::string filenameU = filename + std::string("energy") + std::string(".txt");
    foutU.open(filenameU, curr_time == 0.f ? std::ios_base::out : std::ios_base::app);
    float energy = 0.0f;
    for (int i = 0; i<ni+1; i++)
    {
        for (int j = 0; j<nj; j++)
        {
            energy += u(i,j) *u(i,j);
        }
    }
    for (int i = 0; i<ni; i++)
    {
        for (int j = 0; j<nj+1; j++)
        {
            energy += v(i,j) *v(i,j);
        }
    }
    energy *= 0.5f * h*h;
    std::cout << "Energy = " << energy << std::endl;

    foutU << energy << " " << curr_time << std::endl;
    foutU.close();
}

void CovectorSolver2D::outputError(std::string filename, float curr_time)
{
    boost::filesystem::create_directories(filename);
    std::ofstream foutU;
    std::string filenameU = filename + std::string("RMS_error") + std::string(".txt");
    foutU.open(filenameU, curr_time == 0.f ? std::ios_base::out : std::ios_base::app);
    float error = 0.0f;
    float max_error = 0.f;
    int i_max, j_max;
    float u_saveIt, v_saveIt;
    for (int i = 0; i<ni+1; i++)
    {
        for (int j = 0; j<nj; j++)
        {
            Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.0, 0.5);
            float curr_error = u(i,j) - sin(pos[0])*cos(pos[1]); 
            error += curr_error*curr_error;
            if (max_error < curr_error*curr_error)
            {
                max_error = curr_error*curr_error;
                i_max = i; j_max = j;
                u_saveIt = u(i,j);
            }
        }
    }
    std::cout << "max error X = " << max_error << " at i j: " << i_max << " " << j_max << " u: " << u_saveIt << std::endl;
    max_error = 0.f;
    for (int i = 0; i<ni; i++)
    {
        for (int j = 0; j<nj+1; j++)
        {
            Vec2f pos = h * Vec2f(i, j) + h * Vec2f(0.5, 0.0);
            float curr_error = v(i,j) + cos(pos[0])*sin(pos[1]); 
            error += curr_error*curr_error;
            if (max_error < curr_error*curr_error)
            {
                max_error = curr_error*curr_error;
                i_max = i; j_max = j;
                v_saveIt = v(i,j);
            }
        }
    }
    std::cout << "max error Y = " << max_error << " at i j: " << i_max << " " << j_max << " v: " << v_saveIt <<  std::endl;
    error /= ((ni+1)*nj) + (ni*(nj+1));
    error = sqrt(error);

    std::cout << "RMS Error = " << error << std::endl;

    foutU << error << " " << curr_time << std::endl;
    foutU.close();
}

void CovectorSolver2D::outputVorticityIntegral(std::string filename, float curr_time)
{
    boost::filesystem::create_directories(filename);
    std::ofstream foutU, foutU2;
    std::string filenameU = filename + std::string("vorticityIntegral") + std::string(".txt");
    std::string filenameU2 = filename + std::string("absVorticityIntegral") + std::string(".txt");
    foutU.open(filenameU, curr_time == 0.f ? std::ios_base::out : std::ios_base::app);
    foutU2.open(filenameU2, curr_time == 0.f ? std::ios_base::out : std::ios_base::app);
    float vorticityIntegral = 0.0f;
    float absVorticityIntegral = 0.0f;
    for (int i = 0; i<ni+1; i++)
    {
        for (int j = 0; j<nj+1; j++)
        {
            vorticityIntegral += curl(i,j);
            absVorticityIntegral += abs(curl(i,j));
        }
    }
    vorticityIntegral *= h*h;
    absVorticityIntegral *= h*h;
    std::cout << "Vorticity Integral = " << vorticityIntegral << std::endl;
    std::cout << "Abs Vorticity Integral = " << absVorticityIntegral << std::endl;

    foutU << vorticityIntegral << " " << curr_time << std::endl;
    foutU.close();
    foutU2 << absVorticityIntegral << " " << curr_time << std::endl;
    foutU2.close();
}

void CovectorSolver2D::outputErrorVisualized(std::string folder, std::string file, int i)
{
    boost::filesystem::create_directories(folder);
    std::string filestr;
    filestr = folder + file + std::string("\%04d.bmp");
    char filename[1024];
    sprintf(filename, filestr.c_str(), i);
    std::vector<Vec3uc> color;
    color.resize(ni*nj);
    float max_error = 0.0f;
    tbb::parallel_for((int)0, (int)(ni*nj), 1, [&](int tIdx)
    {
        int i = tIdx%ni;
        int j = tIdx/ni;
        Vec2f p1 = h * Vec2f(i, j) + h * Vec2f(0.0, 0.5);
        float e1 = abs(u(i,j) - sin(p1[0])*cos(p1[1]));
        Vec2f p2 = h * Vec2f(i+1, j) + h * Vec2f(0.0, 0.5);
        float e2 = abs(u(i+1,j) - sin(p2[0])*cos(p2[1]));

        Vec2f q1 = h * Vec2f(i, j) + h * Vec2f(0.5, 0.0);
        float E1 = abs(v(i,j) + cos(q1[0])*sin(q1[1]));
        Vec2f q2 = h * Vec2f(i, j+1) + h * Vec2f(0.5, 0.0);
        float E2 = abs(v(i,j+1) + cos(q2[0])*sin(q2[1]));
        max_error = max(max_error, max(e1, max(e2, max(E1, E2))));
    });
    cBar = color_bar(0,0);
    tbb::parallel_for((int)0, (int)(ni*nj), 1, [&](int tIdx)
    {
        int i = tIdx%ni;
        int j = tIdx/ni;
        Vec2f p1 = h * Vec2f(i, j) + h * Vec2f(0.0, 0.5);
        float e1 = abs(u(i,j) - sin(p1[0])*cos(p1[1]));
        Vec2f p2 = h * Vec2f(i+1, j) + h * Vec2f(0.0, 0.5);
        float e2 = abs(u(i+1,j) - sin(p2[0])*cos(p2[1]));

        Vec2f q1 = h * Vec2f(i, j) + h * Vec2f(0.5, 0.0);
        float E1 = abs(v(i,j) + cos(q1[0])*sin(q1[1]));
        Vec2f q2 = h * Vec2f(i, j+1) + h * Vec2f(0.5, 0.0);
        float E2 = abs(v(i,j+1) + cos(q2[0])*sin(q2[1]));

        float avg_error = 0.25*(e1+e2+E1+E2)*100.0f;
        color[j*ni + i] = cBar.toRGB(avg_error);
    });
    wrtieBMPuc3(filename, ni, nj, (unsigned char*)(&(color[0])));
}
