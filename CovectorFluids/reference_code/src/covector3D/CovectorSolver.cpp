#include "CovectorSolver.h"

CovectorSolver::CovectorSolver(uint nx, uint ny, uint nz, float L, float vis_coeff, float blend_coeff, Scheme myscheme, gpuMapper *mymapper)
{
    _nx = nx;
    _ny = ny;
    _nz = nz;
    _h = L/nx;
    max_v = 0.f;
    viscosity = vis_coeff;
    sim_scheme = myscheme;

    _un.init(_nx+1, _ny, _nz, _h, 0.5f, 0.f, 0.f);
    _vn.init(_nx, _ny+1, _nz, _h, 0.f, 0.5f, 0.f);
    _wn.init(_nx, _ny, _nz+1, _h, 0.f, 0.f, 0.5f);
    _utemp.init(_nx+1, _ny, _nz, _h, 0.5f, 0.f, 0.f);
    _vtemp.init(_nx, _ny+1, _nz, _h, 0.f, 0.5f, 0.f);
    _wtemp.init(_nx, _ny, _nz+1, _h, 0.f, 0.f, 0.5f);
    _uinit.init(_nx+1, _ny, _nz, _h, 0.5f, 0.f, 0.f);
    _vinit.init(_nx, _ny+1, _nz, _h, 0.f, 0.5f, 0.f);
    _winit.init(_nx, _ny, _nz+1, _h, 0.f, 0.f, 0.5f);
    _uprev.init(_nx+1, _ny, _nz, _h, 0.5f, 0.f, 0.f);
    _vprev.init(_nx, _ny+1, _nz, _h, 0.f, 0.5f, 0.f);
    _wprev.init(_nx, _ny, _nz+1, _h, 0.f, 0.f, 0.5f);
    _duproj.init(_nx+1, _ny, _nz, _h, 0.5f, 0.f, 0.f);
    _dvproj.init(_nx, _ny+1, _nz, _h, 0.f, 0.5f, 0.f);
    _dwproj.init(_nx, _ny, _nz+1, _h, 0.f, 0.f, 0.5f);
    _duextern.init(_nx+1, _ny, _nz, _h, 0.5f, 0.f, 0.f);
    _dvextern.init(_nx, _ny+1, _nz, _h, 0.f, 0.5f, 0.f);
    _dwextern.init(_nx, _ny, _nz+1, _h, 0.f, 0.f, 0.5f);

    _rho.init(_nx, _ny, _nz, _h, 0.f, 0.f, 0.f);
    _rhotemp.init(_nx, _ny, _nz, _h, 0.f, 0.f, 0.f);
    _rhoinit.init(_nx, _ny, _nz, _h, 0.f, 0.f, 0.f);
    _rhoprev.init(_nx, _ny, _nz, _h, 0.f, 0.f, 0.f);
    _drhoextern.init(_nx, _ny, _nz, _h, 0.f, 0.f, 0.f);

    _T.init(_nx, _ny, _nz, _h, 0.f, 0.f, 0.f);
    _Ttemp.init(_nx, _ny, _nz, _h, 0.f, 0.f, 0.f);
    _Tinit.init(_nx, _ny, _nz, _h, 0.f, 0.f, 0.f);
    _Tprev.init(_nx, _ny, _nz, _h, 0.f, 0.f, 0.f);
    _dTextern.init(_nx, _ny, _nz, _h, 0.f, 0.f, 0.f);

    _usolid.init(_nx+1, _ny, _nz, _h, 0.5f, 0.f, 0.f);
    _vsolid.init(_nx, _ny+1, _nz, _h, 0.f, 0.5f, 0.f);
    _wsolid.init(_nx, _ny, _nz+1, _h, 0.f, 0.f, 0.5f);
    u_valid.resize(_nx+1,_ny,_nz);
    v_valid.resize(_nx,_ny+1,_nz);
    w_valid.resize(_nx,_ny,_nz+1);



    _b_desc.init(_nx,_ny,_nz);
    // initialize BIMOCQ advector
    VelocityAdvector.init(_nx, _ny, _nz, _h, blend_coeff, mymapper);
    ScalarAdvector.init(_nx, _ny, _nz, _h, blend_coeff, mymapper);
    gpuSolver = mymapper;
}

void CovectorSolver::advance(int framenum, float dt)
{
    switch (sim_scheme)
    {
        case SEMILAG:
            advanceSemilag(framenum, dt);
            break;
        case REFLECTION:
            advanceReflection(framenum, dt, true);
            break;
        case SCPF:
            advanceSCPF(framenum, dt);
            break;
        case MACCORMACK:
            advanceMacCormack(framenum, dt);
            break;
        case MAC_REFLECTION:
            advanceReflection(framenum, dt);
            break;
        case BIMOCQ:
            advanceBimocq(framenum, dt);
            break;
        case COVECTOR:
        case COVECTOR_BIMOCQ:
            advanceCovector(framenum, dt);
            break;
        default:
            break;
    }
}

void CovectorSolver::fullAdvect(int framenum, float dt, bool do_full_advect)
{
    float cfldt = getCFL();
    //if (framenum == 0) max_v = _h;
    cout << YELLOW << "[ CFL number is: " << max_v * dt / _h << " ] " << RESET << endl;

    if (!do_full_advect)
    {
        VelocityAdvector.moveToTemp(1);
        _drhoextern.copy(_rho); _dTextern.copy(_T);
    }
    if (do_full_advect)
    {
        VelocityAdvector.updateMapping(_un, _vn, _wn, cfldt, dt, do_dmc);
        if (!do_vel_advection_only)
        {
            ScalarAdvector.updateMapping(_un, _vn, _wn, cfldt, dt, do_dmc);
        }
        cout << "[ One-Form Update Mapping Done! dt = " << dt << " ]" << endl;

        if (!sim_boundary.empty())
        {
            semilagAdvect(cfldt, dt);
            cout << "[ Semilag Advect Fields Done! ]" << endl;
        }

        VelocityAdvector.advectVelocityCovector(_un, _vn, _wn, _uinit, _vinit, _winit, _uprev, _vprev, _wprev, do_EC, do_EC_with_clamp, do_antialiasing, cfldt, dt, delayed_reinit_num == 1 && do_true_mid_vel_covector);
        if (!do_vel_advection_only)
        {
            ScalarAdvector.advectFieldCovector(_rho, _rhoinit, _rhoprev, do_EC, false && do_antialiasing);
            ScalarAdvector.advectFieldCovector(_T, _Tinit, _Tprev, do_EC, false && do_antialiasing);
        }
        cout << "[ One-Form Advect Fields Done! ]" << endl;
    }
    else
    {
        VelocityAdvector.updateMapping(_un, _vn, _wn, cfldt, dt, do_dmc);
        cout << "[ One-Form Update Mapping Done! dt = " << dt << " ]" << endl;

        if (!sim_boundary.empty())
        {
            semilagAdvect(cfldt, dt, true);
            cout << "[ Semilag Advect Fields Done! ]" << endl;
        }

        VelocityAdvector.advectVelocityCovector(_un, _vn, _wn, _uinit, _vinit, _winit, _uprev, _vprev, _wprev, do_EC, do_EC_with_clamp, do_antialiasing, cfldt, dt, delayed_reinit_num == 1 && do_true_mid_vel_covector);
        cout << "[ One-Form Advect Fields Done! ]" << endl;
    }

    blendBoundary(_un, _utemp);
    blendBoundary(_vn, _vtemp);
    blendBoundary(_wn, _wtemp);
    if (do_full_advect && !do_vel_advection_only)
    {
        blendBoundary(_rho, _rhotemp);
        blendBoundary(_T, _Ttemp);
    }
    cout << "[ Blend Boundary Fields Done! ]" << endl;


    if (do_full_advect)
    {
        // save current fields to calculate change
        _utemp.copy(_un);
        _vtemp.copy(_vn);
        _wtemp.copy(_wn);
        if (!do_vel_advection_only)
        {
            _rhotemp.copy(_rho);
            _Ttemp.copy(_T);
        }
    }

    if (!do_vel_advection_only)
    {
        clearBoundary(_rho);
        clearBoundary(_T);
    }
    emitSmoke(framenum, dt);
    addBuoyancy(_un, _vn, _wn, dt);

    // add viscosity
    if (viscosity)
    {
        diffuse_field(dt, viscosity, _un);
        diffuse_field(dt, viscosity, _vn);
        diffuse_field(dt, viscosity, _wn);
    }

    if (do_full_advect)
    {
        // calculate velocity change due to external forces(e.g. buoyancy)
        _duextern.copy(_un); _duextern -= _utemp;
        _dvextern.copy(_vn); _dvextern -= _vtemp;
        _dwextern.copy(_wn); _dwextern -= _wtemp;
        if (!do_vel_advection_only)
        {
            _drhoextern.copy(_rho); _drhoextern -= _rhotemp;
            _dTextern.copy(_T); _dTextern -= _Ttemp;
        }
    }
    
    projection();


    if (do_full_advect)
    {
        // accumuate buffer changes
        VelocityAdvector.accumulateVelocityCovector(_uinit, _vinit, _winit, _duextern, _dvextern, _dwextern, _un, _vn, _wn, 1.f, do_antialiasing, cfldt, dt, delayed_reinit_num == 1 && do_true_mid_vel_covector);
        if (!do_vel_advection_only)
        {
            ScalarAdvector.accumulateField(_rhoinit, _drhoextern, false && do_antialiasing);
            ScalarAdvector.accumulateField(_Tinit, _dTextern, false && do_antialiasing);
        }
    }

    if (!do_full_advect)
    {
        _rho.copy(_drhoextern); _T.copy(_dTextern);
        VelocityAdvector.moveToTemp(0);
    }
}

void CovectorSolver::advanceCovector(int framenum, float dt)
{
    bool velReinit = false;
    bool scalarReinit = false;

    // for second-order (non-linear) one-form advection scheme
    if (do_2nd_order)
    {
        // only velocity fields are advected by 0.5*dt
        fullAdvect(framenum, 0.5 * dt, false);
    }

    fullAdvect(framenum, dt, true);

    if (delayed_reinit_num > 1)
    {
        float VelocityDistortion = VelocityAdvector.estimateDistortion(_b_desc) / (max_v * dt);
        cout << "[ Velocity Distortion is " << VelocityDistortion << " ]" << endl;
        if (VelocityDistortion > 0.5f || framenum - vel_lastReinit >= delayed_reinit_num)
        {
            velReinit = true;
            vel_lastReinit = framenum;
        }
        if (!do_vel_advection_only)
        {
            float ScalarDistortion = ScalarAdvector.estimateDistortion(_b_desc) / (max_v * dt);
            cout << "[ Scalar Distortion is " << ScalarDistortion << " ]" << endl;
            if (ScalarDistortion > 2.5f || framenum - scalar_lastReinit >= delayed_reinit_num)
            {
                scalarReinit = true;
                scalar_lastReinit = framenum;
            }
        }
    }
    else
    {
        velReinit = true;
        scalarReinit = true;
        vel_lastReinit = framenum;
        scalar_lastReinit = framenum;
    }

    if (velReinit)
    {
        VelocityAdvector.reinitializeMapping();
        velocityReinitialize();
        cout << RED << "[ One-Form Velocity Re-initialize, total reinitialize count: " << VelocityAdvector.total_reinit_count << " ]" << RESET << endl;
    }
    if (scalarReinit && !do_vel_advection_only)
    {
        ScalarAdvector.reinitializeMapping();
        scalarReinitialize();
        cout << RED << "[ One-Form Scalar Re-initialize, total reinitialize count: " << ScalarAdvector.total_reinit_count << " ]" << RESET << endl;
    }
}

void CovectorSolver::advanceSCPF(int framenum, float dt)
{
    float cfldt = getCFL();
    //if (framenum == 0) max_v = _h;
    cout << YELLOW << "[ CFL number is: " << max_v * dt / _h << " ] " << RESET << endl;

    VelocityAdvector.updateMapping(_un, _vn, _wn, cfldt, dt, 0);
    if (!do_vel_advection_only)
    {
        ScalarAdvector.updateMapping(_un, _vn, _wn, cfldt, dt, 0);
    }
    cout << "[ SCPF Update Mapping Done! dt = " << dt << " ]" << endl;

    if (!sim_boundary.empty())
    {
        semilagAdvect(cfldt, dt);
        cout << "[ Semilag Advect Fields Done! ]" << endl;
    }

    VelocityAdvector.advectVelocityCovector(_un, _vn, _wn, _uinit, _vinit, _winit, _uprev, _vprev, _wprev, false, false, do_antialiasing, cfldt, dt, false, true);
    if (!do_vel_advection_only)
    {
        ScalarAdvector.advectFieldCovector(_rho, _rhoinit, _rhoprev, false, false && do_antialiasing);
        ScalarAdvector.advectFieldCovector(_T, _Tinit, _Tprev, false, false && do_antialiasing);
    }
    cout << "[ SCPF Advect Fields Done! ]" << endl;

    blendBoundary(_un, _utemp);
    blendBoundary(_vn, _vtemp);
    blendBoundary(_wn, _wtemp);
    if (!do_vel_advection_only)
    {
        blendBoundary(_rho, _rhotemp);
        blendBoundary(_T, _Ttemp);
    }
    cout << "[ Blend Boundary Fields Done! ]" << endl;

    if (!do_vel_advection_only)
    {
        clearBoundary(_rho);
        clearBoundary(_T);
    }
    emitSmoke(framenum, dt);
    addBuoyancy(_un, _vn, _wn, dt);

    // add viscosity
    if (viscosity)
    {
        diffuse_field(dt, viscosity, _un);
        diffuse_field(dt, viscosity, _vn);
        diffuse_field(dt, viscosity, _wn);
    }

    projection();

    VelocityAdvector.reinitializeMapping();
    velocityReinitialize();
    cout << RED << "[ One-Form Velocity Re-initialize, total reinitialize count: " << VelocityAdvector.total_reinit_count << " ]" << RESET << endl;

    ScalarAdvector.reinitializeMapping();
    scalarReinitialize();
    cout << RED << "[ One-Form Scalar Re-initialize, total reinitialize count: " << ScalarAdvector.total_reinit_count << " ]" << RESET << endl;
}

void CovectorSolver::advanceBimocq(int framenum, float dt)
{
    float proj_coeff = 2.f;
    bool velReinit = false;
    bool scalarReinit = false;
    float cfldt = getCFL();
    //if (framenum == 0) max_v = _h;
    cout << YELLOW << "[ CFL number is: " << max_v*dt/_h << " ] " << RESET << endl;

    VelocityAdvector.updateMapping(_un, _vn, _wn, cfldt, dt, do_dmc);
    if (!do_vel_advection_only)
    {
        ScalarAdvector.updateMapping(_un, _vn, _wn, cfldt, dt, do_dmc);
    }
    cout << "[ Update Mapping Done! ]" << endl;

    semilagAdvect(cfldt, dt);
    cout << "[ Semilag Advect Fields Done! ]" << endl;

    VelocityAdvector.advectVelocity(_un, _vn, _wn, _uinit, _vinit, _winit, _uprev, _vprev, _wprev, do_EC, do_EC_with_clamp, do_antialiasing);
    if (!do_vel_advection_only)
    {
        ScalarAdvector.advectField(_rho, _rhoinit, _rhoprev, do_EC, false && do_antialiasing);
        ScalarAdvector.advectField(_T, _Tinit, _Tprev, do_EC, false && do_antialiasing);
    }
    cout << "[ Bimocq Advect Fields Done! ]" << endl;

    blendBoundary(_un, _utemp);
    blendBoundary(_vn, _vtemp);
    blendBoundary(_wn, _wtemp);
    if (!do_vel_advection_only)
    {
        blendBoundary(_rho, _rhotemp);
        blendBoundary(_T, _Ttemp);
    }
    cout << "[ Blend Boundary Fields Done! ]" << endl;


    // save current fields to calculate change
    _utemp.copy(_un);
    _vtemp.copy(_vn);
    _wtemp.copy(_wn);
    if (!do_vel_advection_only)
    {
        _rhotemp.copy(_rho);
        _Ttemp.copy(_T);
    }

    if (!do_vel_advection_only)
    {
        clearBoundary(_rho);
        clearBoundary(_T);
    }
    emitSmoke(framenum, dt);
    addBuoyancy(_un, _vn, _wn, dt);

    // add viscosity
    if (viscosity)
    {
        diffuse_field(dt, viscosity, _un);
        diffuse_field(dt, viscosity, _vn);
        diffuse_field(dt, viscosity, _wn);
    }

    // calculate velocity change due to external forces(e.g. buoyancy)
    _duextern.copy(_un); _duextern -= _utemp;
    _dvextern.copy(_vn); _dvextern -= _vtemp;
    _dwextern.copy(_wn); _dwextern -= _wtemp;
    if (!do_vel_advection_only)
    {
        _drhoextern.copy(_rho); _drhoextern -= _rhotemp;
        _dTextern.copy(_T); _dTextern -= _Ttemp;
    }

    _utemp.copy(_un);
    _vtemp.copy(_vn);
    _wtemp.copy(_wn);
    projection();
    // calculate velocity change due to pressure projection
    _duproj.copy(_un); _duproj -= _utemp;
    _dvproj.copy(_vn); _dvproj -= _vtemp;
    _dwproj.copy(_wn); _dwproj -= _wtemp;

    float VelocityDistortion = VelocityAdvector.estimateDistortion(_b_desc) / (max_v * dt);
    cout << "[ Velocity Distortion is " << VelocityDistortion << " ]" << endl;
    if (VelocityDistortion > 1.f || framenum - vel_lastReinit >= delayed_reinit_num)
    {
        velReinit = true;
        vel_lastReinit = framenum;
        proj_coeff = 1.f;
    }
    if (!do_vel_advection_only)
    {
        float ScalarDistortion = ScalarAdvector.estimateDistortion(_b_desc) / (max_v * dt);
        cout << "[ Scalar Distortion is " << ScalarDistortion << " ]" << endl;
        if (ScalarDistortion > 5.f || framenum - scalar_lastReinit >= delayed_reinit_num)
        {
            scalarReinit = true;
            scalar_lastReinit = framenum;
        }
    }

    // accumuate buffer changes
    VelocityAdvector.accumulateVelocity(_uinit, _vinit, _winit, _duextern, _dvextern, _dwextern, 1.f, do_antialiasing);
    VelocityAdvector.accumulateVelocity(_uinit, _vinit, _winit, _duproj, _dvproj, _dwproj, proj_coeff, do_antialiasing);

    if (!do_vel_advection_only)
    {
        ScalarAdvector.accumulateField(_rhoinit, _drhoextern, false && do_antialiasing);
        ScalarAdvector.accumulateField(_Tinit, _dTextern, false && do_antialiasing);
    }

    cout << "[ Accumulate Fields Done! ]" << endl;
    if (velReinit)
    {
        VelocityAdvector.reinitializeMapping();
        velocityReinitialize();
        VelocityAdvector.accumulateVelocity(_uinit, _vinit, _winit, _duproj, _dvproj, _dwproj, 1.f, do_antialiasing);
        cout << RED << "[ Bimocq Velocity Re-initialize, total reinitialize count: " << VelocityAdvector.total_reinit_count << " ]" << RESET << endl;
    }
    if (scalarReinit && !do_vel_advection_only)
    {
        ScalarAdvector.reinitializeMapping();
        scalarReinitialize();
        cout << RED << "[ Bimocq Scalar Re-initialize, total reinitialize count: " << ScalarAdvector.total_reinit_count << " ]" << RESET << endl;
    }
}

void CovectorSolver::advanceSemilag(int framenum, float dt)
{
    // first copy velocity buffers to GPU.u, GPU.v, GPU.w
    gpuSolver->copyHostToDevice(_un, gpuSolver->u_host, gpuSolver->u, (_nx+1)*_ny*_nz*sizeof(float));
    gpuSolver->copyHostToDevice(_vn, gpuSolver->v_host, gpuSolver->v, _nx*(_ny+1)*_nz*sizeof(float));
    gpuSolver->copyHostToDevice(_wn, gpuSolver->w_host, gpuSolver->w, _nx*_ny*(_nz+1)*sizeof(float));
    // semi-lagrangian advect any other fluid fields
    // reuse gpu.du, gpu.dv to save GPU buffer
    // copy field to gpu.dv for semi-lagrangian advection
    // advect density
    float cfldt = getCFL();
    cout << YELLOW << "[ CFL number is: " << max_v*dt/_h << " ] " << RESET << endl;
    
    if (!do_vel_advection_only)
    {
        gpuSolver->copyHostToDevice(_rho, gpuSolver->x_host, gpuSolver->dfield, _nx * _ny * _nz * sizeof(float));
        gpuSolver->semilagAdvectField(!(false && do_antialiasing), cfldt, -dt);
        gpuSolver->copyDeviceToHost(_rho, gpuSolver->x_host, gpuSolver->field);
        cout << "[ Semilag Advect Density Done! ]" << endl;

        // advect Temperature
        gpuSolver->copyHostToDevice(_T, gpuSolver->x_host, gpuSolver->dfield, _nx * _ny * _nz * sizeof(float));
        gpuSolver->semilagAdvectField(!(false && do_antialiasing), cfldt, -dt);
        gpuSolver->copyDeviceToHost(_T, gpuSolver->x_host, gpuSolver->field);
        cout << "[ Semilag Advect Temperature Done! ]" << endl;
    }

    // semi-lagrangian advected velocity will be stored in gpu.du, gpu.dv, gpu.dw
    // negate dt for tracing back

    cudaMemcpy(gpuSolver->u_src, gpuSolver->u, (_nx+1)*_ny*_nz*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpuSolver->v_src, gpuSolver->v, _nx*(_ny+1)*_nz*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpuSolver->w_src, gpuSolver->w, _nx*_ny*(_nz+1)*sizeof(float), cudaMemcpyDeviceToDevice);
    gpuSolver->semilagAdvectVelocity(!do_antialiasing, cfldt, -dt);
    gpuSolver->copyDeviceToHost(_un, gpuSolver->u_host, gpuSolver->du);
    gpuSolver->copyDeviceToHost(_vn, gpuSolver->v_host, gpuSolver->dv);
    gpuSolver->copyDeviceToHost(_wn, gpuSolver->w_host, gpuSolver->dw);
    cout << "[ Semilag Advect Velocity Done! ]" << endl;

    if (!do_vel_advection_only)
    {
        clearBoundary(_rho);
        clearBoundary(_T);
    }
    emitSmoke(framenum, dt);
    addBuoyancy(_un, _vn, _wn, dt);

    // add viscosity
    if (viscosity)
    {
        diffuse_field(dt, viscosity, _un);
        diffuse_field(dt, viscosity, _vn);
        diffuse_field(dt, viscosity, _wn);
    }

    projection();
}

void CovectorSolver::advanceMacCormack(int framenum, float dt)
{
    gpuSolver->copyHostToDevice(_un, gpuSolver->u_host, gpuSolver->u, (_nx+1)*_ny*_nz*sizeof(float));
    gpuSolver->copyHostToDevice(_vn, gpuSolver->v_host, gpuSolver->v, _nx*(_ny+1)*_nz*sizeof(float));
    gpuSolver->copyHostToDevice(_wn, gpuSolver->w_host, gpuSolver->w, _nx*_ny*(_nz+1)*sizeof(float));
    // MacCormack advect any other fluid fields
    // reuse gpu.du, gpu.dv to save GPU buffer
    // copy field to gpu.dv for advection
    // advect density
    float cfldt = getCFL();
    cout << YELLOW << "[ CFL number is: " << max_v*dt/_h << " ] " << RESET << endl;

    if (!do_vel_advection_only)
    {
        gpuSolver->copyHostToDevice(_rho, gpuSolver->x_host, gpuSolver->dfield, _nx * _ny * _nz * sizeof(float));
        gpuSolver->semilagAdvectField(!(false && do_antialiasing), cfldt, -dt);
        cudaMemcpy(gpuSolver->dfield, gpuSolver->field, sizeof(float) * _nx * _ny * _nz, cudaMemcpyDeviceToDevice);
        cudaMemcpy(gpuSolver->field_src, gpuSolver->field, sizeof(float) * _nx * _ny * _nz, cudaMemcpyDeviceToDevice);
        gpuSolver->semilagAdvectField(!(false && do_antialiasing), cfldt, dt);
        gpuSolver->add(gpuSolver->dfield, gpuSolver->field, -0.5f, _nx * _ny * _nz);
        gpuSolver->copyHostToDevice(_rho, gpuSolver->x_host, gpuSolver->field, _nx * _ny * _nz * sizeof(float));
        gpuSolver->add(gpuSolver->dfield, gpuSolver->field, 0.5f, _nx * _ny * _nz);
        // clamp extrema, clamped new density will be in GPU.dfield
        gpuSolver->clampExtremaField();
        // update rho
        gpuSolver->copyDeviceToHost(_rho, gpuSolver->x_host, gpuSolver->dfield);
        cout << "[ MacCormack Advect Density Done! ]" << endl;

        // advect Temperature
        gpuSolver->copyHostToDevice(_T, gpuSolver->x_host, gpuSolver->dfield, _nx * _ny * _nz * sizeof(float));
        gpuSolver->semilagAdvectField(!(false && do_antialiasing), cfldt, -dt);
        cudaMemcpy(gpuSolver->dfield, gpuSolver->field, sizeof(float) * _nx * _ny * _nz, cudaMemcpyDeviceToDevice);
        cudaMemcpy(gpuSolver->field_src, gpuSolver->field, sizeof(float) * _nx * _ny * _nz, cudaMemcpyDeviceToDevice);
        gpuSolver->semilagAdvectField(!(false && do_antialiasing), cfldt, dt);
        gpuSolver->add(gpuSolver->dfield, gpuSolver->field, -0.5f, _nx * _ny * _nz);
        gpuSolver->copyHostToDevice(_T, gpuSolver->x_host, gpuSolver->field, _nx * _ny * _nz * sizeof(float));
        gpuSolver->add(gpuSolver->dfield, gpuSolver->field, 0.5f, _nx * _ny * _nz);
        // clamp extrema, clamped new density will be in GPU.dfield
        gpuSolver->clampExtremaField();
        // update temperature
        gpuSolver->copyDeviceToHost(_T, gpuSolver->x_host, gpuSolver->dfield);
        cout << "[ MacCormack Advect Temperature Done! ]" << endl;
    }

    // semi-lagrangian advected velocity will be stored in gpu.du, gpu.dv, gpu.dw
    // negate dt for tracing back
    cudaMemcpy(gpuSolver->u_src, gpuSolver->u, (_nx+1)*_ny*_nz*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpuSolver->v_src, gpuSolver->v, _nx*(_ny+1)*_nz*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpuSolver->w_src, gpuSolver->w, _nx*_ny*(_nz+1)*sizeof(float), cudaMemcpyDeviceToDevice);
    gpuSolver->semilagAdvectVelocity(!do_antialiasing, cfldt, -dt);
    cudaMemcpy(gpuSolver->u_src, gpuSolver->du, (_nx+1)*_ny*_nz*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpuSolver->v_src, gpuSolver->dv, _nx*(_ny+1)*_nz*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpuSolver->w_src, gpuSolver->dw, _nx*_ny*(_nz+1)*sizeof(float), cudaMemcpyDeviceToDevice);
    gpuSolver->semilagAdvectVelocity(!do_antialiasing, cfldt, dt);
    cudaMemcpy(gpuSolver->u, gpuSolver->u_src, (_nx+1)*_ny*_nz*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpuSolver->v, gpuSolver->v_src, _nx*(_ny+1)*_nz*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpuSolver->w, gpuSolver->w_src, _nx*_ny*(_nz+1)*sizeof(float), cudaMemcpyDeviceToDevice);
    gpuSolver->add(gpuSolver->u, gpuSolver->du, -0.5f, (_nx+1)*_ny*_nz);
    gpuSolver->add(gpuSolver->v, gpuSolver->dv, -0.5f, _nx*(_ny+1)*_nz);
    gpuSolver->add(gpuSolver->w, gpuSolver->dw, -0.5f, _nx*_ny*(_nz+1));
    gpuSolver->copyHostToDevice(_un, gpuSolver->u_host, gpuSolver->du, (_nx+1)*_ny*_nz*sizeof(float));
    gpuSolver->copyHostToDevice(_vn, gpuSolver->v_host, gpuSolver->dv, _nx*(_ny+1)*_nz*sizeof(float));
    gpuSolver->copyHostToDevice(_wn, gpuSolver->w_host, gpuSolver->dw, _nx*_ny*(_nz+1)*sizeof(float));
    gpuSolver->add(gpuSolver->u, gpuSolver->du, 0.5f, (_nx+1)*_ny*_nz);
    gpuSolver->add(gpuSolver->v, gpuSolver->dv, 0.5f, _nx*(_ny+1)*_nz);
    gpuSolver->add(gpuSolver->w, gpuSolver->dw, 0.5f, _nx*_ny*(_nz+1));
    // clamp extrema, clamped new velocity will be in GPU.u_src, GPU.v_src, GPU.w_src
    if (do_EC_with_clamp)
        gpuSolver->clampExtremaVelocity();
    // copy velocity back to CPU buffers
    gpuSolver->copyDeviceToHost(_un, gpuSolver->u_host, gpuSolver->u);
    gpuSolver->copyDeviceToHost(_vn, gpuSolver->v_host, gpuSolver->v);
    gpuSolver->copyDeviceToHost(_wn, gpuSolver->w_host, gpuSolver->w);
    cout << "[ MacCormack Advect Velocity Done! ]" << endl;

    if (!do_vel_advection_only)
    {
        clearBoundary(_rho);
        clearBoundary(_T);
    }
    emitSmoke(framenum, dt);
    addBuoyancy(_un, _vn, _wn, dt);

    // add viscosity
    if (viscosity)
    {
        diffuse_field(dt, viscosity, _un);
        diffuse_field(dt, viscosity, _vn);
        diffuse_field(dt, viscosity, _wn);
    }

    projection();
}

void CovectorSolver::advanceReflection(int framenum, float dt, bool do_SF)
{
    gpuSolver->copyHostToDevice(_un, gpuSolver->u_host, gpuSolver->u, (_nx+1)*_ny*_nz*sizeof(float));
    gpuSolver->copyHostToDevice(_vn, gpuSolver->v_host, gpuSolver->v, _nx*(_ny+1)*_nz*sizeof(float));
    gpuSolver->copyHostToDevice(_wn, gpuSolver->w_host, gpuSolver->w, _nx*_ny*(_nz+1)*sizeof(float));
    // Reflection advect any other fluid fields
    // reuse gpu.du, gpu.dv to save GPU buffer
    // copy field to gpu.dv for advection
    // advect density
    float cfldt = getCFL();
    cout << YELLOW << "[ CFL number is: " << max_v*dt/_h << " ] " << RESET << endl;

    if (!do_vel_advection_only)
    {
        gpuSolver->copyHostToDevice(_rho, gpuSolver->x_host, gpuSolver->dfield, _nx * _ny * _nz * sizeof(float));
        gpuSolver->semilagAdvectField(!(false && do_antialiasing), cfldt, -dt);
        if (!do_SF)
        {
            cudaMemcpy(gpuSolver->dfield, gpuSolver->field, sizeof(float) * _nx * _ny * _nz, cudaMemcpyDeviceToDevice);
            cudaMemcpy(gpuSolver->field_src, gpuSolver->field, sizeof(float) * _nx * _ny * _nz, cudaMemcpyDeviceToDevice);
            gpuSolver->semilagAdvectField(!(false && do_antialiasing), cfldt, dt);
            gpuSolver->add(gpuSolver->dfield, gpuSolver->field, -0.5f, _nx * _ny * _nz);
            gpuSolver->copyHostToDevice(_rho, gpuSolver->x_host, gpuSolver->field, _nx * _ny * _nz * sizeof(float));
            gpuSolver->add(gpuSolver->dfield, gpuSolver->field, 0.5f, _nx * _ny * _nz);
            // clamp extrema, clamped new density will be in GPU.dfield
            gpuSolver->clampExtremaField();
            // update rho
            gpuSolver->copyDeviceToHost(_rho, gpuSolver->x_host, gpuSolver->dfield);
        }
        else
        {
            // update rho
            gpuSolver->copyDeviceToHost(_rho, gpuSolver->x_host, gpuSolver->field);
        }
        cout << "[ Reflection Advect Density Done! ]" << endl;

        // advect Temperature
        gpuSolver->copyHostToDevice(_T, gpuSolver->x_host, gpuSolver->dfield, _nx * _ny * _nz * sizeof(float));
        gpuSolver->semilagAdvectField(!(false && do_antialiasing), cfldt, -dt);
        if (!do_SF)
        {
            cudaMemcpy(gpuSolver->dfield, gpuSolver->field, sizeof(float) * _nx * _ny * _nz, cudaMemcpyDeviceToDevice);
            cudaMemcpy(gpuSolver->field_src, gpuSolver->field, sizeof(float) * _nx * _ny * _nz, cudaMemcpyDeviceToDevice);
            gpuSolver->semilagAdvectField(!(false && do_antialiasing), cfldt, dt);
            gpuSolver->add(gpuSolver->dfield, gpuSolver->field, -0.5f, _nx * _ny * _nz);
            gpuSolver->copyHostToDevice(_T, gpuSolver->x_host, gpuSolver->field, _nx * _ny * _nz * sizeof(float));
            gpuSolver->add(gpuSolver->dfield, gpuSolver->field, 0.5f, _nx * _ny * _nz);
            // clamp extrema, clamped new density will be in GPU.dfield
            gpuSolver->clampExtremaField();
            // update temperature
            gpuSolver->copyDeviceToHost(_T, gpuSolver->x_host, gpuSolver->dfield);
        }
        else
        {
            // update temperature
            gpuSolver->copyDeviceToHost(_T, gpuSolver->x_host, gpuSolver->field);
        }
        cout << "[ Reflection Advect Temperature Done! ]" << endl;
    }

    _uinit.copy(_un);
    _vinit.copy(_vn);
    _winit.copy(_wn);
    // semi-lagrangian advected velocity will be stored in gpu.du, gpu.dv, gpu.dw
    // negate dt for tracing back
    cudaMemcpy(gpuSolver->u_src, gpuSolver->u, (_nx+1)*_ny*_nz*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpuSolver->v_src, gpuSolver->v, _nx*(_ny+1)*_nz*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpuSolver->w_src, gpuSolver->w, _nx*_ny*(_nz+1)*sizeof(float), cudaMemcpyDeviceToDevice);
    gpuSolver->semilagAdvectVelocity(!do_antialiasing, cfldt, -0.5f*dt);
    if (!do_SF)
    {
        cudaMemcpy(gpuSolver->u_src, gpuSolver->du, (_nx + 1) * _ny * _nz * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(gpuSolver->v_src, gpuSolver->dv, _nx * (_ny + 1) * _nz * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(gpuSolver->w_src, gpuSolver->dw, _nx * _ny * (_nz + 1) * sizeof(float), cudaMemcpyDeviceToDevice);
        gpuSolver->semilagAdvectVelocity(!do_antialiasing, cfldt, 0.5f * dt);
        cudaMemcpy(gpuSolver->u, gpuSolver->u_src, (_nx + 1) * _ny * _nz * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(gpuSolver->v, gpuSolver->v_src, _nx * (_ny + 1) * _nz * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(gpuSolver->w, gpuSolver->w_src, _nx * _ny * (_nz + 1) * sizeof(float), cudaMemcpyDeviceToDevice);
        gpuSolver->add(gpuSolver->u, gpuSolver->du, -0.5f, (_nx + 1) * _ny * _nz);
        gpuSolver->add(gpuSolver->v, gpuSolver->dv, -0.5f, _nx * (_ny + 1) * _nz);
        gpuSolver->add(gpuSolver->w, gpuSolver->dw, -0.5f, _nx * _ny * (_nz + 1));
        gpuSolver->copyHostToDevice(_un, gpuSolver->u_host, gpuSolver->du, (_nx + 1) * _ny * _nz * sizeof(float));
        gpuSolver->copyHostToDevice(_vn, gpuSolver->v_host, gpuSolver->dv, _nx * (_ny + 1) * _nz * sizeof(float));
        gpuSolver->copyHostToDevice(_wn, gpuSolver->w_host, gpuSolver->dw, _nx * _ny * (_nz + 1) * sizeof(float));
        gpuSolver->add(gpuSolver->u, gpuSolver->du, 0.5f, (_nx + 1) * _ny * _nz);
        gpuSolver->add(gpuSolver->v, gpuSolver->dv, 0.5f, _nx * (_ny + 1) * _nz);
        gpuSolver->add(gpuSolver->w, gpuSolver->dw, 0.5f, _nx * _ny * (_nz + 1));
        // clamp extrema, clamped new velocity will be in GPU.u_src, GPU.v_src, GPU.w_src
        if (do_EC_with_clamp)
            gpuSolver->clampExtremaVelocity();
        // copy velocity back to CPU buffers
        gpuSolver->copyDeviceToHost(_un, gpuSolver->u_host, gpuSolver->u);
        gpuSolver->copyDeviceToHost(_vn, gpuSolver->v_host, gpuSolver->v);
        gpuSolver->copyDeviceToHost(_wn, gpuSolver->w_host, gpuSolver->w);
    }
    else
    {
        // copy velocity back to CPU buffers
        gpuSolver->copyDeviceToHost(_un, gpuSolver->u_host, gpuSolver->du);
        gpuSolver->copyDeviceToHost(_vn, gpuSolver->v_host, gpuSolver->dv);
        gpuSolver->copyDeviceToHost(_wn, gpuSolver->w_host, gpuSolver->dw);
    }

    cout << "[ Reflection Advect Velocity First Half Done! ]" << endl;

    if (!do_vel_advection_only)
    {
        clearBoundary(_rho);
        clearBoundary(_T);
    }
    emitSmoke(framenum, dt);
    addBuoyancy(_un, _vn, _wn, 0.5f * dt);

    // add viscosity
    if (viscosity)
    {
        diffuse_field(0.5f*dt, viscosity, _un);
        diffuse_field(0.5f*dt, viscosity, _vn);
        diffuse_field(0.5f*dt, viscosity, _wn);
    }

    _utemp.copy(_un);
    _vtemp.copy(_vn);
    _wtemp.copy(_wn);
    projection();
    _duproj.copy(_un);
    _dvproj.copy(_vn);
    _dwproj.copy(_wn);
    _duproj *= 2.f;
    _dvproj *= 2.f;
    _dwproj *= 2.f;
    if (do_2nd_order)
    {
        _un = _duproj - _uinit;
        _vn = _dvproj - _vinit;
        _wn = _dwproj - _winit;
    }
    _duproj -= _utemp;
    _dvproj -= _vtemp;
    _dwproj -= _wtemp;
    if (do_2nd_order)
    {
        addBuoyancy(_duproj, _dvproj, _dwproj, 0.5f * dt);

        // add viscosity
        if (viscosity)
        {
            diffuse_field(0.5f * dt, viscosity, _duproj);
            diffuse_field(0.5f * dt, viscosity, _dvproj);
            diffuse_field(0.5f * dt, viscosity, _dwproj);
        }
    }

    gpuSolver->copyHostToDevice(_un, gpuSolver->u_host, gpuSolver->u, (_nx+1)*_ny*_nz*sizeof(float));
    gpuSolver->copyHostToDevice(_vn, gpuSolver->v_host, gpuSolver->v, _nx*(_ny+1)*_nz*sizeof(float));
    gpuSolver->copyHostToDevice(_wn, gpuSolver->w_host, gpuSolver->w, _nx*_ny*(_nz+1)*sizeof(float));
    gpuSolver->copyHostToDevice(_duproj, gpuSolver->u_host, gpuSolver->u_src, (_nx+1)*_ny*_nz*sizeof(float));
    gpuSolver->copyHostToDevice(_dvproj, gpuSolver->v_host, gpuSolver->v_src, _nx*(_ny+1)*_nz*sizeof(float));
    gpuSolver->copyHostToDevice(_dwproj, gpuSolver->w_host, gpuSolver->w_src, _nx*_ny*(_nz+1)*sizeof(float));
    gpuSolver->semilagAdvectVelocity(!do_antialiasing, cfldt, -0.5f*dt);
    if (!do_SF)
    {
        cudaMemcpy(gpuSolver->u_src, gpuSolver->du, (_nx + 1) * _ny * _nz * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(gpuSolver->v_src, gpuSolver->dv, _nx * (_ny + 1) * _nz * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(gpuSolver->w_src, gpuSolver->dw, _nx * _ny * (_nz + 1) * sizeof(float), cudaMemcpyDeviceToDevice);
        gpuSolver->semilagAdvectVelocity(!do_antialiasing, cfldt, 0.5f * dt);
        cudaMemcpy(gpuSolver->u, gpuSolver->u_src, (_nx + 1) * _ny * _nz * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(gpuSolver->v, gpuSolver->v_src, _nx * (_ny + 1) * _nz * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(gpuSolver->w, gpuSolver->w_src, _nx * _ny * (_nz + 1) * sizeof(float), cudaMemcpyDeviceToDevice);
        gpuSolver->add(gpuSolver->u, gpuSolver->du, -0.5f, (_nx + 1) * _ny * _nz);
        gpuSolver->add(gpuSolver->v, gpuSolver->dv, -0.5f, _nx * (_ny + 1) * _nz);
        gpuSolver->add(gpuSolver->w, gpuSolver->dw, -0.5f, _nx * _ny * (_nz + 1));
        gpuSolver->copyHostToDevice(_duproj, gpuSolver->u_host, gpuSolver->du, (_nx + 1) * _ny * _nz * sizeof(float));
        gpuSolver->copyHostToDevice(_dvproj, gpuSolver->v_host, gpuSolver->dv, _nx * (_ny + 1) * _nz * sizeof(float));
        gpuSolver->copyHostToDevice(_dwproj, gpuSolver->w_host, gpuSolver->dw, _nx * _ny * (_nz + 1) * sizeof(float));
        gpuSolver->add(gpuSolver->u, gpuSolver->du, 0.5f, (_nx + 1) * _ny * _nz);
        gpuSolver->add(gpuSolver->v, gpuSolver->dv, 0.5f, _nx * (_ny + 1) * _nz);
        gpuSolver->add(gpuSolver->w, gpuSolver->dw, 0.5f, _nx * _ny * (_nz + 1));
        // clamp extrema, clamped new velocity will be in GPU.u_src, GPU.v_src, GPU.w_src
        if (do_EC_with_clamp)
            gpuSolver->clampExtremaVelocity();
        // copy velocity back to CPU buffers
        gpuSolver->copyDeviceToHost(_un, gpuSolver->u_host, gpuSolver->u);
        gpuSolver->copyDeviceToHost(_vn, gpuSolver->v_host, gpuSolver->v);
        gpuSolver->copyDeviceToHost(_wn, gpuSolver->w_host, gpuSolver->w);
    }
    else
    {
        // copy velocity back to CPU buffers
        gpuSolver->copyDeviceToHost(_un, gpuSolver->u_host, gpuSolver->du);
        gpuSolver->copyDeviceToHost(_vn, gpuSolver->v_host, gpuSolver->dv);
        gpuSolver->copyDeviceToHost(_wn, gpuSolver->w_host, gpuSolver->dw);
    }
    cout << "[ Reflection Advect Velocity Second Half Done! ]" << endl;

    if (!do_2nd_order)
    {
        addBuoyancy(_un, _vn, _wn, 0.5f * dt);

        // add viscosity
        if (viscosity)
        {
            diffuse_field(0.5f * dt, viscosity, _un);
            diffuse_field(0.5f * dt, viscosity, _vn);
            diffuse_field(0.5f * dt, viscosity, _wn);
        }
    }

    projection();
}

void CovectorSolver::diffuse_field(double dt, double nu, buffer3Df &field)
{
    buffer3Df field_temp;
    field_temp.init(field._nx, field._ny, field._nz, field._hx, 0,0,0);
    field_temp.setZero();
    field_temp.copy(field);
    int compute_elements = field_temp._blockx*field_temp._blocky*field_temp._blockz;

    int slice = field_temp._blockx*field_temp._blocky;
    double coef = nu*(dt/(_h*_h));

    for(int iter = 0; iter<20; iter++) {
        tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

            int bk = thread_idx / slice;
            int bj = (thread_idx % slice) / field_temp._blockx;
            int bi = thread_idx % (field_temp._blockx);

            for (int kk = 0; kk < 8; kk++)
                for (int jj = 0; jj < 8; jj++)
                    for (int ii = 0; ii < 8; ii++) {
                        int i = bi * 8 + ii, j = bj * 8 + jj, k = bk * 8 + kk;
                        if((i+j+k)%2==0)
                            field_temp(i, j, k) = (field(i, j, k) + coef * (
                                     field_temp.at(i - 1, j, k) +
                                     field_temp.at(i + 1, j, k) +
                                     field_temp.at(i, j - 1, k) +
                                     field_temp.at(i, j + 1, k) +
                                     field_temp.at(i, j, k - 1) +
                                     field_temp.at(i, j, k + 1)
                            )) / (1.0f + 6.0f * coef);
                    }
        });
        tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

            int bk = thread_idx / slice;
            int bj = (thread_idx % slice) / field_temp._blockx;
            int bi = thread_idx % (field_temp._blockx);

            for (int kk = 0; kk < 8; kk++)
                for (int jj = 0; jj < 8; jj++)
                    for (int ii = 0; ii < 8; ii++) {
                        int i = bi * 8 + ii, j = bj * 8 + jj, k = bk * 8 + kk;
                        if((i+j+k)%2==1)
                            field_temp(i, j, k) = (field(i, j, k) + coef * (
                                 field_temp.at(i - 1, j, k) +
                                 field_temp.at(i + 1, j, k) +
                                 field_temp.at(i, j - 1, k) +
                                 field_temp.at(i, j + 1, k) +
                                 field_temp.at(i, j, k - 1) +
                                 field_temp.at(i, j, k + 1)
                            )) / (1.0f + 6.0f * coef);
                    }
        });
    }
    field.copy(field_temp);
    field_temp.free();
}

void CovectorSolver::clampExtrema(float dt, buffer3Df &f_n, buffer3Df &f_np1)
{
    int compute_elements = f_np1._blockx*f_np1._blocky*f_np1._blockz;
    int slice = f_np1._blockx*f_np1._blocky;

    tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

        uint bk = thread_idx/slice;
        uint bj = (thread_idx%slice)/f_np1._blockx;
        uint bi = thread_idx%(f_np1._blockx);

        for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
                {
                    uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
                    if(i<f_np1._nx && j<f_np1._ny && k<f_np1._nz)
                    {

                        float world_x = ((float)i-f_np1._ox)*_h;
                        float world_y = ((float)j-f_np1._oy)*_h;
                        float world_z = ((float)k-f_np1._oz)*_h;
                        //Vec3f pos(world_x,world_y,world_z);
                        //Vec3f trace_pos = trace(dt, pos);
                        float u = _un.sample_linear(world_x,world_y,world_z);
                        float v = _vn.sample_linear(world_x,world_y,world_z);
                        float w = _wn.sample_linear(world_x,world_y,world_z);

                        float px = world_x - 0.5*dt * u, py = world_y - 0.5*dt *v, pz = world_z - 0.5*dt*w;
                        u = _un.sample_linear(px,py,pz);
                        v = _vn.sample_linear(px,py,pz);
                        w = _wn.sample_linear(px,py,pz);

                        px = world_x - dt * u, py = world_y - dt *v, pz = world_z - dt*w;

                        float v0,v1,v2,v3,v4,v5,v6,v7;
                        //f_n.sample_cube(px,py,pz,v0,v1,v2,v3,v4,v5,v6,v7);
                        float SLv = f_n.sample_cube_lerp(px,py,pz,
                                                         v0,v1,v2,v3,v4,v5,v6,v7);

                        float min_value = min(v0,min(v1,min(v2,min(v3,min(v4,min(v5,min(v6,v7)))))));
                        float max_value = max(v0,max(v1,max(v2,max(v3,max(v4,max(v5,max(v6,v7)))))));

                        if(f_np1(i,j,k)<min_value || f_np1(i,j,k)>max_value)
                        {
                            f_np1(i,j,k) = SLv;
                        }
                        //f_np1(i,j,k) = max(min(max_value, f_np1(i,j,k)),min_value);
                    }
                }
    });
}

void CovectorSolver::semilagAdvect(float cfldt, float dt, bool only_vel_update)
{
    // NOTE: TO SAVE TRANSFER TIME, NEED U,V,W BE STORED IN GPU.U, GPU.V, GPU.W ALREADY
    // semi-lagrangian advected velocity will be stored in gpu.du, gpu.dv, gpu.dw
    // negate dt for tracing back
    cudaMemcpy(gpuSolver->u_src, gpuSolver->u, (_nx+1)*_ny*_nz*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpuSolver->v_src, gpuSolver->v, _nx*(_ny+1)*_nz*sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(gpuSolver->w_src, gpuSolver->w, _nx*_ny*(_nz+1)*sizeof(float), cudaMemcpyDeviceToDevice);
    gpuSolver->semilagAdvectVelocity(!do_antialiasing, cfldt, -dt);
    gpuSolver->copyDeviceToHost(_utemp, gpuSolver->u_host, gpuSolver->du);
    gpuSolver->copyDeviceToHost(_vtemp, gpuSolver->v_host, gpuSolver->dv);
    gpuSolver->copyDeviceToHost(_wtemp, gpuSolver->w_host, gpuSolver->dw);
    if (!only_vel_update && !do_vel_advection_only)
    {
        // semi-lagrangian advect any other fluid fields
        // reuse gpu.du, gpu.dv to save GPU buffer
        // copy field to gpu.dv for semi-lagrangian advection
        // advect density
        gpuSolver->copyHostToDevice(_rho, gpuSolver->x_host, gpuSolver->dfield, _nx * _ny * _nz * sizeof(float));
        gpuSolver->semilagAdvectField(!do_antialiasing, cfldt, -dt);
        gpuSolver->copyDeviceToHost(_rhotemp, gpuSolver->x_host, gpuSolver->field);
        // advect Temperature
        gpuSolver->copyHostToDevice(_T, gpuSolver->x_host, gpuSolver->dfield, _nx * _ny * _nz * sizeof(float));
        gpuSolver->semilagAdvectField(!do_antialiasing, cfldt, -dt);
        gpuSolver->copyDeviceToHost(_Ttemp, gpuSolver->x_host, gpuSolver->field);
    }
}

void CovectorSolver::setInitialVelocity(float inflow_vel)
{
    int compute_elements = _un._nx * _un._ny * _un._nz;
    int slice = _un._nx * _un._ny;

    tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

        uint k = thread_idx/slice;
        uint j = (thread_idx%slice)/_un._nx;
        uint i = thread_idx%(_un._nx);

        _un(i,j,k) = inflow_vel;
    });
}

void CovectorSolver::setVelocityFromEmitter(bool do_only_x_dir_vel)
{
    for(auto &emitter : sim_emitter)
    {
        openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> box_sampler(*emitter.e_sdf);

//            float in_value = -emitter.e_sdf->background();

            int compute_elements = 0;
            int slice = 0;
            if (emitter.do_set_velocities)
            {
                compute_elements = _un._blockx * _un._blocky * _un._blockz;
                slice = _un._blockx * _un._blocky;

                tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

                    uint bk = thread_idx/slice;
                    uint bj = (thread_idx%slice)/_un._blockx;
                    uint bi = thread_idx%(_un._blockx);

                    for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
                    {
                        uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;

                        if(i<_un._nx && j<_un._ny && k<_un._nz)
                        {
                            float w_x = ((float)i-_un._ox)*_h;
                            float w_y = ((float)j-_un._oy)*_h;
                            float w_z = ((float)k-_un._oz)*_h;
                            Vec3f world_pos(w_x, w_y, w_z);
                            float sdf_value = box_sampler.wsSample(openvdb::Vec3R(w_x, w_y, w_z));
                            if (sdf_value <= 0)
                            {
                                _un(i,j,k) = emitter.emit_velocity(world_pos)[0];
                            }
                        }
                    }
                });

                if (!do_only_x_dir_vel)
                {
                    compute_elements = _vn._blockx*_vn._blocky*_vn._blockz;
                    slice = _vn._blockx*_vn._blocky;

                    tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

                        uint bk = thread_idx/slice;
                        uint bj = (thread_idx%slice)/_vn._blockx;
                        uint bi = thread_idx%(_vn._blockx);

                        for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
                        {
                            uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
                            if(i<_vn._nx && j<_vn._ny && k<_vn._nz)
                            {
                                float w_x = ((float)i-_vn._ox)*_h;
                                float w_y = ((float)j-_vn._oy)*_h;
                                float w_z = ((float)k-_vn._oz)*_h;
                                Vec3f world_pos(w_x, w_y, w_z);
                                float sdf_value = box_sampler.wsSample(openvdb::Vec3R(w_x, w_y, w_z));
                                if (sdf_value <= 0)
                                {
                                    _vn(i,j,k) = emitter.emit_velocity(world_pos)[1];
                                }
                            }
                        }
                    });

                    compute_elements = _wn._blockx*_wn._blocky*_wn._blockz;
                    slice = _wn._blockx*_wn._blocky;

                    tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

                        uint bk = thread_idx/slice;
                        uint bj = (thread_idx%slice)/_wn._blockx;
                        uint bi = thread_idx%(_wn._blockx);

                        for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
                        {
                            uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
                            if(i<_wn._nx && j<_wn._ny && k<_wn._nz)
                            {
                                float w_x = ((float)i-_wn._ox)*_h;
                                float w_y = ((float)j-_wn._oy)*_h;
                                float w_z = ((float)k-_wn._oz)*_h;
                                Vec3f world_pos(w_x, w_y, w_z);
                                float sdf_value = box_sampler.wsSample(openvdb::Vec3R(w_x, w_y, w_z));
                                if (sdf_value <= 0)
                                {
                                    _wn(i,j,k) = emitter.emit_velocity(world_pos)[2];
                                }
                            }
                        }
                    });
                }
            }
    }
}

void CovectorSolver::emitSmoke(int framenum, float dt)
{
    for(auto &emitter : sim_emitter)
    {
        emitter.update(framenum, _h, dt);
        openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> box_sampler(*emitter.e_sdf);
        if(framenum < emitter.emitFrame)
        {
//            float in_value = -emitter.e_sdf->background();

            int compute_elements = 0;
            int slice = 0;
            if (emitter.do_set_velocities)
            {
                compute_elements = _un._blockx * _un._blocky * _un._blockz;
                slice = _un._blockx * _un._blocky;

                tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

                    uint bk = thread_idx/slice;
                    uint bj = (thread_idx%slice)/_un._blockx;
                    uint bi = thread_idx%(_un._blockx);

                    for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
                    {
                        uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;

                        if(i<_un._nx && j<_un._ny && k<_un._nz)
                        {
                            float w_x = ((float)i-_un._ox)*_h;
                            float w_y = ((float)j-_un._oy)*_h;
                            float w_z = ((float)k-_un._oz)*_h;
                            Vec3f world_pos(w_x, w_y, w_z);
                            float sdf_value = box_sampler.wsSample(openvdb::Vec3R(w_x, w_y, w_z));
                            if (sdf_value <= 0)
                            {
                                _un(i,j,k) = emitter.emit_velocity(world_pos)[0];
                            }
                        }
                    }
                });

                compute_elements = _vn._blockx*_vn._blocky*_vn._blockz;
                slice = _vn._blockx*_vn._blocky;

                tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

                    uint bk = thread_idx/slice;
                    uint bj = (thread_idx%slice)/_vn._blockx;
                    uint bi = thread_idx%(_vn._blockx);

                    for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
                    {
                        uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
                        if(i<_vn._nx && j<_vn._ny && k<_vn._nz)
                        {
                            float w_x = ((float)i-_vn._ox)*_h;
                            float w_y = ((float)j-_vn._oy)*_h;
                            float w_z = ((float)k-_vn._oz)*_h;
                            Vec3f world_pos(w_x, w_y, w_z);
                            float sdf_value = box_sampler.wsSample(openvdb::Vec3R(w_x, w_y, w_z));
                            if (sdf_value <= 0)
                            {
                                _vn(i,j,k) = emitter.emit_velocity(world_pos)[1];
                            }
                        }
                    }
                });

                compute_elements = _wn._blockx*_wn._blocky*_wn._blockz;
                slice = _wn._blockx*_wn._blocky;

                tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

                    uint bk = thread_idx/slice;
                    uint bj = (thread_idx%slice)/_wn._blockx;
                    uint bi = thread_idx%(_wn._blockx);

                    for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
                    {
                        uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
                        if(i<_wn._nx && j<_wn._ny && k<_wn._nz)
                        {
                            float w_x = ((float)i-_wn._ox)*_h;
                            float w_y = ((float)j-_wn._oy)*_h;
                            float w_z = ((float)k-_wn._oz)*_h;
                            Vec3f world_pos(w_x, w_y, w_z);
                            float sdf_value = box_sampler.wsSample(openvdb::Vec3R(w_x, w_y, w_z));
                            if (sdf_value <= 0)
                            {
                                _wn(i,j,k) = emitter.emit_velocity(world_pos)[2];
                            }
                        }
                    }
                });
            }

            if (!do_vel_advection_only)
            {
                compute_elements = _rho._blockx * _rho._blocky * _rho._blockz;
                slice = _rho._blockx * _rho._blocky;

                tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

                    uint bk = thread_idx / slice;
                    uint bj = (thread_idx % slice) / _rho._blockx;
                    uint bi = thread_idx % (_rho._blockx);

                    for (uint kk = 0; kk < 8; kk++)for (uint jj = 0; jj < 8; jj++)for (uint ii = 0; ii < 8; ii++)
                    {
                        uint i = bi * 8 + ii, j = bj * 8 + jj, k = bk * 8 + kk;
                        if (i < _rho._nx && j < _rho._ny && k < _rho._nz)
                        {
                            float w_x = ((float)i - _rho._ox) * _h;
                            float w_y = ((float)j - _rho._oy) * _h;
                            float w_z = ((float)k - _rho._oz) * _h;
                            float sdf_value = box_sampler.wsSample(openvdb::Vec3R(w_x, w_y, w_z));
                            if (sdf_value <= 0)
                            {
                                float multiplier = 1.0;
                                float thickness = 2 * _h;
                                //if (thickness + sdf_value > 0.f)
                                    multiplier = -sdf_value / thickness;
                                float rand_mulitplier = 1.f;
                                if (emitter.do_randomize_density)
                                {
                                    rand_mulitplier += (float)rand() / RAND_MAX;
                                }
                                _rho(i, j, k) += rand_mulitplier * multiplier * emitter.emit_density;
                                _T(i, j, k) += rand_mulitplier * multiplier * emitter.emit_temperature;
                            }
                        }
                    }
                });
            }
        }
    }
}

void CovectorSolver::addBuoyancy(buffer3Df& u_to_change, buffer3Df& v_to_change, buffer3Df& w_to_change, float dt)
{
    if (_alpha == 0.0f && _beta == 0.0f)
        return;

    int compute_elements = _rho._blockx*_rho._blocky*_rho._blockz;
    int slice = _rho._blockx*_rho._blocky;

    tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

        uint bk = thread_idx/slice;
        uint bj = (thread_idx%slice)/_rho._blockx;
        uint bi = thread_idx%(_rho._blockx);

        for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
        {
            uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
            if(i<_nx && j<_ny && k<_nz)
            {
                float density = _rho(i,j,k);
                float temperature = _T(i,j,k);
                float f = -dt*_alpha*density + dt*_beta*temperature;

                v_to_change(i,j,k) += 0.5*f*sin(theta);
                u_to_change(i,j,k) += 0.5*f*cos(theta)*cos(phi);
                w_to_change(i,j,k) += 0.5*f*cos(theta)*sin(phi);
            }
        }
    });
    tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

        uint bk = thread_idx/slice;
        uint bj = (thread_idx%slice)/_rho._blockx;
        uint bi = thread_idx%(_rho._blockx);

        for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
        {
            uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
            if(i<_nx && j>0&&j<_ny && k<_nz)
            {
                float density = _rho(i,j,k);
                float temperature = _T(i,j,k);
                float f = -dt*_alpha*density + dt*_beta*temperature;

                v_to_change(i,j+1,k) += 0.5*f*sin(theta);
                u_to_change(i+1,j,k) += 0.5*f*cos(theta)*cos(phi);
                w_to_change(i,j,k+1) += 0.5*f*cos(theta)*sin(phi);
            }
        }
    });
}

void CovectorSolver::setSmoke(float drop, float raise, const std::vector<Emitter> &emitters)
{
    _alpha = drop;
    _beta = raise;
    sim_emitter = emitters;
}

void CovectorSolver::blendBoundary(buffer3Df &field, const buffer3Df &blend_field)
{
    for(const auto &boundary : sim_boundary)
    {
        openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> box_sampler(*boundary.b_sdf);

        int compute_elements = field._blockx*field._blocky*field._blockz;
        int slice = field._blockx*field._blocky;

        tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

            uint bk = thread_idx/slice;
            uint bj = (thread_idx%slice)/field._blockx;
            uint bi = thread_idx%(field._blockx);

            for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
            {
                uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
                if(i<field._nx && j<field._ny && k<field._nz)
                {
                    float w_x = ((float)i-field._ox)*_h;
                    float w_y = ((float)j-field._oy)*_h;
                    float w_z = ((float)k-field._oz)*_h;
                    float sdf_value = box_sampler.wsSample(openvdb::Vec3R(w_x, w_y, w_z));
                    float background_value = boundary.b_sdf->background();
                    if (sdf_value > 0.f && sdf_value < background_value)
                    {
                        field(i,j,k) = blend_field(i,j,k);
                    }
                }
            }
        });
    }
}

void CovectorSolver::clearBoundary(buffer3Df& field)
{
    int compute_elements = field._blockx* field._blocky* field._blockz;

    int slice = field._blockx* field._blocky;
    tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

        uint bk = thread_idx/slice;
        uint bj = (thread_idx%slice)/ field._blockx;
        uint bi = thread_idx%(field._blockx);

        for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
        {
            uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
            if(_b_desc(i,j,k)==1 || _b_desc(i,j,k)==3)
            {
                field(i,j,k) = 0;
            }
        }
    });
}

void CovectorSolver::updateBoundary(int framenum, float dt)
{
    _b_desc.setZero();
    for (int k=0;k<_nz;k++)for(int j=0;j<_ny;j++)for(int i=0;i<_nx;i++)
    {
        //0:fluid;1:air;2:solid
        if (do_sides_solid)
        {
            if (i < 1) _b_desc(i, j, k) = do_inflow_solid ? 2 : 1;
            if (j < 1) _b_desc(i, j, k) = 2;
            if (k < 1) _b_desc(i, j, k) = 2;

            if (i >= _nx - 1) _b_desc(i, j, k) = 1;
            if (j >= _ny - 1) _b_desc(i, j, k) = 2;
            if (k >= _nz - 1) _b_desc(i, j, k) = 2;
        }
        else if (do_inflow_solid)
        {
            if (i < 1) _b_desc(i, j, k) = 2;
            if (j < 1) _b_desc(i, j, k) = 1;
            if (k < 1) _b_desc(i, j, k) = 1;

            if (i >= _nx - 1) _b_desc(i, j, k) = 1;
            if (j >= _ny - 1) _b_desc(i, j, k) = 1;
            if (k >= _nz - 1) _b_desc(i, j, k) = 1;
        }
        else if (do_empty_top)
        {
            if (i < 1) _b_desc(i, j, k) = 1;
            if (j < 1) _b_desc(i, j, k) = 1;
            if (k < 1) _b_desc(i, j, k) = 1;

            if (i >= _nx - 1) _b_desc(i, j, k) = 1;
            if (j >= _ny - 1 - _extraPad) _b_desc(i, j, k) = 1;
            if (k >= _nz - 1) _b_desc(i, j, k) = 1;
        }
        else
        {
            if(i<1) _b_desc(i,j,k) = 2;
            if(j<1) _b_desc(i,j,k) = 2;
            if(k<1) _b_desc(i,j,k) = 2;

            if(i>=_nx-1) _b_desc(i,j,k) = 2;
            if(j>=_ny-1) _b_desc(i,j,k) = 2;
            if(k>=_nz-1) _b_desc(i,j,k) = 2;
        }
    }
    for(auto &boundary : sim_boundary)
    {
        Vec3f boundary_vel = boundary.vel_func(framenum);
        boundary.update(framenum, _h, dt);
        openvdb::tools::GridSampler<openvdb::FloatGrid, openvdb::tools::BoxSampler> box_sampler(*boundary.b_sdf);

        int compute_elements = _b_desc._blockx*_b_desc._blocky*_b_desc._blockz;
        int slice = _b_desc._blockx*_b_desc._blocky;

        tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

            uint bk = thread_idx/slice;
            uint bj = (thread_idx%slice)/_b_desc._blockx;
            uint bi = thread_idx%(_b_desc._blockx);

            for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
            {
                uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
                if(i<_b_desc._nx && j<_b_desc._ny && k<_b_desc._nz)
                {
                    float w_x = ((float)i-_b_desc._ox)*_h;
                    float w_y = ((float)j-_b_desc._oy)*_h;
                    float w_z = ((float)k-_b_desc._oz)*_h;
                    float sdf_value = box_sampler.wsSample(openvdb::Vec3R(w_x, w_y, w_z));
                    if (sdf_value <= 0.f)
                    {
                        _b_desc(i,j,k) = 3;
                    }
                }
            }
        });

        compute_elements = _un._blockx*_un._blocky*_un._blockz;
        slice = _un._blockx*_un._blocky;

        tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

            uint bk = thread_idx/slice;
            uint bj = (thread_idx%slice)/_un._blockx;
            uint bi = thread_idx%(_un._blockx);

            for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
            {
                uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
                if(i<_un._nx && j<_un._ny && k<_un._nz)
                {
                    float w_x = ((float)i-_un._ox)*_h;
                    float w_y = ((float)j-_un._oy)*_h;
                    float w_z = ((float)k-_un._oz)*_h;
                    float sdf_value = box_sampler.wsSample(openvdb::Vec3R(w_x, w_y, w_z));
                    if (sdf_value <= 0)
                    {
                        _usolid(i,j,k) = boundary_vel[0];
                    }
                }
            }
        });

        compute_elements = _vn._blockx*_vn._blocky*_vn._blockz;
        slice = _vn._blockx*_vn._blocky;

        tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

            uint bk = thread_idx/slice;
            uint bj = (thread_idx%slice)/_vn._blockx;
            uint bi = thread_idx%(_vn._blockx);

            for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
            {
                uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
                if(i<_vn._nx && j<_vn._ny && k<_vn._nz)
                {
                    float w_x = ((float)i-_vn._ox)*_h;
                    float w_y = ((float)j-_vn._oy)*_h;
                    float w_z = ((float)k-_vn._oz)*_h;
                    float sdf_value = box_sampler.wsSample(openvdb::Vec3R(w_x, w_y, w_z));
                    if (sdf_value <= 0)
                    {
                        _vsolid(i,j,k) = boundary_vel[1];
                    }
                }
            }
        });

        compute_elements = _wn._blockx*_wn._blocky*_wn._blockz;
        slice = _wn._blockx*_wn._blocky;

        tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

            uint bk = thread_idx/slice;
            uint bj = (thread_idx%slice)/_wn._blockx;
            uint bi = thread_idx%(_wn._blockx);

            for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
            {
                uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
                if(i<_wn._nx && j<_wn._ny && k<_wn._nz)
                {
                    float w_x = ((float)i-_wn._ox)*_h;
                    float w_y = ((float)j-_wn._oy)*_h;
                    float w_z = ((float)k-_wn._oz)*_h;
                    float sdf_value = box_sampler.wsSample(openvdb::Vec3R(w_x, w_y, w_z));
                    if (sdf_value <= 0)
                    {
                        _wsolid(i,j,k) = boundary_vel[2];
                    }
                }
            }
        });
    }
}

void CovectorSolver::setBoundary(const std::vector<Boundary> &boundaries)
{
    sim_boundary = boundaries;
}

float CovectorSolver::getCFL()
{
    max_v = 1e-4;
    for (uint k=0; k<_nz;k++) for (uint j=0; j<_ny;j++) for (uint i=0; i<_nx+1;i++)
    {
        if (fabs(_un(i,j,k))>max_v)
        {
            max_v = fabs(_un(i,j,k));
        }
    }
    for (uint k=0; k<_nz;k++) for (uint j=0; j<_ny+1;j++) for (uint i=0; i<_nx;i++)
    {
        if (fabs(_vn(i,j,k))>max_v)
        {
            max_v = fabs(_vn(i,j,k));
        }
    }
    for (uint k=0; k<_nz+1;k++) for (uint j=0; j<_ny;j++) for (uint i=0; i<_nx;i++)
    {
        if (fabs(_wn(i,j,k))>max_v)
        {
            max_v = fabs(_wn(i,j,k));
        }
    }
    return _h / max_v;
}

void CovectorSolver::setupPressureProjection()
{
    if (is_fixed_domain)
    {
        int ni = _nx;
        int nj = _ny;
        int nk = _nz;
        int system_size = ni * nj * nk;
        matrix.resize(system_size);
        matrix.zero();
        //set up solver
        int compute_num = ni*nj*nk;
        int slice = ni*nj;
        tbb::parallel_for(0,compute_num,1,[&](int thread_idx)
        {
            int k = thread_idx/slice;
            int j = (thread_idx%slice)/ni;
            int i = thread_idx%ni;
            if(i>=1 && i<ni-1 && j>=1 && j<nj-1 && k>=1 && k<nk-1)
            {
                int index = i + ni*j + ni*nj*k;

                if( _b_desc(i,j,k)==0 )//a fluid cell
                {

                    //right neighbour
                    if( _b_desc(i+1,j,k)==0/* || _b_desc(i+1,j,k)==3*/ ) {//a fluid cell
                        matrix.add_to_element(index, index, 1.0/_h/_h);
                        matrix.add_to_element(index, index + 1, -1.0/_h/_h);
                    }
                    else if( _b_desc(i+1,j,k)==1 )//an empty cell
                    {
                        matrix.add_to_element(index, index, 1.0/_h/_h);
                    }

                    //left neighbour
                    if( _b_desc(i-1,j,k)==0/* || _b_desc(i-1,j,k)==3*/ ) {
                        matrix.add_to_element(index, index, 1.0/_h/_h);
                        matrix.add_to_element(index, index - 1, -1.0/_h/_h);
                    }
                    else if( _b_desc(i-1,j,k)==1 ){

                        matrix.add_to_element(index, index, 1.0/_h/_h);
                    }

                    //top neighbour
                    if( _b_desc(i,j+1,k)==0/* || _b_desc(i,j+1,k)==3*/ ) {//a fluid cell
                        matrix.add_to_element(index, index, 1.0/_h/_h);
                        matrix.add_to_element(index, index + ni, -1.0/_h/_h);
                    }
                    else if( _b_desc(i,j+1,k)==1 )//an empty cell
                    {
                        matrix.add_to_element(index, index, 1.0/_h/_h);
                    }

                    //bottom neighbour
                    if( _b_desc(i,j-1,k)==0/* || _b_desc(i,j-1,k)==3*/ ) {
                        matrix.add_to_element(index, index, 1.0/_h/_h);
                        matrix.add_to_element(index, index - ni, -1.0/_h/_h);
                    }
                    else if( _b_desc(i,j-1,k)==1 ){

                        matrix.add_to_element(index, index, 1.0/_h/_h);
                    }

                    //back neighbour
                    if( _b_desc(i,j,k+1)==0/* || _b_desc(i,j,k+1)==3*/ ) {//a fluid cell
                        matrix.add_to_element(index, index, 1.0/_h/_h);
                        matrix.add_to_element(index, index + ni*nj, -1.0/_h/_h);
                    }
                    else if( _b_desc(i,j,k+1)==1 )//an empty cell
                    {
                        matrix.add_to_element(index, index, 1.0/_h/_h);
                    }

                    //front neighbour
                    if( _b_desc(i,j,k-1)==0/* || _b_desc(i,j,k-1)==3*/ ) {
                        matrix.add_to_element(index, index, 1.0/_h/_h);
                        matrix.add_to_element(index, index - ni*nj, -1.0/_h/_h);
                    }
                    else if( _b_desc(i,j,k-1)==1 ){

                        matrix.add_to_element(index, index, 1.0/_h/_h);
                    }
                }
            }
        });
        fixed_matrix.construct_from_matrix(matrix);
        amg_levelGen.generateLevelsGalerkinCoarsening(A_L, R_L, P_L, S_L, total_level, fixed_matrix, ni, nj, nk);
    }
}

void CovectorSolver::projection()
{
    for (int count = 0; count < pp_repeat_count; count++)
    {
        int ni = _nx;
        int nj = _ny;
        int nk = _nz;

        //write boundary velocity;
        int compute_num = ni*nj*nk;
        int slice = ni*nj;
        tbb::parallel_for(0,compute_num,1,[&](int thread_idx)
        {
            int k = thread_idx/slice;
            int j = (thread_idx%slice)/ni;
            int i = thread_idx%ni;
            if ( _b_desc(i,j,k)==2 || _b_desc(i,j,k)==3)//solid
            {
                _un(i,j,k) = _usolid(i,j,k);
                _un(i+1,j,k) = _usolid(i+1,j,k);
                _vn(i,j,k) = _vsolid(i,j,k);
                _vn(i,j+1,k) = _vsolid(i,j+1,k);
                _wn(i,j,k) = _wsolid(i,j,k);
                _wn(i,j,k+1) = _wsolid(i,j,k+1);
            }
        });

        if (set_velocity_inflow)
            setVelocityFromEmitter(true);

        int system_size = ni * nj * nk;
        bool do_resize_matrix = false;
        if (rhs.size() != system_size) {
            rhs.resize(system_size);
            pressure.resize(system_size);
            do_resize_matrix = true;
        }

        rhs.assign(rhs.size(), 0);
        pressure.assign(pressure.size(), 0);

        if (is_fixed_domain)
        {
            compute_num = ni*nj*nk;
            slice = ni*nj;
            tbb::parallel_for(0,compute_num,1,[&](int thread_idx)
            {
                int k = thread_idx/slice;
                int j = (thread_idx%slice)/ni;
                int i = thread_idx%ni;
                if(i>=1 && i<ni-1 && j>=1 && j<nj-1 && k>=1 && k<nk-1)
                {
                    int index = i + ni*j + ni*nj*k;

                    rhs[index] = 0;
                    pressure[index] = 0;

                    if( _b_desc(i,j,k)==0 )//a fluid cell
                    {
                        rhs[index] -= _un(i+1,j,k) / _h;
                        rhs[index] += _un(i,j,k) / _h;
                        rhs[index] -= _vn(i,j+1,k) / _h;
                        rhs[index] += _vn(i,j,k) / _h;
                        rhs[index] -= _wn(i,j,k+1) / _h;
                        rhs[index] += _wn(i,j,k) / _h;
                    }
                }
            });

            double tolerance;
            int iterations;
            //solver.set_solver_parameters(1e-6, 1000);
            //bool success = solver.solve(matrix, rhs, pressure, tolerance, iterations);
            bool success = AMGPCGSolve(fixed_matrix, rhs, pressure, A_L, R_L, P_L, S_L, total_level, 1e-6, 1000, tolerance, iterations, _nx, _ny, _nz);

            printf("Solver took %d iterations and had residual %e\n", iterations, tolerance);
            if (!success) {
                printf("WARNING: Pressure solve failed!************************************************\n");
            }
        }
        else
        {
            if (do_resize_matrix) {
                matrix.resize(system_size);
            }
            matrix.zero();

            //set up solver
            compute_num = ni*nj*nk;
            slice = ni*nj;
            tbb::parallel_for(0,compute_num,1,[&](int thread_idx)
            {
                int k = thread_idx/slice;
                int j = (thread_idx%slice)/ni;
                int i = thread_idx%ni;
                if(i>=1 && i<ni-1 && j>=1 && j<nj-1 && k>=1 && k<nk-1)
                {
                    int index = i + ni*j + ni*nj*k;

                    rhs[index] = 0;
                    pressure[index] = 0;

                    if( _b_desc(i,j,k)==0 )//a fluid cell
                    {

                        //right neighbour
                        if( _b_desc(i+1,j,k)==0 ) {//a fluid cell
                            matrix.add_to_element(index, index, 1.0/_h/_h);
                            matrix.add_to_element(index, index + 1, -1.0/_h/_h);
                        }
                        else if( _b_desc(i+1,j,k)==1 )//an empty cell
                        {
                            matrix.add_to_element(index, index, 1.0/_h/_h);
                        }
                        rhs[index] -= _un(i+1,j,k) / _h;

                        //left neighbour
                        if( _b_desc(i-1,j,k)==0 ) {
                            matrix.add_to_element(index, index, 1.0/_h/_h);
                            matrix.add_to_element(index, index - 1, -1.0/_h/_h);
                        }
                        else if( _b_desc(i-1,j,k)==1 ){

                            matrix.add_to_element(index, index, 1.0/_h/_h);
                        }
                        rhs[index] += _un(i,j,k) / _h;

                        //top neighbour
                        if( _b_desc(i,j+1,k)==0 ) {//a fluid cell
                            matrix.add_to_element(index, index, 1.0/_h/_h);
                            matrix.add_to_element(index, index + ni, -1.0/_h/_h);
                        }
                        else if( _b_desc(i,j+1,k)==1 )//an empty cell
                        {
                            matrix.add_to_element(index, index, 1.0/_h/_h);
                        }
                        rhs[index] -= _vn(i,j+1,k) / _h;

                        //bottom neighbour
                        if( _b_desc(i,j-1,k)==0 ) {
                            matrix.add_to_element(index, index, 1.0/_h/_h);
                            matrix.add_to_element(index, index - ni, -1.0/_h/_h);
                        }
                        else if( _b_desc(i,j-1,k)==1 ){

                            matrix.add_to_element(index, index, 1.0/_h/_h);
                        }
                        rhs[index] += _vn(i,j,k) / _h;
                        //rhs[index] += _burn_div(i,j,k);



                        //back neighbour
                        if( _b_desc(i,j,k+1)==0 ) {//a fluid cell
                            matrix.add_to_element(index, index, 1.0/_h/_h);
                            matrix.add_to_element(index, index + ni*nj, -1.0/_h/_h);
                        }
                        else if( _b_desc(i,j,k+1)==1 )//an empty cell
                        {
                            matrix.add_to_element(index, index, 1.0/_h/_h);
                        }
                        rhs[index] -= _wn(i,j,k+1) / _h;

                        //front neighbour
                        if( _b_desc(i,j,k-1)==0 ) {
                            matrix.add_to_element(index, index, 1.0/_h/_h);
                            matrix.add_to_element(index, index - ni*nj, -1.0/_h/_h);
                        }
                        else if( _b_desc(i,j,k-1)==1 ){

                            matrix.add_to_element(index, index, 1.0/_h/_h);
                        }
                        rhs[index] += _wn(i,j,k) / _h;


                        //rhs[index] += _burn_div(i,j,k);

                    }
                }
            });

            //Solve the system using a AMGPCG solver

            double tolerance;
            int iterations;
            //solver.set_solver_parameters(1e-6, 1000);
            //bool success = solver.solve(matrix, rhs, pressure, tolerance, iterations);
            bool success = AMGPCGSolve(matrix,rhs,pressure,1e-6,1000,tolerance,iterations,_nx,_ny,_nz);

            printf("Solver took %d iterations and had residual %e\n", iterations, tolerance);
            if(!success) {
                printf("WARNING: Pressure solve failed!************************************************\n");
            }
        }

        u_valid.assign(0);
        v_valid.assign(0);
        w_valid.assign(0);

        //write boundary velocity
        compute_num = ni*nj*nk;
        slice = ni*nj;
        tbb::parallel_for(0,compute_num,1,[&](int thread_idx)
        {
            uint k = thread_idx/slice;
            uint j = (thread_idx%slice)/ni;
            uint i = thread_idx%ni;
            if ( _b_desc(i,j,k)==2 || _b_desc(i,j,k)==3)//solid
            {
                _un(i,j,k) = _usolid(i,j,k);
                u_valid(i,j,k) = 1;
                _un(i+1,j,k) = _usolid(i+1,j,k);
                u_valid(i+1,j,k) =1;
                _vn(i,j,k) = _vsolid(i,j,k);
                v_valid(i,j,k) = 1;
                _vn(i,j+1,k) = _vsolid(i,j+1,k);
                v_valid(i,j+1,k) = 1;
                _wn(i,j,k) = _wsolid(i,j,k);
                w_valid(i,j,k) = 1;
                _wn(i,j,k+1) = _wsolid(i,j,k+1);
                w_valid(i,j,k+1) =1;
            }
        });

        //apply grad
        compute_num = _un._nx*_un._ny*_un._nz;
        slice = _un._nx*_un._ny;
        tbb::parallel_for(0, compute_num, 1, [&](int thread_idx)
        {
            uint k = thread_idx/slice;
            uint j = (thread_idx%slice)/_un._nx;
            uint i = thread_idx%_un._nx;
            if(u_valid(i,j,k) == 0 &&
               i<_un._nx-1 && i>0 &&
               (_b_desc(i,j,k) == 0 || _b_desc(i-1,j,k) == 0)) {
                int index = i + j*ni + k*ni*nj;
                _un(i,j,k) -=  (float)(pressure[index] - pressure[index-1]) / _h ;
                u_valid(i,j,k) = 1;
            }
        });

        compute_num = _vn._nx*_vn._ny*_vn._nz;
        slice = _vn._nx*_vn._ny;
        tbb::parallel_for(0, compute_num, 1, [&](int thread_idx)
        {
            uint k = thread_idx/slice;
            uint j = (thread_idx%slice)/_vn._nx;
            uint i = thread_idx%_vn._nx;
            if(v_valid(i,j,k) == 0 &&
               j>0 && j<_vn._ny-1 &&
               (_b_desc(i,j,k) == 0 || _b_desc(i,j-1,k) == 0)) {
                int index = i + j*ni + k*ni*nj;
                _vn(i,j,k) -=  (float)(pressure[index] - pressure[index-ni]) / _h ;
                v_valid(i,j,k) = 1;
            }
        });

        compute_num = _wn._nx*_wn._ny*_wn._nz;
        slice = _wn._nx*_wn._ny;
        tbb::parallel_for(0, compute_num, 1, [&](int thread_idx)
        {
            uint k = thread_idx/slice;
            uint j = (thread_idx%slice)/_wn._nx;
            uint i = thread_idx%_wn._nx;
            if(w_valid(i,j,k) == 0 &&
               k>0 && k<_wn._nz-1 &&
               (_b_desc(i,j,k) == 0 || _b_desc(i,j,k-1) == 0)) {
                int index = i + j*ni + k*ni*nj;
                _wn(i,j,k) -=  (float)(pressure[index] - pressure[index-ni*nj]) / _h ;
                w_valid(i,j,k) = 1;
            }
        });

        if (set_velocity_inflow)
            setVelocityFromEmitter(true);

        //extrapolate velocity
        //int pad = 4;
        //compute_num = _vn._nx*_vn._ny*_vn._nz;
        //slice = _vn._nx*_vn._ny;
        //tbb::parallel_for(0, compute_num, 1, [&](int thread_idx)
        //{
        //    uint k = thread_idx/slice;
        //    uint j = (thread_idx%slice)/_vn._nx;
        //    uint i = thread_idx%_vn._nx;
        //    int last_index_j = _vn._ny - 2 - pad;
        //    if(j>=last_index_j && j<_vn._ny-1) {
        //        _un(i,j,k) = lerp(_un(i,j,k),0.f,clamp((float(j-last_index_j)+0.5f)/(float(pad)+1.f),0.f,1.f));
        //        _un(i+1,j,k) = lerp(_un(i+1,j,k),0.f,clamp((float(j-last_index_j)+0.5f)/(float(pad) +1.f),0.f,1.f));
        //        _vn(i,j,k) = lerp(_vn(i,j,k),1.f,clamp(float(j-last_index_j)/(float(pad) +1.f),0.f,1.f));
        //        _wn(i,j,k) = lerp(_wn(i,j,k),0.f,clamp((float(j-last_index_j)+0.5f)/(float(pad) +1.f),0.f,1.f));
        //        _wn(i,j,k+1) = lerp(_wn(i,j,k+1),0.f,clamp((float(j-last_index_j)+0.5f)/(float(pad) +1.f),0.f,1.f));
        //    }
        //});

        compute_num = _un._nx*_un._ny*_un._nz;
        slice = _un._nx*_un._ny;
        tbb::parallel_for(0, compute_num, 1, [&](int thread_idx)
        {
            uint k = thread_idx/slice;
            uint j = (thread_idx%slice)/_un._nx;
            uint i = thread_idx%_un._nx;
            if(k<_un._nz&& j<_un._ny && i<_un._nx )
            {
                if(u_valid(i,j,k)==0)
                {
                    _un(i,j,k) = 0;
                }
            }
        });

        compute_num = _vn._nx*_vn._ny*_vn._nz;
        slice = _vn._nx*_vn._ny;
        tbb::parallel_for(0, compute_num, 1, [&](int thread_idx)
        {
            uint k = thread_idx/slice;
            uint j = (thread_idx%slice)/_vn._nx;
            uint i = thread_idx%_vn._nx;
            if(k<_vn._nz&& j<_vn._ny && i<_vn._nx )
            {
                if(v_valid(i,j,k)==0)
                {
                    _vn(i,j,k) = 0;
                }
            }
        });

        compute_num = _wn._nx*_wn._ny*_wn._nz;
        slice = _wn._nx*_wn._ny;
        tbb::parallel_for(0, compute_num, 1, [&](int thread_idx)
        {
            uint k = thread_idx/slice;
            uint j = (thread_idx%slice)/_wn._nx;
            uint i = thread_idx%_wn._nx;
            if(k<_wn._nz&& j<_wn._ny && i<_wn._nx )
            {
                if(w_valid(i,j,k)==0)
                {
                    _wn(i,j,k) = 0;
                }
            }
        });

        //extrapolate(u_extrap,_un,u_valid);
        //extrapolate(v_extrap,_vn,v_valid);
        //extrapolate(w_extrap,_wn,w_valid);
    }
}

void CovectorSolver::outputResult(uint frame, string filepath)
{
    if (!do_vel_advection_only)
    {
        writeVDB(frame, filepath, _h, _rho, "density_1");
        writeVDB(frame, filepath, _h, _T, "density_2");
    }
    writeVDB(frame, filepath, _h, _un, "vel_x", true);
    writeVDB(frame, filepath, _h, _vn, "vel_y", true);
    writeVDB(frame, filepath, _h, _wn, "vel_z", true);
    int boundary_index = 0;
    for (auto &b : sim_boundary)
    {
        char file_name[256];
        sprintf(file_name,"%s/sim_boundary%02d_%04d.obj", filepath.c_str(), boundary_index, frame);
        std::string objname(file_name);

        std::vector<openvdb::Vec3s> points;
        std::vector<openvdb::Vec4I> quads;
        openvdb::tools::volumeToMesh<openvdb::FloatGrid>(*b.b_sdf, points, quads);
        writeObj(objname, points, quads);
        boundary_index += 1;
    }
}

void CovectorSolver::setupFromVDBFiles(const std::string& filepathVelField,
                                     const std::string& filepathDensityRhoField, 
                                     const std::string& filepathDensityTempField)
{
    readVDBField(filepathDensityRhoField, _rho, "density");
    readVDBField(filepathDensityTempField, _T, "density");
    readVDBField(filepathVelField, _un, "vel.x");
    readVDBField(filepathVelField, _vn, "vel.y");
    readVDBField(filepathVelField, _wn, "vel.z");
    _rhoinit.copy(_rho);
    _Tinit.copy(_T);
    _uinit.copy(_un);
    _vinit.copy(_vn);
    _winit.copy(_wn);
}

void CovectorSolver::pressureProjectVelField()
{
    gpuSolver->copyHostToDevice(_un, gpuSolver->u_host, gpuSolver->u, (_nx + 1) * _ny * _nz * sizeof(float));
    gpuSolver->copyHostToDevice(_vn, gpuSolver->v_host, gpuSolver->v, _nx * (_ny + 1) * _nz * sizeof(float));
    gpuSolver->copyHostToDevice(_wn, gpuSolver->w_host, gpuSolver->w, _nx * _ny * (_nz + 1) * sizeof(float));
    projection();
    _uinit.copy(_un);
    _vinit.copy(_vn);
    _winit.copy(_wn);
}


void CovectorSolver::velocityReinitialize()
{
    _uprev.copy(_uinit);
    _vprev.copy(_vinit);
    _wprev.copy(_winit);
    // set current buffer as next initial buffer
    _uinit.copy(_un);
    _vinit.copy(_vn);
    _winit.copy(_wn);
}

void CovectorSolver::scalarReinitialize()
{
    _rhoprev.copy(_rhoinit);
    _Tprev.copy(_Tinit);
    // set current buffer as next initial buffer
    _rhoinit.copy(_rho);
    _Tinit.copy(_T);
}