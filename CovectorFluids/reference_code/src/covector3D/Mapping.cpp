//
// Created by ziyin on 19-3-19.
//

#include "Mapping.h"

void MapperBase::moveToTemp(int tempToOrig_origToTemp)
{
    if (tempToOrig_origToTemp == 0)
    {
        forward_x.copy(forward_temp_x);
        forward_y.copy(forward_temp_y);
        forward_z.copy(forward_temp_z);
        backward_x.copy(backward_temp_x);
        backward_y.copy(backward_temp_y);
        backward_z.copy(backward_temp_z);
    }
    else
    {
        backward_temp_x.copy(backward_x);
        backward_temp_y.copy(backward_y);
        backward_temp_z.copy(backward_z);
        forward_temp_x.copy(forward_x);
        forward_temp_y.copy(forward_y);
        forward_temp_z.copy(forward_z);
    }
}

void MapperBase::updateBackward(float cfldt, float dt, int do_SF_DMC)
{
    // the backward mapping will be updated in gpu.x_in, gpu.y_in, gpu.z_in
    gpuSolver->copyHostToDevice(backward_x, gpuSolver->x_host, gpuSolver->x_in, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(backward_y, gpuSolver->y_host, gpuSolver->y_in, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(backward_z, gpuSolver->z_host, gpuSolver->z_in, _ni*_nj*_nk*sizeof(float));
    float T = 0.f;
    float substep = cfldt;
    while (T < dt)
    {
        if (T + substep > dt) substep = dt - T;
        gpuSolver->solveBackward(substep, do_SF_DMC);
        T += substep;
    }
    gpuSolver->copyDeviceToHost(backward_x, gpuSolver->x_host, gpuSolver->x_in);
    gpuSolver->copyDeviceToHost(backward_y, gpuSolver->y_host, gpuSolver->y_in);
    gpuSolver->copyDeviceToHost(backward_z, gpuSolver->z_host, gpuSolver->z_in);
}

void MapperBase::updateForward(float cfldt, float dt)
{
    // the forward mapping will be updated in gpu.x_out, gpu.y_out, gpu.z_out
    gpuSolver->copyHostToDevice(forward_x, gpuSolver->x_host, gpuSolver->x_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(forward_y, gpuSolver->y_host, gpuSolver->y_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(forward_z, gpuSolver->z_host, gpuSolver->z_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->solveForward(cfldt, dt);
    gpuSolver->copyDeviceToHost(forward_x, gpuSolver->x_host, gpuSolver->x_out);
    gpuSolver->copyDeviceToHost(forward_y, gpuSolver->y_host, gpuSolver->y_out);
    gpuSolver->copyDeviceToHost(forward_z, gpuSolver->z_host, gpuSolver->z_out);
}

void MapperBase::updateMapping(const buffer3Df &un, const buffer3Df &vn, const buffer3Df &wn, float cfldt, float dt, int do_SF_DMC)
{
    // copy velocity buffer from host to device
    gpuSolver->copyHostToDevice(un, gpuSolver->u_host, gpuSolver->u, (_ni+1)*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(vn, gpuSolver->v_host, gpuSolver->v, _ni*(_nj+1)*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(wn, gpuSolver->w_host, gpuSolver->w, _ni*_nj*(_nk+1)*sizeof(float));
    // update forward and backward mapping
    // forward mapping will be stored in GPU.x_out, GPU.y_out, GPU.z_out
    // backward mapping will be stored in GPU.x_in, GPU,y_in, GPU.z_in
    // NOTE: ORDER MUST NOT CHANGE DUE TO REUSE OF GPU BUFFER
    updateBackward(cfldt, dt, do_SF_DMC);
    updateForward(cfldt, dt);
}

void MapperBase::accumulateVelocityCovector(buffer3Df& u_init, buffer3Df& v_init, buffer3Df& w_init,
                                           const buffer3Df& u_change, const buffer3Df& v_change, const buffer3Df& w_change,
                                           const buffer3Df& un, const buffer3Df& vn, const buffer3Df& wn,
                                           float coeff, bool do_anitaliasing, float cfldt, float dt, bool do_true_mid_vel, bool do_SCPF)
{
    gpuSolver->copyHostToDevice(forward_x, gpuSolver->x_host, gpuSolver->x_out, _ni * _nj * _nk * sizeof(float));
    gpuSolver->copyHostToDevice(forward_y, gpuSolver->y_host, gpuSolver->y_out, _ni * _nj * _nk * sizeof(float));
    gpuSolver->copyHostToDevice(forward_z, gpuSolver->z_host, gpuSolver->z_out, _ni * _nj * _nk * sizeof(float));
    // copy accumulated velocity change at initial state to gpu.du, gpu.dv, gpu.dw
    gpuSolver->copyHostToDevice(u_init, gpuSolver->u_host, gpuSolver->du, (_ni + 1) * _nj * _nk * sizeof(float));
    gpuSolver->copyHostToDevice(v_init, gpuSolver->v_host, gpuSolver->dv, _ni * (_nj + 1) * _nk * sizeof(float));
    gpuSolver->copyHostToDevice(w_init, gpuSolver->w_host, gpuSolver->dw, _ni * _nj * (_nk + 1) * sizeof(float));
    gpuSolver->copyHostToDevice(un, gpuSolver->u_host, gpuSolver->u_tmp, (_ni + 1) * _nj * _nk * sizeof(float));
    gpuSolver->copyHostToDevice(vn, gpuSolver->v_host, gpuSolver->v_tmp, _ni * (_nj + 1) * _nk * sizeof(float));
    gpuSolver->copyHostToDevice(wn, gpuSolver->w_host, gpuSolver->w_tmp, _ni * _nj * (_nk + 1) * sizeof(float));
    // copy velocity change (e.g. buoyancy, projection) at time t to gpu.u, gpu.v, gpu.w
    gpuSolver->copyHostToDevice(u_change, gpuSolver->u_host, gpuSolver->u, (_ni + 1) * _nj * _nk * sizeof(float));
    gpuSolver->copyHostToDevice(v_change, gpuSolver->v_host, gpuSolver->v, _ni * (_nj + 1) * _nk * sizeof(float));
    gpuSolver->copyHostToDevice(w_change, gpuSolver->w_host, gpuSolver->w, _ni * _nj * (_nk + 1) * sizeof(float));
    // now add velocity change multiply with coefficient back to initial state, which is stored in gpu.du, gpu.dv, gpu.dw
    gpuSolver->accumulateVelocityCovector(!do_anitaliasing, coeff, cfldt, dt, do_true_mid_vel, do_SCPF);
    // copy accumulated velocity back to host buffer
    gpuSolver->copyDeviceToHost(u_init, gpuSolver->u_host, gpuSolver->du);
    gpuSolver->copyDeviceToHost(v_init, gpuSolver->v_host, gpuSolver->dv);
    gpuSolver->copyDeviceToHost(w_init, gpuSolver->w_host, gpuSolver->dw);
}

void MapperBase::accumulateVelocity(buffer3Df &u_init, buffer3Df &v_init, buffer3Df &w_init,
                                    const buffer3Df &u_change, const buffer3Df &v_change, const buffer3Df &w_change,
                                    float coeff, bool do_anitaliasing)
{
    gpuSolver->copyHostToDevice(forward_x, gpuSolver->x_host, gpuSolver->x_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(forward_y, gpuSolver->y_host, gpuSolver->y_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(forward_z, gpuSolver->z_host, gpuSolver->z_out, _ni*_nj*_nk*sizeof(float));
    // copy accumulated velocity change at initial state to gpu.du, gpu.dv, gpu.dw
    gpuSolver->copyHostToDevice(u_init, gpuSolver->u_host, gpuSolver->du, (_ni+1)*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(v_init, gpuSolver->v_host, gpuSolver->dv, _ni*(_nj+1)*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(w_init, gpuSolver->w_host, gpuSolver->dw, _ni*_nj*(_nk+1)*sizeof(float));
    // copy velocity change (e.g. buoyancy, projection) at time t to gpu.u, gpu.v, gpu.w
    gpuSolver->copyHostToDevice(u_change, gpuSolver->u_host, gpuSolver->u, (_ni+1)*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(v_change, gpuSolver->v_host, gpuSolver->v, _ni*(_nj+1)*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(w_change, gpuSolver->w_host, gpuSolver->w, _ni*_nj*(_nk+1)*sizeof(float));
    // now add velocity change multiply with coefficient back to initial state, which is stored in gpu.du, gpu.dv, gpu.dw
    gpuSolver->accumulateVelocity(!do_anitaliasing, coeff);
    // copy accumulated velocity back to host buffer
    gpuSolver->copyDeviceToHost(u_init, gpuSolver->u_host, gpuSolver->du);
    gpuSolver->copyDeviceToHost(v_init, gpuSolver->v_host, gpuSolver->dv);
    gpuSolver->copyDeviceToHost(w_init, gpuSolver->w_host, gpuSolver->dw);
}

void MapperBase::accumulateField(buffer3Df &field_init, const buffer3Df &field_change, bool do_anitaliasing)
{
    gpuSolver->copyHostToDevice(forward_x, gpuSolver->x_host, gpuSolver->x_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(forward_y, gpuSolver->y_host, gpuSolver->y_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(forward_z, gpuSolver->z_host, gpuSolver->z_out, _ni*_nj*_nk*sizeof(float));
    // reuse gpu.u, gpu.du to save GPU buffer
    // copy accumulated field change at initial state to gpu.du
    gpuSolver->copyHostToDevice(field_init, gpuSolver->x_host, gpuSolver->du, (_ni+1)*_nj*_nk*sizeof(float));
    // copy field change from external(e.g. source emitter) at time t to gpu.u, gpu.v, gpu.w
    gpuSolver->copyHostToDevice(field_change, gpuSolver->x_host, gpuSolver->u, (_ni+1)*_nj*_nk*sizeof(float));
    // now add field change back to initial state, which is stored in gpu.du
    gpuSolver->accumulateField(!do_anitaliasing, 1.f);
    // copy accumulated field back to host buffer
    gpuSolver->copyDeviceToHost(field_init, gpuSolver->u_host, gpuSolver->du);
}

float MapperBase::estimateDistortion(const buffer3Dc &boundary)
{
    gpuSolver->copyHostToDevice(forward_x, gpuSolver->x_host, gpuSolver->x_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(forward_y, gpuSolver->y_host, gpuSolver->y_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(forward_z, gpuSolver->z_host, gpuSolver->z_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(backward_x, gpuSolver->x_host, gpuSolver->x_in, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(backward_y, gpuSolver->y_host, gpuSolver->y_in, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(backward_z, gpuSolver->z_host, gpuSolver->z_in, _ni*_nj*_nk*sizeof(float));
    gpuSolver->estimateDistortionCUDA();
    cudaMemcpy(gpuSolver->u_host, gpuSolver->du, (_ni+1)*_nj*_nk*sizeof(float), cudaMemcpyDeviceToHost);
    float max_dist = 0.f;
    for (uint i = 0; i < forward_x._nx; i++)
    {
        for (uint j = 0; j < forward_x._ny; j++)
        {
            for (uint k = 0; k < forward_x._nz; k++)
            {
                if (boundary(i,j,k) != 2)
                {
                    uint index = i + forward_x._nx*j + forward_x._nx * forward_x._ny * k;
                    max_dist = max(max_dist, gpuSolver->u_host[index]);
                }
            }
        }
    }
    return sqrt(max_dist);
}

void MapperBase::init(uint ni, uint nj, uint nk, float h, float coeff, gpuMapper *mymapper)
{
    _ni = ni;
    _nj = nj;
    _nk = nk;
    _h = h;
    blend_coeff = coeff;
    total_reinit_count = 0;
    forward_x.init(ni, nj, nk, h, 0.0, 0.0, 0.0);
    forward_y.init(ni, nj, nk, h, 0.0, 0.0, 0.0);
    forward_z.init(ni, nj, nk, h, 0.0, 0.0, 0.0);
    backward_x.init(ni, nj, nk, h, 0.0, 0.0, 0.0);
    backward_y.init(ni, nj, nk, h, 0.0, 0.0, 0.0);
    backward_z.init(ni, nj, nk, h, 0.0, 0.0, 0.0);
    backward_xprev.init(ni, nj, nk, h, 0.0, 0.0, 0.0);
    backward_yprev.init(ni, nj, nk, h, 0.0, 0.0, 0.0);
    backward_zprev.init(ni, nj, nk, h, 0.0, 0.0, 0.0);
    forward_temp_x.init(ni, nj, nk, h, 0.0, 0.0, 0.0);
    forward_temp_y.init(ni, nj, nk, h, 0.0, 0.0, 0.0);
    forward_temp_z.init(ni, nj, nk, h, 0.0, 0.0, 0.0);
    backward_temp_x.init(ni, nj, nk, h, 0.0, 0.0, 0.0);
    backward_temp_y.init(ni, nj, nk, h, 0.0, 0.0, 0.0);
    backward_temp_z.init(ni, nj, nk, h, 0.0, 0.0, 0.0);
    gpuSolver = mymapper;
    int compute_elements = forward_x._blockx*forward_x._blocky*forward_x._blockz;
    int slice = forward_x._blockx*forward_x._blocky;

    tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

        uint bk = thread_idx/slice;
        uint bj = (thread_idx%slice)/forward_x._blockx;
        uint bi = thread_idx%(forward_x._blockx);

        for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
                {
                    uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
                    if(i<forward_x._nx && j<forward_x._ny && k<forward_x._nz)
                    {
                        float world_x = ((float)i-forward_x._ox)*_h;
                        float world_y = ((float)j-forward_x._oy)*_h;
                        float world_z = ((float)k-forward_x._oz)*_h;
                        forward_x(i,j,k) = world_x;
                        forward_y(i,j,k) = world_y;
                        forward_z(i,j,k) = world_z;
                        backward_x(i,j,k) = world_x;
                        backward_y(i,j,k) = world_y;
                        backward_z(i,j,k) = world_z;
                    }
                }
    });
    backward_xprev.copy(backward_x);
    backward_yprev.copy(backward_y);
    backward_zprev.copy(backward_z);
}

void MapperBase::advectVelocityCovector(buffer3Df& un, buffer3Df& vn, buffer3Df& wn,
                                       const buffer3Df& u_init, const buffer3Df& v_init, const buffer3Df& w_init,
                                       const buffer3Df& u_prev, const buffer3Df& v_prev, const buffer3Df& w_prev,
                                       bool do_EC, bool do_EC_with_clamp, bool do_anitaliasing, float cfldt, float dt, bool do_true_mid_vel, bool do_SCPF)
{
    // forward and backward mapping are already in gpu.x_out and gpu.x_in correspondingly
    gpuSolver->copyHostToDevice(forward_x, gpuSolver->x_host, gpuSolver->x_out, _ni * _nj * _nk * sizeof(float));
    gpuSolver->copyHostToDevice(forward_y, gpuSolver->y_host, gpuSolver->y_out, _ni * _nj * _nk * sizeof(float));
    gpuSolver->copyHostToDevice(forward_z, gpuSolver->z_host, gpuSolver->z_out, _ni * _nj * _nk * sizeof(float));
    gpuSolver->copyHostToDevice(backward_x, gpuSolver->x_host, gpuSolver->x_in, _ni * _nj * _nk * sizeof(float));
    gpuSolver->copyHostToDevice(backward_y, gpuSolver->y_host, gpuSolver->y_in, _ni * _nj * _nk * sizeof(float));
    gpuSolver->copyHostToDevice(backward_z, gpuSolver->z_host, gpuSolver->z_in, _ni * _nj * _nk * sizeof(float));
    // ready for advection and compensation
    // init velocity buffer will be stored in gpu.du, gpu.dv, gpu.dw
    gpuSolver->copyHostToDevice(u_init, gpuSolver->u_host, gpuSolver->du, (_ni + 1) * _nj * _nk * sizeof(float));
    gpuSolver->copyHostToDevice(v_init, gpuSolver->v_host, gpuSolver->dv, _ni * (_nj + 1) * _nk * sizeof(float));
    gpuSolver->copyHostToDevice(w_init, gpuSolver->w_host, gpuSolver->dw, _ni * _nj * (_nk + 1) * sizeof(float));
    // current velocity buffer will be stored in gpu.u_tmp, gpu.v_tmp, gpu.w_tmp
    gpuSolver->copyHostToDevice(un, gpuSolver->u_host, gpuSolver->u_tmp, (_ni + 1) * _nj * _nk * sizeof(float));
    gpuSolver->copyHostToDevice(vn, gpuSolver->v_host, gpuSolver->v_tmp, _ni * (_nj + 1) * _nk * sizeof(float));
    gpuSolver->copyHostToDevice(wn, gpuSolver->w_host, gpuSolver->w_tmp, _ni * _nj * (_nk + 1) * sizeof(float));
    // updated velocity(no compensation) will be store in gpu.u, gpu.v, gpu.w
    gpuSolver->advectVelocityCovector(!do_anitaliasing, cfldt, dt, do_true_mid_vel, do_SCPF);
    // compensated and clamped velocity will be stored in gpu.u, gpu.v, gpu.w
    if (do_EC)
        gpuSolver->compensateVelocityCovector(!do_anitaliasing, do_EC_with_clamp, cfldt, dt, do_true_mid_vel, do_SCPF);
    // copy velocity back to host
    gpuSolver->copyDeviceToHost(un, gpuSolver->u_host, gpuSolver->u);
    gpuSolver->copyDeviceToHost(vn, gpuSolver->v_host, gpuSolver->v);
    gpuSolver->copyDeviceToHost(wn, gpuSolver->w_host, gpuSolver->w);
}

void MapperBase::advectVelocity(buffer3Df &un, buffer3Df &vn, buffer3Df &wn,
                                const buffer3Df &u_init, const buffer3Df &v_init, const buffer3Df &w_init,
                                const buffer3Df &u_prev, const buffer3Df &v_prev, const buffer3Df &w_prev,
                                bool do_EC, bool do_EC_with_clamp, bool do_anitaliasing)
{
    // forward and backward mapping are already in gpu.x_out and gpu.x_in correspondingly
    gpuSolver->copyHostToDevice(forward_x, gpuSolver->x_host, gpuSolver->x_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(forward_y, gpuSolver->y_host, gpuSolver->y_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(forward_z, gpuSolver->z_host, gpuSolver->z_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(backward_x, gpuSolver->x_host, gpuSolver->x_in, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(backward_y, gpuSolver->y_host, gpuSolver->y_in, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(backward_z, gpuSolver->z_host, gpuSolver->z_in, _ni*_nj*_nk*sizeof(float));
    // ready for advection and compensation
    // init velocity buffer will be stored in gpu.du, gpu.dv, gpu.dw
    gpuSolver->copyHostToDevice(u_init, gpuSolver->u_host, gpuSolver->du, (_ni+1)*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(v_init, gpuSolver->v_host, gpuSolver->dv, _ni*(_nj+1)*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(w_init, gpuSolver->w_host, gpuSolver->dw, _ni*_nj*(_nk+1)*sizeof(float));
    // updated velocity(no compensation) will be store in gpu.u, gpu.v, gpu.w
    gpuSolver->advectVelocity(!do_anitaliasing);
    // compensated and clamped velocity will be stored in gpu.u, gpu.v, gpu.w
    if (do_EC)
        gpuSolver->compensateVelocity(!do_anitaliasing, do_EC_with_clamp);
    // now copy backward_prev to gpu.x_out, gpu._yout, gpu.z_out
    gpuSolver->copyHostToDevice(backward_xprev, gpuSolver->x_host, gpuSolver->x_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(backward_yprev, gpuSolver->y_host, gpuSolver->y_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(backward_zprev, gpuSolver->z_host, gpuSolver->z_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(u_prev, gpuSolver->u_host, gpuSolver->du, (_ni+1)*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(v_prev, gpuSolver->v_host, gpuSolver->dv, _ni*(_nj+1)*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(w_prev, gpuSolver->w_host, gpuSolver->dw, _ni*_nj*(_nk+1)*sizeof(float));
    // velocity from backward_prev will be blended with velocity from backward_curr
    // if reinitialization is not happen, prev buffers are not valid
    if (total_reinit_count != 0)
        gpuSolver->advectVelocityDouble(!do_anitaliasing, blend_coeff);
    else
        gpuSolver->advectVelocityDouble(!do_anitaliasing, 1.f);
    // copy velocity back to host
    gpuSolver->copyDeviceToHost(un, gpuSolver->u_host, gpuSolver->u);
    gpuSolver->copyDeviceToHost(vn, gpuSolver->v_host, gpuSolver->v);
    gpuSolver->copyDeviceToHost(wn, gpuSolver->w_host, gpuSolver->w);
}

void MapperBase::advectFieldCovector(buffer3Df& field, const buffer3Df& field_init, const buffer3Df& field_prev,
                                    bool do_EC, bool do_anitaliasing)
{
    gpuSolver->copyHostToDevice(forward_x, gpuSolver->x_host, gpuSolver->x_out, _ni * _nj * _nk * sizeof(float));
    gpuSolver->copyHostToDevice(forward_y, gpuSolver->y_host, gpuSolver->y_out, _ni * _nj * _nk * sizeof(float));
    gpuSolver->copyHostToDevice(forward_z, gpuSolver->z_host, gpuSolver->z_out, _ni * _nj * _nk * sizeof(float));
    gpuSolver->copyHostToDevice(backward_x, gpuSolver->x_host, gpuSolver->x_in, _ni * _nj * _nk * sizeof(float));
    gpuSolver->copyHostToDevice(backward_y, gpuSolver->y_host, gpuSolver->y_in, _ni * _nj * _nk * sizeof(float));
    gpuSolver->copyHostToDevice(backward_z, gpuSolver->z_host, gpuSolver->z_in, _ni * _nj * _nk * sizeof(float));
    // reuse gpu.dfield for other fields(e.g. density, temperature)
    gpuSolver->copyHostToDevice(field_init, gpuSolver->x_host, gpuSolver->dfield, _ni * _nj * _nk * sizeof(float));
    // updated field will be stored in gpu.field
    gpuSolver->advectFieldCovector(!do_anitaliasing);
    // compensated field will be stored in gpu.field
    if (do_EC)
        gpuSolver->compensateFieldCovector(!do_anitaliasing);
    // copy field back to host
    gpuSolver->copyDeviceToHost(field, gpuSolver->x_host, gpuSolver->field);
}

void MapperBase::advectField(buffer3Df &field, const buffer3Df &field_init, const buffer3Df &field_prev,
                             bool do_EC, bool do_anitaliasing)
{
    gpuSolver->copyHostToDevice(forward_x, gpuSolver->x_host, gpuSolver->x_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(forward_y, gpuSolver->y_host, gpuSolver->y_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(forward_z, gpuSolver->z_host, gpuSolver->z_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(backward_x, gpuSolver->x_host, gpuSolver->x_in, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(backward_y, gpuSolver->y_host, gpuSolver->y_in, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(backward_z, gpuSolver->z_host, gpuSolver->z_in, _ni*_nj*_nk*sizeof(float));
    // reuse gpu.dfield for other fields(e.g. density, temperature)
    gpuSolver->copyHostToDevice(field_init, gpuSolver->x_host, gpuSolver->dfield, _ni*_nj*_nk*sizeof(float));
    // updated field will be stored in gpu.field
    gpuSolver->advectField(!do_anitaliasing);
    // compensated field will be stored in gpu.field
    if (do_EC)
        gpuSolver->compensateField(!do_anitaliasing);
    // now copy backward_prev to gpu.x_out, gpu._yout, gpu.z_out
    gpuSolver->copyHostToDevice(backward_xprev, gpuSolver->x_host, gpuSolver->x_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(backward_yprev, gpuSolver->y_host, gpuSolver->y_out, _ni*_nj*_nk*sizeof(float));
    gpuSolver->copyHostToDevice(backward_zprev, gpuSolver->z_host, gpuSolver->z_out, _ni*_nj*_nk*sizeof(float));
    //copy field_prev to gpu.dfield
    gpuSolver->copyHostToDevice(field_prev, gpuSolver->x_host, gpuSolver->dfield, _ni*_nj*_nk*sizeof(float));
    // field from backward_prev will be blended with field from backward_curr
    // if reinitialization is not happen, prev buffers are not valid
    if (total_reinit_count != 0)
        gpuSolver->advectFieldDouble(!do_anitaliasing, blend_coeff);
    else
        gpuSolver->advectFieldDouble(!do_anitaliasing, 1.f);
    // copy field back to host
    gpuSolver->copyDeviceToHost(field, gpuSolver->x_host, gpuSolver->field);
}

void MapperBase::reinitializeMapping()
{
    total_reinit_count ++;
    backward_xprev.copy(backward_x);
    backward_yprev.copy(backward_y);
    backward_zprev.copy(backward_z);

    int compute_elements = forward_x._blockx*forward_x._blocky*forward_x._blockz;
    int slice = forward_x._blockx*forward_x._blocky;

    tbb::parallel_for(0, compute_elements, 1, [&](int thread_idx) {

        uint bk = thread_idx/slice;
        uint bj = (thread_idx%slice)/forward_x._blockx;
        uint bi = thread_idx%(forward_x._blockx);

        for (uint kk=0;kk<8;kk++)for(uint jj=0;jj<8;jj++)for(uint ii=0;ii<8;ii++)
        {
            uint i=bi*8+ii, j=bj*8+jj, k=bk*8+kk;
            if(i<forward_x._nx && j<forward_x._ny && k<forward_x._nz)
            {
                float world_x = ((float)i-forward_x._ox)*_h;
                float world_y = ((float)j-forward_x._oy)*_h;
                float world_z = ((float)k-forward_x._oz)*_h;
                forward_x(i,j,k) = world_x;
                forward_y(i,j,k) = world_y;
                forward_z(i,j,k) = world_z;
                backward_x(i,j,k) = world_x;
                backward_y(i,j,k) = world_y;
                backward_z(i,j,k) = world_z;
            }
        }
    });
}