#include <cuda_runtime.h>
#include <cuda_occupancy.h>
#include <cuda_profiler_api.h>
#include <helper_cuda.h>
#include "GPU_Advection.h"
#include <iostream>


__constant__ float MIXING_AA_ALPHA = 0.9f;
__constant__ int PAD_BACK_TO_SF = -1;

__device__ float3 operator+(const float3& a, const float3& b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator-(const float3& a, const float3& b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 operator*(const float& a, const float3& b)
{
    return make_float3(a * b.x, a * b.y, a * b.z);
}

__device__ float3 operator*(const float3& a, const float& b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ float dot(const float3& a, const float3& b)
{
    return (a.x * b.x + a.y * b.y + a.z * b.z);
}

__device__ float clamp(float a, float minv, float maxv)
{
    return fminf(fmaxf(minv, a),maxv);
}

__device__ float3 clampv3(const float3& in, const float3& minv, const float3& maxv)
{
    float xout = clamp(in.x,minv.x,maxv.x);
    float yout = clamp(in.y,minv.y,maxv.y);
    float zout = clamp(in.z,minv.z,maxv.z);
    return make_float3(xout, yout, zout);
}

__device__ float lerp(float a, float b, float c)
{
    return (1.0-c)*a + c*b;
}

__device__ float triLerp(float v000, float v001, float v010, float v011, float v100, float v101,
        float v110, float v111, float a, float b, float c)
{
    return lerp(
            lerp(
                    lerp(v000, v001, a),
                    lerp(v010, v011, a),
                    b),
            lerp(
                    lerp(v100, v101, a),
                    lerp(v110, v111, a),
                    b),
            c);

}

__device__ float sample_buffer(float * b, int nx, int ny, int nz, float h, const float3& off_set, const float3& pos)
{
    float3 samplepos = make_float3(pos.x-off_set.x, pos.y-off_set.y, pos.z-off_set.z);
    samplepos = make_float3(fmaxf(0.f, samplepos.x), fmaxf(0.f, samplepos.y), fmaxf(0.f, samplepos.z));

    int i = int(floorf(samplepos.x/h));
    int j = int(floorf(samplepos.y/h));
    int k = int(floorf(samplepos.z/h));
    float fx = samplepos.x/h - float(i);
    float fy = samplepos.y/h - float(j);
    float fz = samplepos.z/h - float(k);

    int ip = (int)fminf(nx - 1, i + 1);
    int jp = (int)fminf(ny - 1, j + 1);
    int kp = (int)fminf(nz - 1, k + 1);
    int idx000 = i  + nx*j  + nx*ny*k;
    int idx001 = ip + nx*j  + nx*ny*k;
    int idx010 = i  + nx*jp + nx*ny*k;
    int idx011 = ip + nx*jp + nx*ny*k;
    int idx100 = i  + nx*j  + nx*ny*kp;
    int idx101 = ip + nx*j  + nx*ny*kp;
    int idx110 = i  + nx*jp + nx*ny*kp;
    int idx111 = ip + nx*jp + nx*ny*kp;
    return triLerp(b[idx000],b[idx001],b[idx010],b[idx011],
                   b[idx100],b[idx101],b[idx110],b[idx111],
                   fx, fy, fz);
}

__device__ float3 getVelocity(float *u, float *v, float *w, float h, int nx, int ny, int nz, float3 pos)
{
    float clampPad = 0.5f * h;
    float3 lower_limit = make_float3(0.f + clampPad, 0.f + clampPad, 0.f + clampPad);
    float3 upper_limit = make_float3(float(nx-1) * h - clampPad, float(ny-1) * h - clampPad, float(nz-1) * h - clampPad);
    pos = clampv3(pos, lower_limit, upper_limit);

    float _u = sample_buffer(u, nx+1, ny, nz, h, make_float3(-0.5*h,0,0), pos);
    float _v = sample_buffer(v, nx, ny+1, nz, h, make_float3(0,-0.5*h,0), pos);
    float _w = sample_buffer(w, nx, ny, nz+1, h, make_float3(0,0,-0.5*h), pos);

    return make_float3(_u,_v,_w);
}

__device__ float getFieldSafe(float* field, float h, int nx, int ny, int nz, int dimx, int dimy, int dimz, float3 pos)
{
    float clampPad = 0.5f * h;
    float3 lower_limit = make_float3(0.f + clampPad, 0.f + clampPad, 0.f + clampPad);
    float3 upper_limit = make_float3(float(nx - 1) * h - clampPad, float(ny - 1) * h - clampPad, float(nz - 1) * h - clampPad);
    pos = clampv3(pos, lower_limit, upper_limit);

    float read_field = sample_buffer(field, nx + dimx, ny + dimy, nz + dimz, h, make_float3(-0.5 * (float)dimx * h, -0.5 * (float)dimy * h, -0.5 * (float)dimz * h), pos);

    return read_field;
}

__device__ float3 getPosition(float* map_x, float* map_y, float* map_z, float h, int nx, int ny, int nz, const float3& pos)
{
    float pos_x = sample_buffer(map_x, nx, ny, nz, h, make_float3(0, 0, 0), pos);
    float pos_y = sample_buffer(map_y, nx, ny, nz, h, make_float3(0, 0, 0), pos);
    float pos_z = sample_buffer(map_z, nx, ny, nz, h, make_float3(0, 0, 0), pos);

    return make_float3(pos_x, pos_y, pos_z);
}

__device__ float3 traceRK3(float *u, float *v, float *w, float h, int ni, int nj, int nk, float dt, const float3& pos)
{
    float c1 = 2.0/9.0*dt, c2 = 3.0/9.0 * dt, c3 = 4.0/9.0 * dt;
    float3 input = pos;
    float3 v1 = getVelocity(u,v,w,h,ni,nj,nk, input);
    float3 midp1 = make_float3(input.x + 0.5*dt*v1.x, input.y + 0.5*dt*v1.y, input.z + 0.5*dt*v1.z);
    float3 v2 = getVelocity(u,v,w,h,ni,nj,nk, midp1);
    float3 midp2 = make_float3(input.x + 0.75*dt*v2.x, input.y + 0.75*dt*v2.y, input.z + 0.75*dt*v2.z);
    float3 v3 = getVelocity(u,v,w,h,ni,nj,nk, midp2);

    float3 output = make_float3(input.x + c1*v1.x + c2*v2.x + c3*v3.x,
                                input.y + c1*v1.y + c2*v2.y + c3*v3.y,
                                input.z + c1*v1.z + c2*v2.z + c3*v3.z);
    
    float clampPad = 0.0f * h;
    float3 lower_limit = make_float3(0.f + clampPad, 0.f + clampPad, 0.f + clampPad);
    float3 upper_limit = make_float3(float(ni-1) * h - clampPad, float(nj-1) * h - clampPad, float(nk-1) * h - clampPad);
    output = clampv3(output, lower_limit, upper_limit);
    return output;
}

__device__ float3 traceRK4(float *u, float *v, float *w, float h, int ni, int nj, int nk, float dt, const float3& pos)
{
    float c1 = 1.0 / 6.0 * dt, c2 = 1.0 / 3.0 * dt, c3 = 1.0 / 3.0 * dt, c4 = 1.0 / 6.0 * dt;
    float3 input = pos;
    float3 v1 = getVelocity(u,v,w,h,ni,nj,nk, input);
    float3 midp1 = make_float3(input.x + 0.5*dt*v1.x, input.y + 0.5*dt*v1.y, input.z + 0.5*dt*v1.z);
    float3 v2 = getVelocity(u,v,w,h,ni,nj,nk, midp1);
    float3 midp2 = make_float3(input.x + 0.5*dt*v2.x, input.y + 0.5*dt*v2.y, input.z + 0.5*dt*v2.z);
    float3 v3 = getVelocity(u,v,w,h,ni,nj,nk, midp2);
    float3 midp3 = make_float3(input.x + dt*v3.x, input.y + dt*v3.y, input.z + dt*v3.z);
    float3 v4 = getVelocity(u,v,w,h,ni,nj,nk, midp3);

    float3 output = make_float3(input.x + c1*v1.x + c2*v2.x + c3*v3.x + c4*v4.x,
                                input.y + c1*v1.y + c2*v2.y + c3*v3.y + c4*v4.y,
                                input.z + c1*v1.z + c2*v2.z + c3*v3.z + c4*v4.z);

    float clampPad = 0.0f * h;
    float3 lower_limit = make_float3(0.f + clampPad, 0.f + clampPad, 0.f + clampPad);
    float3 upper_limit = make_float3(float(ni-1) * h - clampPad, float(nj-1) * h - clampPad, float(nk-1) * h - clampPad);
    output = clampv3(output, lower_limit, upper_limit);
    return output;
}

__device__ float3 trace(float *u, float *v, float *w, float h, int ni, int nj, int nk, float cfldt, float dt, const float3& pos)
{
    if(dt>0)
    {
        float T = dt;
        float3 opos = pos;
        float t = 0;
        float substep = cfldt;
        while(t<T)
        {
            if(t+substep > T)
                substep = T - t;
            opos = traceRK4(u,v,w,h,ni,nj,nk,substep,opos);

            t+=substep;
        }
        return opos;
    }
    else
    {
        float T = -dt;
        float3 opos = pos;
        float t = 0;
        float substep = cfldt;
        while(t<T)
        {
            if(t+substep > T)
                substep = T - t;
            opos = traceRK4(u,v,w,h,ni,nj,nk,-substep,opos);
            t+=substep;
        }
        return opos;
    }
}

__global__ void calc_grad_kernel(float *i_map,
                                 float *ix_grad_map, float *iy_grad_map, float *iz_grad_map,
                                 float h, int ni, int nj, int nk)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    int i = index%ni;
    int j = (index%(ni*nj))/ni;
    int k = index/(ni*nj);

    int xp = i == (ni - 1) ? 0 : 1; int xn = i == 0 ? 0 : -1;
    int yp = j == (nj - 1) ? 0 : 1; int yn = j == 0 ? 0 : -1;
    int zp = k == (nk - 1) ? 0 : 1; int zn = k == 0 ? 0 : -1;
    float3 pos_xp = make_float3(float(i+xp)*h,   float(j)*h,      float(k)*h);
    float3 pos_xn = make_float3(float(i+xn)*h,   float(j)*h,      float(k)*h);
    float3 pos_yp = make_float3(float(i)*h,      float(j+yp)*h,   float(k)*h);
    float3 pos_yn = make_float3(float(i)*h,      float(j+yn)*h,   float(k)*h);
    float3 pos_zp = make_float3(float(i)*h,      float(j)*h,      float(k+zp)*h);
    float3 pos_zn = make_float3(float(i)*h,      float(j)*h,      float(k+zn)*h);
    float alpha_x = (i == 0 || i == (ni - 1)) ? 1 : 2;
    float alpha_y = (j == 0 || j == (nj - 1)) ? 1 : 2;
    float alpha_z = (k == 0 || k == (nk - 1)) ? 1 : 2;

    ix_grad_map[index] = (sample_buffer(i_map, ni, nj, nk, h, make_float3(0,0,0), pos_xp) - sample_buffer(i_map, ni, nj, nk, h, make_float3(0,0,0), pos_xn)) / (alpha_x*h);
    iy_grad_map[index] = (sample_buffer(i_map, ni, nj, nk, h, make_float3(0,0,0), pos_yp) - sample_buffer(i_map, ni, nj, nk, h, make_float3(0,0,0), pos_yn)) / (alpha_y*h);
    iz_grad_map[index] = (sample_buffer(i_map, ni, nj, nk, h, make_float3(0,0,0), pos_zp) - sample_buffer(i_map, ni, nj, nk, h, make_float3(0,0,0), pos_zn)) / (alpha_z*h);    
    __syncthreads();
}

__global__ void forward_kernel(float *u, float *v, float *w,
                               float *x_fwd, float *y_fwd, float *z_fwd,
                               float h, int ni, int nj, int nk, float cfldt, float dt)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    int i = index%ni;
    int j = (index%(ni*nj))/ni;
    int k = index/(ni*nj);

    float3 point = make_float3(x_fwd[index], y_fwd[index], z_fwd[index]);
    float3 pointout = trace(u,v,w,h,ni,nj,nk,cfldt,dt,point);
    x_fwd[index] = pointout.x;
    y_fwd[index] = pointout.y;
    z_fwd[index] = pointout.z;

    __syncthreads();
}

__global__ void clampExtrema_covector_kernel(float *before, float *after, int ni, int nj, int nk)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    int i = index%ni;
    int j = (index%(ni*nj))/ni;
    int k = index/(ni*nj);
    float max_value = before[index];
    float min_value = before[index];

    for(int kk=k-1;kk<=k+1;kk++)for(int jj=j-1;jj<=j+1;jj++)for(int ii=i-1;ii<=i+1;ii++)
    {
        int idx = ii + jj*ni + kk*ni*nj;
        if (ii >= 0 && ii <= ni - 1 && jj >= 0 && jj <= nj - 1 && kk >= 0 && kk <= nk - 1)
        {
            float before_val = before[idx];
            if (before_val > max_value)
                max_value = before_val;
            if (before_val < min_value)
                min_value = before_val;
        }
    }
    after[index] = min(max(min_value, after[index]), max_value);

    __syncthreads();
}

__global__ void clampExtrema_kernel(float *before, float *after, int ni, int nj, int nk)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    int i = index%ni;
    int j = (index%(ni*nj))/ni;
    int k = index/(ni*nj);
    float max_value = before[index];
    float min_value = before[index];

    for(int kk=k-1;kk<=k+1;kk++)for(int jj=j-1;jj<=j+1;jj++)for(int ii=i-1;ii<=i+1;ii++)
    {
        int idx = ii + jj*ni + kk*ni*nj;
        if (ii >= 0 && ii <= ni - 1 && jj >= 0 && jj <= nj - 1 && kk >= 0 && kk <= nk - 1)
        {
            float before_val = before[idx];
            if (before_val > max_value)
                max_value = before_val;
            if (before_val < min_value)
                min_value = before_val;
        }
    }
    after[index] = min(max(min_value, after[index]), max_value);

    __syncthreads();
}

__global__ void SF_backward_kernel(float *u, float *v, float *w,
                                   float *x_in, float *y_in, float *z_in,
                                   float *x_out, float *y_out, float *z_out,
                                   float h, int ni, int nj, int nk, float substep)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    int i = index%ni;
    int j = (index%(ni*nj))/ni;
    int k = index/(ni*nj);

    float3 point = make_float3(h*float(i), h*float(j), h*float(k));

    float3 pointnew = trace(u, v, w, h, ni, nj, nk, substep, -substep, point);

    float3 out_pos = getPosition(x_in, y_in, z_in, h, ni, nj, nk, pointnew);
    x_out[index] = out_pos.x;
    y_out[index] = out_pos.y;
    z_out[index] = out_pos.z;

    __syncthreads();
}

__global__ void DMC_backward_kernel(float *u, float *v, float *w,
                                    float *x_in, float *y_in, float *z_in,
                                    float *x_out, float *y_out, float *z_out,
                                    float h, int ni, int nj, int nk, float substep)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    int i = index%ni;
    int j = (index%(ni*nj))/ni;
    int k = index/(ni*nj);

    float3 point = make_float3(h*float(i),h*float(j),h*float(k));

    float3 vel = getVelocity(u, v, w, h, ni, nj, nk, point);

    float temp_x = (vel.x > 0)? point.x - h: point.x + h;
    float temp_y = (vel.y > 0)? point.y - h: point.y + h;
    float temp_z = (vel.z > 0)? point.z - h: point.z + h;
    float3 temp_point = make_float3(temp_x, temp_y, temp_z);
    float3 temp_vel = getVelocity(u, v, w, h, ni, nj, nk, temp_point);

    float a_x = (vel.x - temp_vel.x) / (point.x - temp_point.x);
    float a_y = (vel.y - temp_vel.y) / (point.y - temp_point.y);
    float a_z = (vel.z - temp_vel.z) / (point.z - temp_point.z);

    float new_x = (fabs(a_x) > 1e-4)? point.x - (1 - exp(-a_x*substep))*vel.x/a_x : point.x - vel.x*substep;
    float new_y = (fabs(a_y) > 1e-4)? point.y - (1 - exp(-a_y*substep))*vel.y/a_y : point.y - vel.y*substep;
    float new_z = (fabs(a_z) > 1e-4)? point.z - (1 - exp(-a_z*substep))*vel.z/a_z : point.z - vel.z*substep;
    float3 pointnew = make_float3(new_x, new_y, new_z);

    float clampPad = 0.f * h;
    float3 lower_limit = make_float3(0.f + clampPad, 0.f + clampPad, 0.f + clampPad);
    float3 upper_limit = make_float3(float(ni - 1) * h - clampPad, float(nj - 1) * h - clampPad, float(nk - 1) * h - clampPad);
    pointnew = clampv3(pointnew, lower_limit, upper_limit);
        
    float3 out_pos = getPosition(x_in, y_in, z_in, h, ni, nj, nk, pointnew);
    x_out[index] = out_pos.x;
    y_out[index] = out_pos.y;
    z_out[index] = out_pos.z;

    __syncthreads();
}

__global__ void semilag_kernel(float *field, float *field_src,
                               float *u, float *v, float *w,
                               int dimx, int dimy, int dimz,
                               float h, int ni, int nj, int nk, bool is_point, float cfldt, float dt)
{
    float3 volume[8];
    int evaluations = 8;
    volume[0] = make_float3(0.25f * h, 0.25f * h, 0.25f * h);  volume[1] = make_float3(0.25f * h, 0.25f * h, -0.25f * h);
    volume[2] = make_float3(0.25f * h, -0.25f * h, 0.25f * h); volume[3] = make_float3(0.25f * h, -0.25f * h, -0.25f * h);
    volume[4] = make_float3(-0.25f * h, 0.25f * h, 0.25f * h); volume[5] = make_float3(-0.25f * h, 0.25f * h, -0.25f * h);
    volume[6] = make_float3(-0.25f * h, -0.25f * h, 0.25f * h); volume[7] = make_float3(-0.25f * h, -0.25f * h, -0.25f * h);


    if (is_point) {
        volume[0] = make_float3(0, 0, 0);
        evaluations = 1;
    }


    int index = blockDim.x * blockIdx.x + threadIdx.x;
    float weight = 1.0 / float(evaluations);

    float3 buffer_origin = make_float3(-float(dimx)*0.5f*h, -float(dimy)*0.5f*h, -float(dimz)*0.5f*h);

    int field_buffer_i = ni + dimx;
    int field_buffer_j = nj + dimy;
    int field_buffer_k = nk + dimz;

    int i = index % field_buffer_i;
    int j = (index % (field_buffer_i * field_buffer_j)) / field_buffer_i;
    int k = index/(field_buffer_i*field_buffer_j);

    float sum = 0.0;
    for (int ii = 0; ii < evaluations; ii++)
    {
        float3 point = make_float3(h*float(i) + buffer_origin.x + volume[ii].x,
                                    h*float(j) + buffer_origin.y + volume[ii].y,
                                    h*float(k) + buffer_origin.z + volume[ii].z);

        float3 pointnew = trace(u, v, w, h, ni, nj, nk, cfldt, dt, point);

        sum += weight * getFieldSafe(field_src, h, ni, nj, nk, dimx, dimy, dimz, pointnew);
    }
    float3 point = make_float3(h*float(i) + buffer_origin.x,
                                h*float(j) + buffer_origin.y,
                                h*float(k) + buffer_origin.z);

    float3 pointnew = trace(u, v, w, h, ni, nj, nk, cfldt, dt, point);
    float value = getFieldSafe(field_src, h, ni, nj, nk, dimx, dimy, dimz, pointnew);

    field[index] = (1.f - MIXING_AA_ALPHA) * sum + MIXING_AA_ALPHA * value;

    __syncthreads();
}


__global__ void doubleAdvect_kernel(float *field, float *temp_field,
                                    float *backward_x, float *backward_y, float * backward_z,
                                    float *backward_xprev, float *backward_yprev, float *backward_zprev,
                                    float h, int ni, int nj, int nk,
                                    int dimx, int dimy, int dimz, bool is_point, float blend_coeff)
{
    float3 volume[8];
    int evaluations = 8;
    volume[0] = make_float3(0.25f*h, 0.25f*h, 0.25f*h);  volume[1] = make_float3(0.25f*h, 0.25f*h, -0.25f*h);
    volume[2] = make_float3(0.25f*h, -0.25f*h, 0.25f*h); volume[3] = make_float3(0.25f*h, -0.25f*h, -0.25f*h);
    volume[4] = make_float3(-0.25f*h, 0.25f*h, 0.25f*h); volume[5] = make_float3(-0.25f*h, 0.25f*h, -0.25f*h);
    volume[6] = make_float3(-0.25f*h, -0.25f*h, 0.25f*h);volume[7] = make_float3(-0.25f*h, -0.25f*h, -0.25f*h);


    if(is_point) {
        volume[0] = make_float3(0, 0, 0);
        evaluations = 1;
    }


    int index = blockDim.x*blockIdx.x + threadIdx.x;
    float weight = 1.0/float(evaluations);

    float3 buffer_origin = make_float3(-float(dimx)*0.5f*h, -float(dimy)*0.5f*h, -float(dimz)*0.5f*h);

    int vel_buffer_i = ni + dimx;
    int vel_buffer_j = nj + dimy;
    int vel_buffer_k = nk + dimz;

    int i = index%vel_buffer_i;
    int j = (index%(vel_buffer_i*vel_buffer_j))/vel_buffer_i;
    int k = index/(vel_buffer_i*vel_buffer_j);

    float sum = 0.0;
    for (int ii = 0; ii<evaluations; ii++)
    {
        float3 pos = make_float3(float(i)*h + buffer_origin.x + volume[ii].x,
                                    float(j)*h + buffer_origin.y + volume[ii].y,
                                    float(k)*h + buffer_origin.z + volume[ii].z);
        float3 midpos = getPosition(backward_x, backward_y, backward_z, h, ni, nj, nk, pos);
        float3 finalpos = getPosition(backward_xprev, backward_yprev, backward_zprev, h, ni, nj, nk, midpos);
        sum += weight*getFieldSafe(temp_field, h, ni, nj, nk, dimx, dimy, dimz, finalpos);
    }
    float3 pos = make_float3(float(i)*h + buffer_origin.x,
                                float(j)*h + buffer_origin.y,
                                float(k)*h + buffer_origin.z);
    float3 midpos = getPosition(backward_x, backward_y, backward_z, h, ni, nj, nk, pos);
    float3 finalpos = getPosition(backward_xprev, backward_yprev, backward_zprev, h, ni, nj, nk, midpos);
    float value = getFieldSafe(temp_field, h, ni, nj, nk, dimx, dimy, dimz, finalpos);
    float prev_value = (1.f - MIXING_AA_ALPHA) * sum + MIXING_AA_ALPHA * value;
    field[index] = field[index]*blend_coeff + (1-blend_coeff)*prev_value;

    __syncthreads();
}

__global__ void advect_covector_kernel(float *field,
                                      float *field_init_u, float *field_init_v, float *field_init_w,
                                      float *u_adv, float *v_adv, float *w_adv,
                                      float *backward_x, float *backward_y, float *backward_z,
                                      float h, int ni, int nj, int nk,
                                      int dimx, int dimy, int dimz, bool is_point, float cfldt, float dt, bool do_true_mid_vel, bool do_SCPF)
{
    float3 volume[8];
    int evaluations = 8;
    volume[0] = make_float3(0.25f*h, 0.25f*h, 0.25f*h);  volume[1] = make_float3(0.25f*h, 0.25f*h, -0.25f*h);
    volume[2] = make_float3(0.25f*h, -0.25f*h, 0.25f*h); volume[3] = make_float3(0.25f*h, -0.25f*h, -0.25f*h);
    volume[4] = make_float3(-0.25f*h, 0.25f*h, 0.25f*h); volume[5] = make_float3(-0.25f*h, 0.25f*h, -0.25f*h);
    volume[6] = make_float3(-0.25f*h, -0.25f*h, 0.25f*h);volume[7] = make_float3(-0.25f*h, -0.25f*h, -0.25f*h);

    if(is_point) {
        volume[0] = make_float3(0, 0, 0);
        evaluations = 1;
    }

    int index = blockDim.x*blockIdx.x + threadIdx.x;
    float weight = 1.0/float(evaluations);

    float3 buffer_origin = make_float3(-float(dimx)*0.5f*h, -float(dimy)*0.5f*h, -float(dimz)*0.5f*h);

    int vel_buffer_i = ni + dimx;
    int vel_buffer_j = nj + dimy;
    int vel_buffer_k = nk + dimz;

    int i = index%vel_buffer_i;
    int j = (index%(vel_buffer_i*vel_buffer_j))/vel_buffer_i;
    int k = index/(vel_buffer_i*vel_buffer_j);

    float3 sum_vel = make_float3(0,0,0);
    for (int ii = 0; ii<evaluations; ii++)
    {
        float3 mid_pos = make_float3(float(i)*h + buffer_origin.x + volume[ii].x,
                                    float(j)*h + buffer_origin.y + volume[ii].y,
                                    float(k)*h + buffer_origin.z + volume[ii].z);

        if (do_SCPF)
        {
            float3 front_pos = make_float3(mid_pos.x - buffer_origin.x, mid_pos.y - buffer_origin.y, mid_pos.z - buffer_origin.z);
            float3 back_pos = make_float3(mid_pos.x + buffer_origin.x, mid_pos.y + buffer_origin.y, mid_pos.z + buffer_origin.z);

            float3 front_pos_init = getPosition(backward_x, backward_y, backward_z, h, ni, nj, nk, front_pos);
            float3 back_pos_init = getPosition(backward_x, backward_y, backward_z, h, ni, nj, nk, back_pos);

            float3 front_vel = getVelocity(field_init_u, field_init_v, field_init_w, h, ni, nj, nk, front_pos_init);
            float3 back_vel = getVelocity(field_init_u, field_init_v, field_init_w, h, ni, nj, nk, back_pos_init);

            float3 avg_vel = 0.5f * (front_vel + back_vel);
            sum_vel = sum_vel + (weight * avg_vel);
        }
        else
        {
            float3 mid_pos_init;
            if (do_true_mid_vel)
                mid_pos_init = trace(u_adv, v_adv, w_adv, h, ni, nj, nk, cfldt, dt, mid_pos);
            else
                mid_pos_init = getPosition(backward_x, backward_y, backward_z, h, ni, nj, nk, mid_pos);

            float3 mid_vel = getVelocity(field_init_u, field_init_v, field_init_w, h, ni, nj, nk, mid_pos_init);
            sum_vel = sum_vel + (weight * mid_vel);
        }
    }
    float3 mid_pos = make_float3(float(i)*h + buffer_origin.x,
                                float(j)*h + buffer_origin.y,
                                float(k)*h + buffer_origin.z);

    float3 front_pos = make_float3(mid_pos.x - buffer_origin.x, mid_pos.y - buffer_origin.y, mid_pos.z - buffer_origin.z);
    float3 back_pos = make_float3(mid_pos.x + buffer_origin.x, mid_pos.y + buffer_origin.y, mid_pos.z + buffer_origin.z);

    float3 front_pos_init = getPosition(backward_x, backward_y, backward_z, h, ni, nj, nk, front_pos);
    float3 back_pos_init = getPosition(backward_x, backward_y, backward_z, h, ni, nj, nk, back_pos);
    float3 mid_diff = front_pos_init - back_pos_init;
    if (i <= PAD_BACK_TO_SF || j <= PAD_BACK_TO_SF || k <= PAD_BACK_TO_SF ||
        i >= vel_buffer_i - 1 - PAD_BACK_TO_SF || j >= vel_buffer_j - 1 - PAD_BACK_TO_SF || k >= vel_buffer_k - 1 - PAD_BACK_TO_SF)
        mid_diff = make_float3((float)dimx * h, (float)dimy * h, (float)dimz * h);
    float aa_value = dot(mid_diff, sum_vel) / h;

    float main_value = 0.f;
    if (do_SCPF)
    {
        float3 front_vel = getVelocity(field_init_u, field_init_v, field_init_w, h, ni, nj, nk, front_pos_init);
        float3 back_vel = getVelocity(field_init_u, field_init_v, field_init_w, h, ni, nj, nk, back_pos_init);

        float3 avg_vel = 0.5f * (front_vel + back_vel);
        main_value = dot(mid_diff, avg_vel) / h;
    }
    else
    {
        float3 mid_pos_init;
        if (do_true_mid_vel)
            mid_pos_init = trace(u_adv, v_adv, w_adv, h, ni, nj, nk, cfldt, dt, mid_pos);
        else
            mid_pos_init = getPosition(backward_x, backward_y, backward_z, h, ni, nj, nk, mid_pos);

        float3 mid_vel = getVelocity(field_init_u, field_init_v, field_init_w, h, ni, nj, nk, mid_pos_init);
        main_value = dot(mid_diff, mid_vel) / h;
    }

    field[index] = (1.f - MIXING_AA_ALPHA) * aa_value + MIXING_AA_ALPHA * main_value;

    __syncthreads();
}

__global__ void advect_kernel(float *field, float *field_init,
                              float *backward_x, float *backward_y, float *backward_z,
                              float h, int ni, int nj, int nk,
                              int dimx, int dimy, int dimz, bool is_point)
{
    float3 volume[8];
    int evaluations = 8;
    volume[0] = make_float3(0.25f*h, 0.25f*h, 0.25f*h);  volume[1] = make_float3(0.25f*h, 0.25f*h, -0.25f*h);
    volume[2] = make_float3(0.25f*h, -0.25f*h, 0.25f*h); volume[3] = make_float3(0.25f*h, -0.25f*h, -0.25f*h);
    volume[4] = make_float3(-0.25f*h, 0.25f*h, 0.25f*h); volume[5] = make_float3(-0.25f*h, 0.25f*h, -0.25f*h);
    volume[6] = make_float3(-0.25f*h, -0.25f*h, 0.25f*h);volume[7] = make_float3(-0.25f*h, -0.25f*h, -0.25f*h);

    if(is_point) {
        volume[0] = make_float3(0, 0, 0);
        evaluations = 1;
    }

    int index = blockDim.x*blockIdx.x + threadIdx.x;
    float weight = 1.0/float(evaluations);

    float3 buffer_origin = make_float3(-float(dimx)*0.5f*h, -float(dimy)*0.5f*h, -float(dimz)*0.5f*h);

    int vel_buffer_i = ni + dimx;
    int vel_buffer_j = nj + dimy;
    int vel_buffer_k = nk + dimz;

    int i = index%vel_buffer_i;
    int j = (index%(vel_buffer_i*vel_buffer_j))/vel_buffer_i;
    int k = index/(vel_buffer_i*vel_buffer_j);

    float sum = 0.0;
    for (int ii = 0; ii<evaluations; ii++)
    {
        float3 pos = make_float3(float(i)*h + buffer_origin.x + volume[ii].x,
                                    float(j)*h + buffer_origin.y + volume[ii].y,
                                    float(k)*h + buffer_origin.z + volume[ii].z);

        float3 pos_init = getPosition(backward_x, backward_y, backward_z, h, ni, nj, nk, pos);

        sum += weight*getFieldSafe(field_init, h, ni, nj, nk, dimx, dimy, dimz, pos_init);
    }
    float3 pos = make_float3(float(i)*h + buffer_origin.x,
                                float(j)*h + buffer_origin.y,
                                float(k)*h + buffer_origin.z);

    float3 pos_init = getPosition(backward_x, backward_y, backward_z, h, ni, nj, nk, pos);

    float value = getFieldSafe(field_init, h, ni, nj, nk, dimx, dimy, dimz, pos_init);
    field[index] = (1.f - MIXING_AA_ALPHA) * sum + MIXING_AA_ALPHA * value;

    __syncthreads();
}

__global__ void cumulate_covector_kernel(float *dfield_u, float *dfield_v, float *dfield_w,
                                float *u_adv, float *v_adv, float *w_adv,
                                float *dfield_init,
                                float *x_map, float *y_map, float *z_map,
                                float h, int ni, int nj, int nk,
                                int dimx, int dimy, int dimz, bool is_point, float coeff, float cfldt, float dt, bool do_true_mid_vel, bool do_SCPF)
{
    float3 volume[8];
    int evaluations = 8;
    volume[0] = make_float3(0.25f*h, 0.25f*h, 0.25f*h);  volume[1] = make_float3(0.25f*h, 0.25f*h, -0.25f*h);
    volume[2] = make_float3(0.25f*h, -0.25f*h, 0.25f*h); volume[3] = make_float3(0.25f*h, -0.25f*h, -0.25f*h);
    volume[4] = make_float3(-0.25f*h, 0.25f*h, 0.25f*h); volume[5] = make_float3(-0.25f*h, 0.25f*h, -0.25f*h);
    volume[6] = make_float3(-0.25f*h, -0.25f*h, 0.25f*h);volume[7] = make_float3(-0.25f*h, -0.25f*h, -0.25f*h);

    if(is_point) {
        volume[0] = make_float3(0, 0, 0);
        evaluations = 1;
    }

    int index = blockDim.x*blockIdx.x + threadIdx.x;
    float weight = 1.0/float(evaluations);

    float3 buffer_origin = make_float3(-float(dimx)*0.5f*h, -float(dimy)*0.5f*h, -float(dimz)*0.5f*h);

    int vel_buffer_i = ni + dimx;
    int vel_buffer_j = nj + dimy;
    int vel_buffer_k = nk + dimz;

    int i = index%vel_buffer_i;
    int j = (index%(vel_buffer_i*vel_buffer_j))/vel_buffer_i;
    int k = index/(vel_buffer_i*vel_buffer_j);

        float3 sum_vel = make_float3(0, 0, 0);
        for (int ii = 0; ii<evaluations; ii++)
        {
            float3 mid_pos = make_float3(float(i)*h + buffer_origin.x + volume[ii].x,
                                         float(j)*h + buffer_origin.y + volume[ii].y,
                                         float(k)*h + buffer_origin.z + volume[ii].z);
            
            // forward mapping position also used in compensation
            if (do_SCPF)
            {
                float3 front_pos = make_float3(mid_pos.x - buffer_origin.x, mid_pos.y - buffer_origin.y, mid_pos.z - buffer_origin.z);
                float3 back_pos = make_float3(mid_pos.x + buffer_origin.x, mid_pos.y + buffer_origin.y, mid_pos.z + buffer_origin.z);

                float3 front_pos_init = getPosition(x_map, y_map, z_map, h, ni, nj, nk, front_pos);
                float3 back_pos_init = getPosition(x_map, y_map, z_map, h, ni, nj, nk, back_pos);

                float3 front_vel = getVelocity(dfield_u, dfield_v, dfield_w, h, ni, nj, nk, front_pos_init);
                float3 back_vel = getVelocity(dfield_u, dfield_v, dfield_w, h, ni, nj, nk, back_pos_init);

                float3 avg_vel = 0.5f * (front_vel + back_vel);
                sum_vel = sum_vel + (weight * coeff * avg_vel);
            }
            else
            {
                float3 mid_pos_init;
                if (do_true_mid_vel)
                    mid_pos_init = trace(u_adv, v_adv, w_adv, h, ni, nj, nk, cfldt, dt, mid_pos);
                else
                    mid_pos_init = getPosition(x_map, y_map, z_map, h, ni, nj, nk, mid_pos);

                float3 mid_vel = getVelocity(dfield_u, dfield_v, dfield_w, h, ni, nj, nk, mid_pos_init);
                sum_vel = sum_vel + (weight * coeff * mid_vel);
            }
        }
        float3 mid_pos = make_float3(float(i)*h + buffer_origin.x,
                                     float(j)*h + buffer_origin.y,
                                     float(k)*h + buffer_origin.z);
        // forward mapping position
        float3 front_pos = make_float3(mid_pos.x - buffer_origin.x, mid_pos.y - buffer_origin.y, mid_pos.z - buffer_origin.z);
        float3 back_pos = make_float3(mid_pos.x + buffer_origin.x, mid_pos.y + buffer_origin.y, mid_pos.z + buffer_origin.z);

        float3 front_pos_init = getPosition(x_map, y_map, z_map, h, ni, nj, nk, front_pos);
        float3 back_pos_init = getPosition(x_map, y_map, z_map, h, ni, nj, nk, back_pos);
        float3 mid_diff = front_pos_init - back_pos_init;
        if (i <= PAD_BACK_TO_SF || j <= PAD_BACK_TO_SF || k <= PAD_BACK_TO_SF ||
            i >= vel_buffer_i - 1 - PAD_BACK_TO_SF || j >= vel_buffer_j - 1 - PAD_BACK_TO_SF || k >= vel_buffer_k - 1 - PAD_BACK_TO_SF)
            mid_diff = make_float3((float)dimx * h, (float)dimy * h, (float)dimz * h);
        float aa_value = dot(mid_diff, sum_vel) / h;

        float main_value = 0.f;
        if (do_SCPF)
        {
            float3 front_vel = getVelocity(dfield_u, dfield_v, dfield_w, h, ni, nj, nk, front_pos_init);
            float3 back_vel = getVelocity(dfield_u, dfield_v, dfield_w, h, ni, nj, nk, back_pos_init);

            float3 avg_vel = 0.5f * (front_vel + back_vel);
            main_value = coeff * dot(mid_diff, avg_vel) / h;
        }
        else
        {
            float3 mid_pos_init;
            if (do_true_mid_vel)
                mid_pos_init = trace(u_adv, v_adv, w_adv, h, ni, nj, nk, cfldt, dt, mid_pos);
            else
                mid_pos_init = getPosition(x_map, y_map, z_map, h, ni, nj, nk, mid_pos);

            float3 mid_vel = getVelocity(dfield_u, dfield_v, dfield_w, h, ni, nj, nk, mid_pos_init);
            main_value = coeff * dot(mid_diff, mid_vel) / h;
        }
        dfield_init[index] += (1.f - MIXING_AA_ALPHA) * aa_value + MIXING_AA_ALPHA * main_value;
        
    __syncthreads();
}

__global__ void cumulate_kernel(float *dfield, float *dfield_init,
                                float *x_map, float *y_map, float *z_map,
                                float h, int ni, int nj, int nk,
                                int dimx, int dimy, int dimz, bool is_point, float coeff)
{
    float3 volume[8];
    int evaluations = 8;
    volume[0] = make_float3(0.25f*h, 0.25f*h, 0.25f*h);  volume[1] = make_float3(0.25f*h, 0.25f*h, -0.25f*h);
    volume[2] = make_float3(0.25f*h, -0.25f*h, 0.25f*h); volume[3] = make_float3(0.25f*h, -0.25f*h, -0.25f*h);
    volume[4] = make_float3(-0.25f*h, 0.25f*h, 0.25f*h); volume[5] = make_float3(-0.25f*h, 0.25f*h, -0.25f*h);
    volume[6] = make_float3(-0.25f*h, -0.25f*h, 0.25f*h);volume[7] = make_float3(-0.25f*h, -0.25f*h, -0.25f*h);

    if(is_point) {
        volume[0] = make_float3(0, 0, 0);
        evaluations = 1;
    }

    int index = blockDim.x*blockIdx.x + threadIdx.x;
    float weight = 1.0/float(evaluations);

    float3 buffer_origin = make_float3(-float(dimx)*0.5f*h, -float(dimy)*0.5f*h, -float(dimz)*0.5f*h);

    int vel_buffer_i = ni + dimx;
    int vel_buffer_j = nj + dimy;
    int vel_buffer_k = nk + dimz;

    int i = index%vel_buffer_i;
    int j = (index%(vel_buffer_i*vel_buffer_j))/vel_buffer_i;
    int k = index/(vel_buffer_i*vel_buffer_j);

    float sum = 0.0;
    for (int ii = 0; ii<evaluations; ii++)
    {
        float3 point = make_float3(float(i)*h + buffer_origin.x + volume[ii].x,
                                    float(j)*h + buffer_origin.y + volume[ii].y,
                                    float(k)*h + buffer_origin.z + volume[ii].z);
        // forward mapping position
        // also used in compensation
        float3 map_pos = getPosition(x_map, y_map, z_map, h, ni, nj, nk, point);
        sum += weight * coeff * getFieldSafe(dfield, h, ni, nj, nk, dimx, dimy, dimz, map_pos);
    }
    float3 point = make_float3(float(i)*h + buffer_origin.x,
                                float(j)*h + buffer_origin.y,
                                float(k)*h + buffer_origin.z);
    // forward mapping position
    float3 map_pos = getPosition(x_map, y_map, z_map, h, ni, nj, nk, point);
    float value = coeff * getFieldSafe(dfield, h, ni, nj, nk, dimx, dimy, dimz, map_pos);
    sum = (1.f - MIXING_AA_ALPHA) * sum + MIXING_AA_ALPHA * value;
    dfield_init[index] += sum;

    __syncthreads();
}

__global__ void compensate_covector_kernel(float *src_buffer_u, float *src_buffer_v, float *src_buffer_w,
                                          float *u_adv, float *v_adv, float *w_adv,
                                          float *init_buffer, float *dest_buffer,
                                          float *x_map, float *y_map, float *z_map,
                                          float h, int ni, int nj, int nk,
                                          int dimx, int dimy, int dimz, bool is_point, float cfldt, float dt, bool do_true_mid_vel, bool do_SCPF)
{
    float3 volume[8];
    int evaluations = 8;
    volume[0] = make_float3(0.25f*h, 0.25f*h, 0.25f*h);  volume[1] = make_float3(0.25f*h, 0.25f*h, -0.25f*h);
    volume[2] = make_float3(0.25f*h, -0.25f*h, 0.25f*h); volume[3] = make_float3(0.25f*h, -0.25f*h, -0.25f*h);
    volume[4] = make_float3(-0.25f*h, 0.25f*h, 0.25f*h); volume[5] = make_float3(-0.25f*h, 0.25f*h, -0.25f*h);
    volume[6] = make_float3(-0.25f*h, -0.25f*h, 0.25f*h);volume[7] = make_float3(-0.25f*h, -0.25f*h, -0.25f*h);

    if(is_point) {
        volume[0] = make_float3(0, 0, 0);
        evaluations = 1;
    }

    int index = blockDim.x*blockIdx.x + threadIdx.x;
    float weight = 1.0/float(evaluations);

    float3 buffer_origin = make_float3(-float(dimx)*0.5f*h, -float(dimy)*0.5f*h, -float(dimz)*0.5f*h);

    int vel_buffer_i = ni + dimx;
    int vel_buffer_j = nj + dimy;
    int vel_buffer_k = nk + dimz;

    int i = index%vel_buffer_i;
    int j = (index%(vel_buffer_i*vel_buffer_j))/vel_buffer_i;
    int k = index/(vel_buffer_i*vel_buffer_j);

    float3 sum_vel = make_float3(0, 0, 0);
    for (int ii = 0; ii<evaluations; ii++)
    {
        float3 mid_pos = make_float3(float(i)*h + buffer_origin.x + volume[ii].x,
                                        float(j)*h + buffer_origin.y + volume[ii].y,
                                        float(k)*h + buffer_origin.z + volume[ii].z);
        // forward mapping position
        
        if (do_SCPF)
        {
            float3 front_pos = make_float3(mid_pos.x - buffer_origin.x, mid_pos.y - buffer_origin.y, mid_pos.z - buffer_origin.z);
            float3 back_pos = make_float3(mid_pos.x + buffer_origin.x, mid_pos.y + buffer_origin.y, mid_pos.z + buffer_origin.z);

            float3 front_pos_init = getPosition(x_map, y_map, z_map, h, ni, nj, nk, front_pos);
            float3 back_pos_init = getPosition(x_map, y_map, z_map, h, ni, nj, nk, back_pos);

            float3 front_vel = getVelocity(src_buffer_u, src_buffer_v, src_buffer_w, h, ni, nj, nk, front_pos_init);
            float3 back_vel = getVelocity(src_buffer_u, src_buffer_v, src_buffer_w, h, ni, nj, nk, back_pos_init);

            float3 avg_vel = 0.5f * (front_vel + back_vel);
            sum_vel = sum_vel + (weight * avg_vel);
        }
        else
        {
            float3 mid_pos_init;
            if (do_true_mid_vel)
                mid_pos_init = trace(u_adv, v_adv, w_adv, h, ni, nj, nk, cfldt, dt, mid_pos);
            else
                mid_pos_init = getPosition(x_map, y_map, z_map, h, ni, nj, nk, mid_pos);

            float3 mid_vel = getVelocity(src_buffer_u, src_buffer_v, src_buffer_w, h, ni, nj, nk, mid_pos_init);
            sum_vel = sum_vel + (weight * mid_vel);
        }
    }
    float3 mid_pos = make_float3(float(i)*h + buffer_origin.x, 
                                    float(j)*h + buffer_origin.y,
                                    float(k)*h + buffer_origin.z);
    // forward mapping position
    float3 front_pos = make_float3(mid_pos.x - buffer_origin.x, mid_pos.y - buffer_origin.y, mid_pos.z - buffer_origin.z);
    float3 back_pos = make_float3(mid_pos.x + buffer_origin.x, mid_pos.y + buffer_origin.y, mid_pos.z + buffer_origin.z);

    float3 front_pos_init = getPosition(x_map, y_map, z_map, h, ni, nj, nk, front_pos);
    float3 back_pos_init = getPosition(x_map, y_map, z_map, h, ni, nj, nk, back_pos);

    float3 mid_diff = front_pos_init - back_pos_init;
    if (i <= PAD_BACK_TO_SF || j <= PAD_BACK_TO_SF || k <= PAD_BACK_TO_SF ||
        i >= vel_buffer_i - 1 - PAD_BACK_TO_SF || j >= vel_buffer_j - 1 - PAD_BACK_TO_SF || k >= vel_buffer_k - 1 - PAD_BACK_TO_SF)
        mid_diff = make_float3((float)dimx * h, (float)dimy * h, (float)dimz * h);
    float aa_value = dot(mid_diff, sum_vel) / h;

    float main_value = 0.f;
    if (do_SCPF)
    {
        float3 front_vel = getVelocity(src_buffer_u, src_buffer_v, src_buffer_w, h, ni, nj, nk, front_pos_init);
        float3 back_vel = getVelocity(src_buffer_u, src_buffer_v, src_buffer_w, h, ni, nj, nk, back_pos_init);

        float3 avg_vel = 0.5f * (front_vel + back_vel);
        main_value = dot(mid_diff, avg_vel) / h;
    }
    else
    {
        float3 mid_pos_init;
        if (do_true_mid_vel)
            mid_pos_init = trace(u_adv, v_adv, w_adv, h, ni, nj, nk, cfldt, dt, mid_pos);
        else
            mid_pos_init = getPosition(x_map, y_map, z_map, h, ni, nj, nk, mid_pos);

        float3 mid_vel = getVelocity(src_buffer_u, src_buffer_v, src_buffer_w, h, ni, nj, nk, mid_pos_init);
        main_value = dot(mid_diff, mid_vel) / h;
    }

    dest_buffer[index] = ((1.f - MIXING_AA_ALPHA) * aa_value + MIXING_AA_ALPHA * main_value) - init_buffer[index];

    __syncthreads();
}

__global__ void compensate_kernel(float *src_buffer, float *init_buffer, float *dest_buffer,
                                  float *x_map, float *y_map, float *z_map,
                                  float h, int ni, int nj, int nk,
                                  int dimx, int dimy, int dimz, bool is_point)
{
    float3 volume[8];
    int evaluations = 8;
    volume[0] = make_float3(0.25f*h, 0.25f*h, 0.25f*h);  volume[1] = make_float3(0.25f*h, 0.25f*h, -0.25f*h);
    volume[2] = make_float3(0.25f*h, -0.25f*h, 0.25f*h); volume[3] = make_float3(0.25f*h, -0.25f*h, -0.25f*h);
    volume[4] = make_float3(-0.25f*h, 0.25f*h, 0.25f*h); volume[5] = make_float3(-0.25f*h, 0.25f*h, -0.25f*h);
    volume[6] = make_float3(-0.25f*h, -0.25f*h, 0.25f*h);volume[7] = make_float3(-0.25f*h, -0.25f*h, -0.25f*h);

    if(is_point) {
        volume[0] = make_float3(0, 0, 0);
        evaluations = 1;
    }

    int index = blockDim.x*blockIdx.x + threadIdx.x;
    float weight = 1.0/float(evaluations);

    float3 buffer_origin = make_float3(-float(dimx)*0.5f*h, -float(dimy)*0.5f*h, -float(dimz)*0.5f*h);

    int vel_buffer_i = ni + dimx;
    int vel_buffer_j = nj + dimy;
    int vel_buffer_k = nk + dimz;

    int i = index%vel_buffer_i;
    int j = (index%(vel_buffer_i*vel_buffer_j))/vel_buffer_i;
    int k = index/(vel_buffer_i*vel_buffer_j);

    float sum = 0.0;
    for (int ii = 0; ii<evaluations; ii++)
    {
        float3 point = make_float3(float(i)*h + buffer_origin.x + volume[ii].x,
                                    float(j)*h + buffer_origin.y + volume[ii].y,
                                    float(k)*h + buffer_origin.z + volume[ii].z);
        float3 map_pos = getPosition(x_map, y_map, z_map, h, ni, nj, nk, point);
        sum += weight * getFieldSafe(src_buffer, h, ni, nj, nk, dimx, dimy, dimz, map_pos);
    }
    float3 point = make_float3(float(i)*h + buffer_origin.x,
                                float(j)*h + buffer_origin.y,
                                float(k)*h + buffer_origin.z);
    // forward mapping position
    float3 map_pos = getPosition(x_map, y_map, z_map, h, ni, nj, nk, point);
    float value = getFieldSafe(src_buffer, h , ni, nj, nk, dimx, dimy, dimz, map_pos);
    sum = (1.f - MIXING_AA_ALPHA) * sum + MIXING_AA_ALPHA * value;
    dest_buffer[index] = sum - init_buffer[index];

    __syncthreads();
}

__global__ void estimate_kernel(float *dist_buffer, float *x_first, float *y_first, float *z_first,
                                float *x_second, float *y_second, float *z_second,
                                float h, int ni, int nj, int nk)
{
    int index = blockDim.x*blockIdx.x + threadIdx.x;
    int i = index%ni;
    int j = (index%(ni*nj))/ni;
    int k = index/(ni*nj);
    if (i>1 && i<ni-2 && j>1 && j<nj-2 && k>1 && k<nk-2)
    {
        float3 point = make_float3(h*float(i),h*float(j),h*float(k));
        // backward then forward
        float3 back_pos = getPosition(x_first, y_first, z_first, h, ni, nj, nk, point);
        float3 fwd_back_pos = getPosition(x_second, y_second, z_second, h, ni, nj, nk, back_pos);
        float dist_bf = dot(point - fwd_back_pos, point - fwd_back_pos);
        // forward then backward
        float3 fwd_pos = getPosition(x_second, y_second, z_second, h, ni, nj, nk, point);
        float3 back_fwd_pos = getPosition(x_first, y_first, z_first, h, ni, nj, nk, fwd_pos);
        float dist_fb = dot(point - back_fwd_pos, point - back_fwd_pos);
        dist_buffer[index] = max(dist_bf, dist_fb);
    }
    __syncthreads();
}

__global__ void reduce0(float *g_idata, float *g_odata, int N) {
    extern __shared__ float sdata[];
    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x *blockDim.x + threadIdx.x;

    sdata[tid] = (i<N)?g_idata[i]:0;
    __syncthreads();
    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s > 0; s >>= 1)
    {
        if (tid < s && i < N)
        {
            sdata[tid] = max(sdata[tid], sdata[tid+s]);
        }
        __syncthreads();
    }
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

__global__ void add_kernel(float *field1, float *field2, float coeff)
{
    unsigned int i = blockIdx.x *blockDim.x + threadIdx.x;
    field1[i] += coeff*field2[i];
    __syncthreads();
}

extern "C" void gpu_calc_grad(float *x_map, float *y_map, float *z_map,
                              float *xx_grad_map, float *xy_grad_map, float *xz_grad_map,
                              float *yx_grad_map, float *yy_grad_map, float *yz_grad_map,
                              float *zx_grad_map, float *zy_grad_map, float *zz_grad_map,
                              float h, int ni, int nj, int nk)
{
    int blocksize = 256;
    int numBlocks = (int)ceilf((ni * nj * nk) / blocksize);
    calc_grad_kernel<<< numBlocks, blocksize >>> (x_map,
                                                  xx_grad_map, xy_grad_map, xz_grad_map,
                                                  h, ni, nj, nk);
    
    calc_grad_kernel<<< numBlocks, blocksize >>> (y_map,
                                                  yx_grad_map, yy_grad_map, yz_grad_map,
                                                  h, ni, nj, nk);
    
    calc_grad_kernel<<< numBlocks, blocksize >>> (z_map,
                                                  zx_grad_map, zy_grad_map, zz_grad_map,
                                                  h, ni, nj, nk);
}

extern "C" void gpu_solve_forward(float *u, float *v, float *w,
                                  float *x_fwd, float *y_fwd, float *z_fwd,
                                  float h, int ni, int nj, int nk, float cfldt, float dt)
{
    int blocksize = 256;
    int numBlocks = (int)ceilf((ni * nj * nk) / blocksize);
    forward_kernel<<< numBlocks, blocksize >>> (u, v, w, x_fwd, y_fwd, z_fwd, h, ni, nj, nk, cfldt, dt);
}

extern "C" void gpu_solve_backwardSF(float *u, float *v, float *w,
                                      float *x_in, float *y_in, float *z_in,
                                      float *x_out, float *y_out, float *z_out,
                                      float h, int ni, int nj, int nk, float substep)
{
    int blocksize = 256;
    int numBlocks = (int)ceilf((ni * nj * nk) / blocksize);
    SF_backward_kernel<<< numBlocks, blocksize >>> (u, v, w, x_in, y_in, z_in, x_out, y_out, z_out, h, ni, nj, nk, substep);
}

extern "C" void gpu_solve_backwardDMC(float *u, float *v, float *w,
                                      float *x_in, float *y_in, float *z_in,
                                      float *x_out, float *y_out, float *z_out,
                                      float h, int ni, int nj, int nk, float substep)
{
    int blocksize = 256;
    int numBlocks = (int)ceilf((ni * nj * nk) / blocksize);
    DMC_backward_kernel<<< numBlocks, blocksize >>> (u, v, w, x_in, y_in, z_in, x_out, y_out, z_out, h, ni, nj, nk, substep);
}

extern "C" void gpu_advect_velocity_covector(float *u, float *v, float *w,
                                            float *u_init, float *v_init, float *w_init,
                                            float *u_adv, float *v_adv, float *w_adv,
                                            float *backward_x, float *backward_y, float *backward_z,
                                            float h, int ni, int nj, int nk, bool is_point,
                                            float cfldt, float dt, bool do_true_mid_vel, bool do_SCPF)
{
    int blocksize = 256;
    int numBlocks_u = (int)ceilf(((ni + 1) * nj * nk) / blocksize);
    int numBlocks_v = (int)ceilf((ni * (nj + 1) * nk) / blocksize);
    int numBlocks_w = (int)ceilf((ni * nj * (nk + 1)) / blocksize);
    advect_covector_kernel<<< numBlocks_u, blocksize >>>(u, u_init, v_init, w_init, u_adv, v_adv, w_adv,
                                                        backward_x, backward_y, backward_z,
                                                        h, ni, nj, nk, 1, 0, 0, is_point, cfldt, -dt, do_true_mid_vel, do_SCPF);
    advect_covector_kernel<<< numBlocks_v, blocksize >>>(v, u_init, v_init, w_init, u_adv, v_adv, w_adv,
                                                        backward_x, backward_y, backward_z,
                                                        h, ni, nj, nk, 0, 1, 0, is_point, cfldt, -dt, do_true_mid_vel, do_SCPF);
    advect_covector_kernel<<< numBlocks_w, blocksize >>>(w, u_init, v_init, w_init, u_adv, v_adv, w_adv,
                                                        backward_x, backward_y, backward_z,
                                                        h, ni, nj, nk, 0, 0, 1, is_point, cfldt, -dt, do_true_mid_vel, do_SCPF);
}

extern "C" void gpu_advect_velocity(float *u, float *v, float *w,
                                    float *u_init, float *v_init, float *w_init,
                                    float *backward_x, float *backward_y, float *backward_z,
                                    float h, int ni, int nj, int nk, bool is_point)
{
    int blocksize = 256;
    int numBlocks_u = (int)ceilf(((ni + 1) * nj * nk) / blocksize);
    int numBlocks_v = (int)ceilf((ni * (nj + 1) * nk) / blocksize);
    int numBlocks_w = (int)ceilf((ni * nj * (nk + 1)) / blocksize);
    advect_kernel<<< numBlocks_u, blocksize >>>(u, u_init, backward_x, backward_y, backward_z, h, ni, nj, nk, 1, 0, 0, is_point);
    advect_kernel<<< numBlocks_v, blocksize >>>(v, v_init, backward_x, backward_y, backward_z, h, ni, nj, nk, 0, 1, 0, is_point);
    advect_kernel<<< numBlocks_w, blocksize >>>(w, w_init, backward_x, backward_y, backward_z, h, ni, nj, nk, 0, 0, 1, is_point);
}

extern "C" void gpu_advect_vel_double(float *u, float *v, float *w,
                                      float *utemp, float *vtemp, float *wtemp,
                                      float *backward_x, float *backward_y, float *backward_z,
                                      float *backward_xprev,  float *backward_yprev,  float *backward_zprev,
                                      float h, int ni, int nj, int nk, bool is_point, float blend_coeff)
{
    int blocksize = 256;
    int numBlocks_u = (int)ceilf(((ni + 1) * nj * nk) / blocksize);
    int numBlocks_v = (int)ceilf((ni * (nj + 1) * nk) / blocksize);
    int numBlocks_w = (int)ceilf((ni * nj * (nk + 1)) / blocksize);
    doubleAdvect_kernel<<< numBlocks_u, blocksize >>> (u, utemp, backward_x,backward_y,backward_z,
            backward_xprev, backward_yprev, backward_zprev,h,ni,nj,nk, 1, 0, 0, is_point, blend_coeff);

    doubleAdvect_kernel<<< numBlocks_v, blocksize >>> (v, vtemp, backward_x,backward_y,backward_z,
            backward_xprev, backward_yprev, backward_zprev,h,ni,nj,nk, 0, 1, 0, is_point, blend_coeff);

    doubleAdvect_kernel<<< numBlocks_w, blocksize >>> (w, wtemp, backward_x,backward_y,backward_z,
            backward_xprev, backward_yprev, backward_zprev,h,ni,nj,nk, 0, 0, 1, is_point, blend_coeff);
}

extern "C" void gpu_advect_field_covector(float *field, float *field_init,
                                         float *backward_x, float *backward_y, float *backward_z,
                                         float h, int ni, int nj, int nk, bool is_point)
{
    int blocksize = 256;
    int numBlocks = (int)ceilf((ni * nj * nk) / blocksize);
    advect_kernel<<< numBlocks, blocksize >>>(field, field_init, backward_x, backward_y, backward_z, h, ni, nj, nk, 0, 0, 0, is_point);
}

extern "C" void gpu_advect_field(float *field, float *field_init,
                                 float *backward_x, float *backward_y, float *backward_z,
                                 float h, int ni, int nj, int nk, bool is_point)
{
    int blocksize = 256;
    int numBlocks = (int)ceilf((ni * nj * nk) / blocksize);
    advect_kernel<<< numBlocks, blocksize >>>(field, field_init, backward_x, backward_y, backward_z, h, ni, nj, nk, 0, 0, 0, is_point);
}

extern "C" void gpu_advect_field_double(float *field, float *field_prev,
                                        float *backward_x, float *backward_y, float *backward_z,
                                        float *backward_xprev, float *backward_yprev,   float *backward_zprev,
                                        float h, int ni, int nj, int nk, bool is_point, float blend_coeff)
{
    int blocksize = 256;
    int numBlocks = (int)ceilf((ni * nj * nk) / blocksize);
    doubleAdvect_kernel<<< numBlocks, blocksize >>> (field, field_prev, backward_x, backward_y, backward_z,
            backward_xprev, backward_yprev, backward_zprev,h,ni,nj,nk, 0, 0, 0, is_point, blend_coeff);
}

extern "C" void gpu_compensate_velocity_covector(float *u, float *v, float *w,
                                                float *du, float *dv, float *dw,
                                                float *u_src, float *v_src, float *w_src,
                                                float *u_tmp, float *v_tmp, float *w_tmp,
                                                float *forward_x, float *forward_y, float *forward_z,
                                                float *backward_x, float *backward_y, float *backward_z,
                                                float h, int ni, int nj, int nk, bool is_point, bool do_EC_with_clamp,
                                                float cfldt, float dt, bool do_true_mid_vel, bool do_SCPF)
{
    int blocksize = 256;
    int numBlocks_u = (int)ceilf(((ni + 1) * nj * nk) / blocksize);
    int numBlocks_v = (int)ceilf((ni * (nj + 1) * nk) / blocksize);
    int numBlocks_w = (int)ceilf((ni * nj * (nk + 1)) / blocksize);
    // error at time 0 will be in u_src, v_src, w_src
    compensate_covector_kernel<<< numBlocks_u, blocksize>>>(u, v, w,
                                                           u_tmp, v_tmp, w_tmp,
                                                           du, u_src,
                                                           forward_x, forward_y, forward_z,
                                                           h, ni, nj, nk, 1, 0, 0, is_point, cfldt, dt, do_true_mid_vel, do_SCPF);
    compensate_covector_kernel<<< numBlocks_v, blocksize>>>(u, v, w,
                                                           u_tmp, v_tmp, w_tmp,
                                                           dv, v_src,
                                                           forward_x, forward_y, forward_z,
                                                           h, ni, nj, nk, 0, 1, 0, is_point, cfldt, dt, do_true_mid_vel, do_SCPF);
    compensate_covector_kernel<<< numBlocks_w, blocksize>>>(u, v, w,
                                                           u_tmp, v_tmp, w_tmp,
                                                           dw, w_src,
                                                           forward_x, forward_y, forward_z,
                                                           h, ni, nj, nk, 0, 0, 1, is_point, cfldt, dt, do_true_mid_vel, do_SCPF);
    if (do_EC_with_clamp)
    {
        // now subtract error at time t, compensated velocity will be stored in gpu.u, gpu.v, gpu.w
        cudaMemcpy(du, u, sizeof(float)*(ni+1)*nj*nk, cudaMemcpyDeviceToDevice);
        cudaMemcpy(dv, v, sizeof(float)*ni*(nj+1)*nk, cudaMemcpyDeviceToDevice);
        cudaMemcpy(dw, w, sizeof(float)*ni*nj*(nk+1), cudaMemcpyDeviceToDevice);
    }
    cumulate_covector_kernel<<< numBlocks_u, blocksize >>>(u_src, v_src, w_src,
                                                          u_tmp, v_tmp, w_tmp,
                                                          u,
                                                          backward_x, backward_y, backward_z,
                                                          h, ni, nj, nk, 1, 0, 0, is_point, -0.5f, cfldt, -dt, do_true_mid_vel, do_SCPF);
    cumulate_covector_kernel<<< numBlocks_v, blocksize >>>(u_src, v_src, w_src,
                                                          u_tmp, v_tmp, w_tmp,
                                                          v,
                                                          backward_x, backward_y, backward_z,
                                                          h, ni, nj, nk, 0, 1, 0, is_point, -0.5f, cfldt, -dt, do_true_mid_vel, do_SCPF);
    cumulate_covector_kernel<<< numBlocks_w, blocksize >>>(u_src, v_src, w_src,
                                                          u_tmp, v_tmp, w_tmp,
                                                          w,
                                                          backward_x, backward_y, backward_z,
                                                          h, ni, nj, nk, 0, 0, 1, is_point, -0.5f, cfldt, -dt, do_true_mid_vel, do_SCPF);
    if (do_EC_with_clamp)
    {
        // clamp extrema, clamped result will be in gpu.u, gpu.v, gpu.w
        clampExtrema_covector_kernel<<< numBlocks_u, blocksize >>>(du, u, ni+1, nj, nk);
        clampExtrema_covector_kernel<<< numBlocks_v, blocksize >>>(dv, v, ni, nj+1, nk);
        clampExtrema_covector_kernel<<< numBlocks_w, blocksize >>>(dw, w, ni, nj, nk+1);
    }
}

extern "C" void gpu_compensate_velocity(float *u, float *v, float *w,
                                        float *du, float *dv, float *dw,
                                        float *u_src, float *v_src, float *w_src,
                                        float *forward_x, float *forward_y, float *forward_z,
                                        float *backward_x, float *backward_y, float *backward_z,
                                        float h, int ni, int nj, int nk, bool is_point, bool do_EC_with_clamp)
{
    int blocksize = 256;
    int numBlocks_u = (int)ceilf(((ni + 1) * nj * nk) / blocksize);
    int numBlocks_v = (int)ceilf((ni * (nj + 1) * nk) / blocksize);
    int numBlocks_w = (int)ceilf((ni * nj * (nk + 1)) / blocksize);
    // error at time 0 will be in u_src, v_src, w_src
    compensate_kernel<<< numBlocks_u, blocksize>>>(u, du, u_src, forward_x, forward_y, forward_z, h, ni, nj, nk, 1, 0, 0, is_point);
    compensate_kernel<<< numBlocks_v, blocksize>>>(v, dv, v_src, forward_x, forward_y, forward_z, h, ni, nj, nk, 0, 1, 0, is_point);
    compensate_kernel<<< numBlocks_w, blocksize>>>(w, dw, w_src, forward_x, forward_y, forward_z, h, ni, nj, nk, 0, 0, 1, is_point);
    if (do_EC_with_clamp)
    {
        // now subtract error at time t, compensated velocity will be stored in gpu.u, gpu.v, gpu.w
        cudaMemcpy(du, u, sizeof(float)*(ni+1)*nj*nk, cudaMemcpyDeviceToDevice);
        cudaMemcpy(dv, v, sizeof(float)*ni*(nj+1)*nk, cudaMemcpyDeviceToDevice);
        cudaMemcpy(dw, w, sizeof(float)*ni*nj*(nk+1), cudaMemcpyDeviceToDevice);
    }
    cumulate_kernel<<< numBlocks_u, blocksize >>>(u_src, u, backward_x, backward_y, backward_z, h, ni, nj, nk, 1, 0, 0, is_point, -0.5f);
    cumulate_kernel<<< numBlocks_v, blocksize >>>(v_src, v, backward_x, backward_y, backward_z, h, ni, nj, nk, 0, 1, 0, is_point, -0.5f);
    cumulate_kernel<<< numBlocks_w, blocksize >>>(w_src, w, backward_x, backward_y, backward_z, h, ni, nj, nk, 0, 0, 1, is_point, -0.5f);
    if (do_EC_with_clamp)
    {
        // clamp extrema, clamped result will be in gpu.u, gpu.v, gpu.w
        clampExtrema_kernel<<< numBlocks_u, blocksize >>>(du, u, ni+1, nj, nk);
        clampExtrema_kernel<<< numBlocks_v, blocksize >>>(dv, v, ni, nj+1, nk);
        clampExtrema_kernel<<< numBlocks_w, blocksize >>>(dw, w, ni, nj, nk+1);
    }
}

extern "C" void gpu_compensate_field_covector(float *field, float *dfield, float *field_src,
                                             float *forward_x, float *forward_y, float *forward_z,
                                             float *backward_x, float *backward_y, float *backward_z,
                                             float h, int ni, int nj, int nk, bool is_point)
{
    int blocksize = 256;
    int numBlocks = (int)ceilf((ni * nj * nk) / blocksize);
    // error at time 0 will be in dfield
    compensate_kernel<<< numBlocks, blocksize>>>(field, dfield, field_src, forward_x, forward_y, forward_z, h, ni, nj, nk, 0, 0, 0, is_point);
    // now subtract error at time t, compensated velocity will be stored in gpu.field
    cudaMemcpy(dfield, field, sizeof(float)*ni*nj*nk, cudaMemcpyDeviceToDevice);
    cumulate_kernel<<< numBlocks, blocksize >>>(field_src, field, backward_x, backward_y, backward_z, h, ni, nj, nk, 0, 0, 0, is_point, -0.5f);
    // clamp extrema, clamped result will be in gpu.field
    clampExtrema_kernel<<< numBlocks, blocksize >>>(dfield, field, ni, nj, nk);
}

extern "C" void gpu_compensate_field(float *field, float *dfield, float *field_src,
                                     float *forward_x, float *forward_y, float *forward_z,
                                     float *backward_x, float *backward_y, float *backward_z,
                                     float h, int ni, int nj, int nk, bool is_point)
{
    int blocksize = 256;
    int numBlocks = (int)ceilf((ni * nj * nk) / blocksize);
    // error at time 0 will be in dfield
    compensate_kernel<<< numBlocks, blocksize>>>(field, dfield, field_src, forward_x, forward_y, forward_z, h, ni, nj, nk, 0, 0, 0, is_point);
    // now subtract error at time t, compensated velocity will be stored in gpu.field
    cudaMemcpy(dfield, field, sizeof(float)*ni*nj*nk, cudaMemcpyDeviceToDevice);
    cumulate_kernel<<< numBlocks, blocksize >>>(field_src, field, backward_x, backward_y, backward_z, h, ni, nj, nk, 0, 0, 0, is_point, -0.5f);
    // clamp extrema, clamped result will be in gpu.field
    clampExtrema_kernel<<< numBlocks, blocksize >>>(dfield, field, ni, nj, nk);
}

extern "C" void gpu_accumulate_velocity_covector(float *u_change, float *v_change, float *w_change,
                                                float *u_adv, float *v_adv, float *w_adv,
                                                float *du_init, float *dv_init, float *dw_init,
                                                float *forward_x, float *forward_y, float *forward_z,
                                                float h, int ni, int nj, int nk, bool is_point, float coeff,
                                                float cfldt, float dt, bool do_true_mid_vel, bool do_SCPF)
{
    int blocksize = 256;
    int numBlocks_u = (int)ceilf(((ni + 1) * nj * nk) / blocksize);
    int numBlocks_v = (int)ceilf((ni * (nj + 1) * nk) / blocksize);
    int numBlocks_w = (int)ceilf((ni * nj * (nk + 1)) / blocksize);
    cumulate_covector_kernel <<< numBlocks_u, blocksize >>> (u_change, v_change, w_change,
                                                            u_adv, v_adv, w_adv,
                                                            du_init,
                                                            forward_x, forward_y, forward_z,
                                                            h, ni, nj, nk, 1, 0, 0, is_point, coeff, cfldt, dt, do_true_mid_vel, do_SCPF);
    cumulate_covector_kernel <<< numBlocks_v, blocksize >>> (u_change, v_change, w_change,
                                                            u_adv, v_adv, w_adv,
                                                            dv_init,
                                                            forward_x, forward_y, forward_z,
                                                            h, ni, nj, nk, 0, 1, 0, is_point, coeff, cfldt, dt, do_true_mid_vel, do_SCPF);
    cumulate_covector_kernel <<< numBlocks_w, blocksize >>> (u_change, v_change, w_change,
                                                            u_adv, v_adv, w_adv,
                                                            dw_init,
                                                            forward_x, forward_y, forward_z,
                                                            h, ni, nj, nk, 0, 0, 1, is_point, coeff, cfldt, dt, do_true_mid_vel, do_SCPF);
}

extern "C" void gpu_accumulate_velocity(float *u_change, float *v_change, float *w_change,
                                        float *du_init, float *dv_init, float *dw_init,
                                        float *forward_x, float *forward_y, float *forward_z,
                                        float h, int ni, int nj, int nk, bool is_point, float coeff)
{
    int blocksize = 256;
    int numBlocks_u = (int)ceilf(((ni + 1) * nj * nk) / blocksize);
    int numBlocks_v = (int)ceilf((ni * (nj + 1) * nk) / blocksize);
    int numBlocks_w = (int)ceilf((ni * nj * (nk + 1)) / blocksize);
    cumulate_kernel<<< numBlocks_u, blocksize >>> (u_change, du_init, forward_x, forward_y, forward_z, h, ni, nj, nk, 1, 0, 0, is_point, coeff);
    cumulate_kernel<<< numBlocks_v, blocksize >>> (v_change, dv_init, forward_x, forward_y, forward_z, h, ni, nj, nk, 0, 1, 0, is_point, coeff);
    cumulate_kernel<<< numBlocks_w, blocksize >>> (w_change, dw_init, forward_x, forward_y, forward_z, h, ni, nj, nk, 0, 0, 1, is_point, coeff);
}

extern "C" void gpu_accumulate_field(float *field_change, float *dfield_init,
                                     float *forward_x, float *forward_y, float *forward_z,
                                     float h, int ni, int nj, int nk, bool is_point, float coeff)
{
    int blocksize = 256;
    int numBlocks = (int)ceilf((ni * nj * nk) / blocksize);
    cumulate_kernel<<< numBlocks, blocksize >>> (field_change, dfield_init, forward_x, forward_y, forward_z, h, ni, nj, nk, 0, 0, 0, is_point, coeff);
}

extern "C" void gpu_estimate_distortion(float *du,
                                        float *x_back, float *y_back, float *z_back,
                                        float *x_fwd, float *y_fwd, float *z_fwd,
                                        float h, int ni, int nj, int nk)
{
    int blocksize = 256;
    int est_numBlocks = (int)ceilf((ni * nj * nk) / blocksize);
    // distortion will be stored in gpu.du
    estimate_kernel<<< est_numBlocks, blocksize>>> (du, x_back, y_back, z_back, x_fwd, y_fwd, z_fwd, h, ni, nj, nk);
}

extern "C" void gpu_semilag(float *field, float *field_src,
                            float *u, float *v, float *w,
                            int dimx, int dimy, int dimz,
                            float h, int ni, int nj, int nk, bool is_point, float cfldt, float dt)
{
    int blocksize = 256;
    int total_num = (ni+dimx)*(nj+dimy)*(nk+dimz);
    int numBlocks = (int)ceilf(total_num / blocksize);
    semilag_kernel<<<numBlocks, blocksize>>>(field, field_src, u, v, w, dimx, dimy, dimz, h, ni, nj, nk, is_point, cfldt, dt);
}

extern "C" void gpu_clamp_extrema(float *before, float *after,
                                  int ni, int nj, int nk)
{
    int blocksize = 256;
    int numBlocks = (int)ceilf((ni * nj * nk) / blocksize);
    // clamp extrema, clamped result will be in gpu.field
    clampExtrema_kernel<<< numBlocks, blocksize >>>(before, after, ni, nj, nk);
}

extern "C" void gpu_add(float *field1, float *field2, float coeff, int number)
{
    int blocksize = 256;
    int numBlocks = (int)ceilf(number / blocksize);
    add_kernel<<<numBlocks, blocksize>>>(field1, field2, coeff);
}