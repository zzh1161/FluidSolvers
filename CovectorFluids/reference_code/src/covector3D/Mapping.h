#ifndef COVECTOR_MAPPING_H
#define COVECTOR_MAPPING_H

#include <iostream>
#include <cstdint>
#include "tbb/tbb.h"
#include "../utils/color_macro.h"
#include "../include/fluid_buffer3D.h"
#include "GPU_Advection.h"

// two level BIMOCQ advector
class MapperBase
{
public:
    MapperBase() = default;
    virtual ~MapperBase() = default;

    virtual void init(uint ni, uint nj, uint nk, float h, float coeff, gpuMapper *mymapper);
    virtual void updateForward(float cfldt, float dt);
    virtual void updateBackward(float cfldt, float dt, int do_SF_DMC);
    virtual void updateMapping(const buffer3Df &un, const buffer3Df &vn, const buffer3Df &wn, float cfldt, float dt, int do_SF_DMC=1);
    virtual void moveToTemp(int tempToOrig_origToTemp);
    //virtual void updateMappingGrad();
    //virtual void updateForwardGrad();
    //virtual void updateBackwardGrad();

    virtual void accumulateVelocity(buffer3Df &u_init, buffer3Df &v_init, buffer3Df &w_init,
                                    const buffer3Df &u_change, const buffer3Df &v_change, const buffer3Df &w_change,
                                    float coeff, bool do_anitaliasing);
    virtual void accumulateVelocityCovector(buffer3Df &u_init, buffer3Df &v_init, buffer3Df &w_init,
                                           const buffer3Df &u_change, const buffer3Df &v_change, const buffer3Df &w_change,
                                           const buffer3Df& un, const buffer3Df& vn, const buffer3Df& wn,
                                           float coeff, bool do_anitaliasing, float cfldt, float dt, bool do_true_mid_vel, bool do_SCPF=false);
    virtual void accumulateField(buffer3Df &field_init, const buffer3Df &field_change, bool do_anitaliasing);
    virtual float estimateDistortion(const buffer3Dc &boundary);

    virtual void reinitializeMapping();
    virtual void advectVelocityCovector(buffer3Df &un, buffer3Df &vn, buffer3Df &wn,
                                       const buffer3Df &u_init, const buffer3Df &v_init, const buffer3Df &w_init,
                                       const buffer3Df &u_prev, const buffer3Df &v_prev, const buffer3Df &w_prev,
                                       bool do_EC, bool do_EC_with_clamp, bool do_anitaliasing, float cfldt, float dt, bool do_true_mid_vel, bool do_SCPF=false);
    virtual void advectVelocity(buffer3Df &un, buffer3Df &vn, buffer3Df &wn,
                                const buffer3Df &u_init, const buffer3Df &v_init, const buffer3Df &w_init,
                                const buffer3Df &u_prev, const buffer3Df &v_prev, const buffer3Df &w_prev,
                                bool do_EC, bool do_EC_with_clamp, bool do_anitaliasing);
    virtual void advectFieldCovector(buffer3Df &field, const buffer3Df &field_init, const buffer3Df &field_prev, bool do_EC, bool do_anitaliasing);
    virtual void advectField(buffer3Df& field, const buffer3Df& field_init, const buffer3Df& field_prev, bool do_EC, bool do_anitaliasing);

    float _h;
    // phi_t = blend_coeff * phi_curr + (1 - blend_coeff) * phi_prev
    float blend_coeff;
    uint total_reinit_count;
    uint _ni, _nj, _nk;
    buffer3Df forward_x, forward_y, forward_z;
    buffer3Df forward_temp_x, forward_temp_y, forward_temp_z;
    //buffer3Df forward_grad_xx, forward_grad_xy, forward_grad_xz;
    //buffer3Df forward_grad_yx, forward_grad_yy, forward_grad_yz;
    //buffer3Df forward_grad_zx, forward_grad_zy, forward_grad_zz;
    buffer3Df backward_x, backward_y, backward_z;
    buffer3Df backward_temp_x, backward_temp_y, backward_temp_z;
    //buffer3Df backward_grad_xx, backward_grad_xy, backward_grad_xz;
    //buffer3Df backward_grad_yx, backward_grad_yy, backward_grad_yz;
    //buffer3Df backward_grad_zx, backward_grad_zy, backward_grad_zz;
    buffer3Df backward_xprev, backward_yprev, backward_zprev;
    /// gpu solver
    gpuMapper *gpuSolver;
};

#endif //COVECTOR_MAPPING_H
