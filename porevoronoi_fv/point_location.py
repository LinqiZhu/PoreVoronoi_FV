"""locate(): physical positions to voxels (solids rejected, no wall snapping).
assess(): auxiliary full-field and cut-plane readouts; not a readout of the paper (those are in readouts.py) and not
called by the drivers of this folder.
"""
from __future__ import annotations

import numpy as np

import velocity_recovery as rp


def locate(case, xyz):
    """Map physical positions to containing voxels. Reject solids, never wall-snap."""
    xyz=np.asarray(xyz,float).copy()
    xyz[...,0]%=case.mask.shape[2]*case.h
    ijk=np.floor(xyz[...,::-1]/case.h).astype(np.int64)
    inside=np.all((ijk>=0)&(ijk<np.array(case.mask.shape)),axis=-1)
    flat=np.full(inside.shape,-1,np.int64)
    safe=ijk[inside]
    flat[inside]=np.ravel_multi_index(safe.T,case.mask.shape)
    good=inside.copy()
    good[inside]=case.mask.ravel()[flat[inside]]
    flat[~good]=-1
    return xyz,flat,good


def assess(case,part,z,p,unvisited):
    trace,system=part["trace"],part["system"]
    u=(part["Hfield"]@z).reshape(-1,3)
    flux=rp.face_flux(z,trace)
    net=np.zeros(trace.n_cells)
    np.add.at(net,trace.face_owner,flux)
    np.add.at(net,trace.face_neigh,-flux)
    q=float(np.sum(flux[part["cut"]]*trace.face_normal[part["cut"],0]))
    k=case.nu*q/(case.mask.shape[0]*case.mask.shape[1]*case.h**2*case.force) if part["wrap_same"]==0 else None
    return dict(velocity_relative_l2_volume=rp.normerr(u,case.u_ref),
        velocity_relative_l2_unvisited=rp.normerr(u[unvisited],case.u_ref[unvisited]),
        K_shared_cut=k,K_reference=case.k_ref,K_relative_signed=None if k is None else k/case.k_ref-1,
        K_volume_readout=float(case.nu*trace.cell_volume@(part["R"]@z).reshape(-1,3)[:,0]/(case.mask.size*case.h**3*case.force)),
        direct_face_balance_inf=float(np.max(np.abs(net))),
        face_vs_matrix_balance_inf=float(np.max(np.abs(net-system.divergence_matrix@z))),
        original_momentum_relative_l2=float(np.linalg.norm(system.velocity_matrix@z+system.divergence_matrix.T@p-system.rhs_trace)/np.linalg.norm(system.rhs_trace)))

