"""Pore geometry and component-aware extension of the production MINRES solve.

Ownership is exact nonperiodic 6-neighbour graph distance with lower-site-id ties.
Only physical interface fluxes may be periodic in x, matching the manuscript scope.
"""
from __future__ import annotations
from collections import deque
import heapq
import time
from types import SimpleNamespace
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, minres, splu


def graph(mask):
    pore = np.flatnonzero(mask.ravel())
    ids = np.full(mask.size, -1, np.int64)
    ids[pore] = np.arange(len(pore))
    ids = ids.reshape(mask.shape)
    neighbors = np.full((len(pore),6),-1,np.int64)
    for axis in range(3):
        lo, hi = [slice(None)]*3, [slice(None)]*3
        lo[axis],hi[axis]=slice(None,-1),slice(1,None)
        a,b=ids[tuple(lo)],ids[tuple(hi)]
        ok=(a>=0)&(b>=0)
        neighbors[a[ok],2*axis+1]=b[ok]
        neighbors[b[ok],2*axis]=a[ok]
    return pore, ids, neighbors


def exact_owners(mask, seeds, h=1.):
    pore,ids,neighbors=graph(mask)
    seeds=np.unique(np.asarray(seeds,np.int64))
    if np.any(~mask.ravel()[seeds]):
        raise ValueError("Seeds must already be in pore voxels")
    distance=np.full(len(pore),np.iinfo(np.int32).max,np.int64)
    owner=np.full(len(pore),-1,np.int32)
    heap=[]
    for label,node in enumerate(ids.ravel()[seeds]):
        owner[node],distance[node]=label,0
        heapq.heappush(heap,(0,label,int(node)))
    while heap:
        d,c,i=heapq.heappop(heap)
        if d!=distance[i] or c!=owner[i]:
            continue
        for j in neighbors[i]:
            if j<0:
                continue
            if d+1<distance[j] or (d+1==distance[j] and c<owner[j]):
                distance[j],owner[j]=d+1,c
                heapq.heappush(heap,(d+1,c,int(j)))
    if np.any(owner<0):
        raise ValueError("At least one nonperiodic pore component has no site")
    # Every non-site voxel must have a same-owner distance-decreasing parent.
    same_parent=np.zeros(len(pore),bool)
    for column in range(6):
        nb=neighbors[:,column]
        ok=nb>=0
        same_parent[ok] |= (owner[nb[ok]]==owner[ok]) & (distance[nb[ok]]==distance[ok]-1)
    assert np.all(same_parent | (distance==0))
    labels=np.full(mask.shape,-1,np.int32)
    dist=np.zeros(mask.shape,float)
    labels.ravel()[pore]=owner
    dist.ravel()[pore]=h*distance
    return labels,dist


def fps_seeds(mask, count, candidates=None):
    """Incremental graph-FPS, including component coverage and deterministic ties."""
    pore,ids,neighbors=graph(mask)
    candidate_ids=np.arange(len(pore)) if candidates is None else ids.ravel()[np.unique(candidates)]
    if np.any(candidate_ids<0):
        raise ValueError("Candidate is outside pore space")
    if count>len(candidate_ids):
        raise ValueError("Not enough candidate sites")
    coords=np.column_stack(np.unravel_index(pore,mask.shape))
    center=coords.mean(axis=0)
    first=candidate_ids[np.argmin(((coords[candidate_ids]-center)**2).sum(axis=1))]
    distance=np.full(len(pore),mask.size+1,np.int64)
    chosen=[]
    i=int(first)
    for _ in range(count):
        chosen.append(i)
        distance[i]=0
        queue=deque([i])
        while queue:
            j=queue.popleft()
            d=distance[j]+1
            for k in neighbors[j]:
                if k>=0 and d<distance[k]:
                    distance[k]=d
                    queue.append(int(k))
        i=int(candidate_ids[np.argmax(distance[candidate_ids])])
    if np.any(distance>mask.size):
        # A component with no measured location needs a geometric support site;
        # report these explicitly rather than silently labelling it particle-derived.
        while np.any(distance>mask.size):
            i=int(np.flatnonzero(distance>mask.size)[0])
            chosen.append(i)
            distance[i]=0
            queue=deque([i])
            while queue:
                j=queue.popleft()
                for k in neighbors[j]:
                    if k>=0 and distance[j]+1<distance[k]:
                        distance[k]=distance[j]+1
                        queue.append(int(k))
    return pore[np.asarray(chosen)],len(chosen)-count


def geometry(mask, seeds, h=1., periodic_x=True):
    labels,dist=exact_owners(mask,seeds,h)
    nc=int(labels.max())+1
    pore=np.flatnonzero(mask.ravel())
    coords=np.column_stack(np.unravel_index(pore,mask.shape))[:,::-1]
    xyz=(coords+.5)*h
    own=labels.ravel()[pore]
    counts=np.bincount(own,minlength=nc)
    centroid=np.column_stack([np.bincount(own,weights=xyz[:,a],minlength=nc)/counts for a in range(3)])
    keys=[]
    for axis in range(3):
        lo,hi=[slice(None)]*3,[slice(None)]*3
        lo[axis],hi[axis]=slice(None,-1),slice(1,None)
        a,b=labels[tuple(lo)],labels[tuple(hi)]
        ok=(a>=0)&(b>=0)&(a!=b)
        keys.append(np.minimum(a[ok],b[ok])*nc+np.maximum(a[ok],b[ok]))
    if periodic_x:
        a,b=labels[:,:,-1],labels[:,:,0]
        ok=(a>=0)&(b>=0)&(a!=b)
        keys.append(np.minimum(a[ok],b[ok])*nc+np.maximum(a[ok],b[ok]))
    edge=np.unique(np.concatenate(keys))
    g=SimpleNamespace(mask=mask,labels=labels,dist=dist,n_cells=nc,volume=counts*h**3,
                      centroid=centroid,owner=edge//nc,neigh=edge%nc)
    return g,pore,xyz


def pressure_components(trace):
    nc=trace.n_cells
    a,b=trace.face_owner,trace.face_neigh
    adjacency=sparse.coo_matrix((np.ones(2*len(a)),(np.r_[a,b],np.r_[b,a])),shape=(nc,nc)).tocsr()
    ncomp,component=connected_components(adjacency,directed=False)
    gauge=np.array([np.flatnonzero(component==i)[-1] for i in range(ncomp)])
    keep=np.ones(nc,bool)
    keep[gauge]=False
    return ncomp,component,keep


def solve(system, b=None, A=None, method="minres", rtol=1e-12, maxiter=30000):
    """Production block-Jacobi MINRES with one pressure gauge per component.

    True full momentum and continuity residuals drive iterative refinement.
    Refinement is decided on these residuals, not on a single global saddle-point residual.
    """
    started=time.perf_counter()
    A=system.velocity_matrix if A is None else A
    b=system.rhs_trace if b is None else b
    D=system.divergence_matrix
    nc,nz=D.shape
    ncomp,component,keep=pressure_components(system.trace_geometry)
    Dr=D[keep].tocsr()
    K=sparse.bmat([[A,Dr.T],[Dr,None]],format="csr")
    rhs=np.r_[b,np.zeros(keep.sum())]
    d=A.diagonal()
    if np.any(d<=100*np.finfo(float).eps*max(np.max(np.abs(d)),1.)):
        raise RuntimeError("Non-positive velocity diagonal")
    s=np.asarray(Dr.multiply(Dr)@(1/d)).ravel()
    if np.any(s<=0):
        raise RuntimeError("Non-positive retained pressure Schur diagonal")
    inverse=np.r_[1/d,1/s]
    M=LinearOperator(K.shape,matvec=lambda v:inverse*v,dtype=float)
    setup_s=time.perf_counter()-started
    iterations=0
    def callback(_):
        nonlocal iterations
        iterations+=1
    t=time.perf_counter()
    if method=="direct":
        lu=splu(K.tocsc(),permc_spec="MMD_AT_PLUS_A")
        x=lu.solve(rhs)
        info=0
    else:
        x,info=minres(K,rhs,M=M,rtol=rtol,maxiter=maxiter,callback=callback)
    history=[]
    for refinement in range(5):
        z=x[:nz]
        p=np.zeros(nc)
        p[keep]=x[nz:]
        momentum=A@z+D.T@p-b
        continuity=D@z
        relative=float(np.linalg.norm(momentum,np.inf)/max(np.linalg.norm(b,np.inf),1e-30))
        mass_abs=float(np.max(np.abs(continuity),initial=0))
        mass_rel=mass_abs/max(float(np.max(np.abs(D)@np.abs(z),initial=0)),1e-30)
        history.append({"momentum_relative_inf":relative,"mass_absolute_inf":mass_abs,
                        "mass_relative_inf":mass_rel,"linear_info":int(info)})
        if relative<1e-9 and mass_abs<1e-9 and mass_rel<1e-9 and np.all(np.isfinite(x)):
            break
        if method=="direct":
            x+=lu.solve(rhs-K@x)
        else:
            correction,info=minres(K,rhs-K@x,M=M,rtol=rtol,maxiter=maxiter,callback=callback)
            x+=correction
    else:
        raise RuntimeError(f"Component-aware solver failed: {history}")
    for i in range(ncomp):
        sel=component==i
        p[sel]-=np.mean(p[sel])
    return z,p,{"solver":method,"setup_s":setup_s,"solve_s":time.perf_counter()-t,
                "total_s":time.perf_counter()-started,"iterations":iterations,
                "pressure_components":int(ncomp),"system_dimension":K.shape[0],"kkt_nnz":K.nnz,
                "refinement_history":history,**history[-1]}


def pressure_fit(system,z,face_metric=True):
    """Gradient fit to original momentum residual in the physical face dual norm."""
    trace=system.trace_geometry
    ncomp,component,keep=pressure_components(trace)
    D=system.divergence_matrix
    mass=np.zeros(trace.n_trace_modes)
    for col in range(3):
        ids=trace.face_mode_ids[:,col]
        ok=ids>=0
        np.add.at(mass,ids[ok],trace.voxel_size**2*trace.face_mode_values[ok,col]**2)
    w=1/np.repeat(mass,3) if face_metric else np.ones(D.shape[1])
    Dr=D[keep]
    L=(Dr.multiply(w[None,:])@Dr.T).tocsc()
    rhs=Dr@(w*(system.rhs_trace-system.velocity_matrix@z))
    p=np.zeros(trace.n_cells)
    if L.shape[0]:
        p[keep]=splu(L).solve(rhs)
    for i in range(ncomp):
        sel=component==i
        p[sel]-=np.average(p[sel],weights=trace.cell_volume[sel])
    return -p
