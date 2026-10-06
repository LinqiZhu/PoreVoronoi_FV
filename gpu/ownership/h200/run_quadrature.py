import hashlib, json, socket, time
import numpy as np
from common import ROOT, save, check_input_manifest, strict_owner
from manufactured_fields_export import Manufactured, voxel_integrals, face_integrals, lattice_sites

ORDERS=(3,4,5,6,7)
REFERENCE_ERROR_FRACTION=1e-3
FORCE_RELATIVE_TOLERANCE=1e-10

def metrics(value,reference,weights=None):
    value=np.asarray(value);reference=np.asarray(reference);delta=value-reference
    if weights is None:weights=np.ones(len(value))
    weights=np.asarray(weights).reshape((-1,)+(1,)*(value.ndim-1))
    norm=lambda x:float(np.sqrt(np.sum(weights*x*x)))
    return {'relative_l2':norm(delta)/max(norm(reference),1e-300),
            'max_abs':float(np.max(np.abs(delta))),
            'relative_linf':float(np.max(np.abs(delta)))/max(float(np.max(np.abs(reference))),1e-300),
            'reference_l2':norm(reference),'entries':int(value.size)}

def geometry(n,m,h):
    mask=np.ones((n,n,n),dtype=bool);sites=lattice_sites(n,m)
    p,s,owner,d,nb=strict_owner(mask,sites)
    z,y,x=np.unravel_index(p,mask.shape)
    centres=np.column_stack([x+.5,y+.5,z+.5])*h
    nc=len(s);volume=np.bincount(owner,minlength=nc)*h**3
    centroid=np.column_stack([np.bincount(owner,weights=centres[:,a],minlength=nc) for a in range(3)])/(volume/h**3)[:,None]
    labels=owner.reshape(mask.shape);indices=np.arange(mask.size).reshape(mask.shape)
    left=[];right=[];face_xyz=[];normals=[]
    for axis in range(3):
        lo=[slice(None)]*3;hi=[slice(None)]*3;lo[axis]=slice(0,-1);hi[axis]=slice(1,None)
        a=indices[tuple(lo)].ravel();b=indices[tuple(hi)].ravel();keep=owner[a]!=owner[b]
        a=a[keep];b=b[keep]
        normal=np.zeros((len(a),3));normal[:,2-axis]=1
        left.append(owner[a]);right.append(owner[b]);normals.append(normal)
        face_xyz.append(centres[a]+.5*h*normal)
    left=np.concatenate(left);right=np.concatenate(right)
    edge_keys=left.astype(np.int64)*nc+right
    edges,inverse=np.unique(edge_keys,return_inverse=True)
    return {'owner':owner,'centres':centres,'volume':volume,'centroid':centroid,
            'left':left,'right':right,'face_centres':np.concatenate(face_xyz),
            'normals':np.concatenate(normals),'edge_inverse':inverse,'edge_count':len(edges),'sites':s}

def aggregate(values,g,h,face):
    owner=g['owner'];volume=g['volume'];nc=len(volume)
    def group(v):
        if v.ndim==1:return np.bincount(owner,weights=v,minlength=nc)
        return np.column_stack([np.bincount(owner,weights=v[:,a],minlength=nc) for a in range(v.shape[1])])
    ui=group(values['u']);pi=group(values['p']);fi=group(values['f']);fx=group(values['fx'])
    u=ui/volume[:,None];p=pi/volume;p-=np.sum(p*volume)/volume.sum()
    f=fi/volume[:,None]
    moment=fx.reshape(nc,3,3)-fi[:,:,None]*g['centroid'][:,None,:]
    flux=np.bincount(g['edge_inverse'],weights=face,minlength=g['edge_count'])
    # Same linear load functional as the source: V f_bar . R(z) + M:G(z).
    # Report the physical-facelet coefficient vector before patch compression;
    # do not call this a bound on the final Stokes solution.
    a=g['left'];b=g['right'];normal=g['normals'];point=g['face_centres']
    def local(c,norm):
        r=point-g['centroid'][c]
        return h*h*(norm*np.sum(r*f[c],axis=1)[:,None]+np.einsum('nij,nj->ni',moment[c],norm)/volume[c,None])
    load=local(a,normal)+local(b,-normal)
    return {'cell_velocity':u,'cell_pressure':p,'cell_force':f,'cell_force_first_moment':moment,
            'facelet_flux':face,'aggregate_flux':flux,'facelet_load_coefficients':load}

def main():
    manifest=check_input_manifest()
    unique={}
    for row in manifest['mms_studies']:
        key=(int(row['voxels_per_side']),int(row['sites_per_side']),float(row['voxel_size']),float(row['viscosity']))
        unique.setdefault(key,[]).append(row)
    nkeys=sorted({(n,h,nu) for n,m,h,nu in unique})
    result=[]
    for n,h,nu in nkeys:
        print('QUAD_BEGIN',n,h,nu,flush=True)
        started=time.perf_counter();manufactured=Manufactured(n*h,nu)
        z,y,x=np.indices((n,n,n));centres=np.column_stack([x.ravel()+.5,y.ravel()+.5,z.ravel()+.5])*h
        def fx(X,Y,Z):
            f=manufactured.forcing(X,Y,Z);point=np.stack([X,Y,Z],axis=-1)
            return (f[..., :,None]*point[...,None,:]).reshape(f.shape[:-1]+(9,))
        raw={}
        for order in ORDERS:
            raw[order]={}
            for name,fn in [('u',manufactured.velocity),('p',manufactured.pressure),('f',manufactured.forcing),('fx',fx)]:
                raw[order][name]=voxel_integrals(fn,centres,h,order)
            print('QUAD_VOXELS',n,order,round(time.perf_counter()-started,2),flush=True)
        for key,oldrows in sorted(unique.items()):
            if (key[0],key[2],key[3])!=(n,h,nu):continue
            m=key[1];g=geometry(n,m,h)
            if any(int(r['N_cv'])!=len(g['volume']) or int(r['N_facelets'])!=len(g['left']) for r in oldrows):
                raise AssertionError('Geometry counts differ from manufactured source rows')
            agg={}
            for order in ORDERS:
                face=face_integrals(manufactured.velocity,g['face_centres'],g['normals'],h,order)
                agg[order]=aggregate(raw[order],g,h,face)
            error_floor={name:min(float(r[col])/100 for r in oldrows) for name,col in [('cell_velocity','e_u_percent'),('cell_pressure','e_p_percent'),('aggregate_flux','e_phi_percent')]}
            comparisons={}
            for order in ORDERS[:-1]:
                comparison={}
                for name,value in agg[order].items():
                    weights=g['volume'] if name in ['cell_velocity','cell_pressure','cell_force'] else None
                    comparison[name]=metrics(value,agg[7][name],weights)
                    if name in error_floor:
                        comparison[name]['fraction_of_smallest_reported_discrete_error']=comparison[name]['relative_l2']/error_floor[name]
                for name,value in raw[order].items():comparison['voxel_'+name]=metrics(value,raw[7][name])
                comparisons[str(order)]=comparison
            passed_error=all(comparisons['5'][k]['fraction_of_smallest_reported_discrete_error']<=REFERENCE_ERROR_FRACTION for k in error_floor)
            passed_force=all(comparisons['5'][k]['relative_l2']<=FORCE_RELATIVE_TOLERANCE for k in ['cell_force','cell_force_first_moment','facelet_load_coefficients'])
            stable_high=all(item['relative_l2']<=FORCE_RELATIVE_TOLERANCE for item in comparisons['6'].values())
            row={'n':n,'m':m,'h':h,'viscosity':nu,'N_cells':len(g['volume']),'N_facelets':len(g['left']),
                 'reference_order':7,'tabulated_order':5,'comparisons':comparisons,'tabulated_rows':[r['study']+'/'+r['level_label'] for r in oldrows],
                 'acceptance':{'reference_change_below_fraction_of_discrete_error':passed_error,'force_and_moment_relative_change_small':passed_force,'orders_6_7_agree':stable_high},
                 'elapsed_n_group_s':time.perf_counter()-started}
            result.append(row);save(ROOT/'out/quadrature'/f'n{n}_m{m}.json',row)
            print('QUAD_CELL',n,m,row['acceptance'],flush=True)
        del raw
    allpass=all(all(r['acceptance'].values()) for r in result)
    # A deliberate 1e-5 local perturbation is visible despite unchanged global
    # pressure sum, proving the check is sensitive to cancelling local errors.
    ref=np.array([1.,-1.,2.,-2.]);perturbed=ref+np.array([1e-5,-1e-5,0.,0.])
    negative=metrics(perturbed,ref)
    assert perturbed.sum()==ref.sum() and negative['relative_l2']>FORCE_RELATIVE_TOLERANCE
    save(ROOT/'out/quadrature.json',{'host':socket.gethostname(),'rows':result,'orders':ORDERS,
         'acceptance_thresholds':{'reference_error_fraction':REFERENCE_ERROR_FRACTION,'force_relative':FORCE_RELATIVE_TOLERANCE},
         'cancelling_local_error_negative_control':negative,'all_passed':allpass,
         'scope':'All unique geometries from the two retained MMS ladders; the Gauss routines used for the tabulated rows and source-exported symbolic fields. No new Stokes solve. Facelet load coefficients checked before patch-basis compression; no solution-sensitivity bound inferred.'})
    save(ROOT/'out/quadrature_complete.json',{'complete':len(result)==len(unique) and allpass,'unique_geometries':len(result),'host':socket.gethostname(),'all_passed':allpass})
    if not allpass:raise AssertionError('Quadrature needs further investigation; preserve existing evidence')

if __name__=='__main__':main()
