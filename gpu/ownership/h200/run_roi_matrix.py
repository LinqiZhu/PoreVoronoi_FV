import csv, hashlib, json, platform, socket, time
import numpy as np
from common import ROOT, save, array_meta, check_input_manifest, strict_owner
import cupy as cp
import ptv_ownership_gpu6 as G
import ptv_ownership_gpu6_sm_query as G_sm_query

PARAMS={'block_size':256,'closure_block_size':64,'lower_bound_method':'jump_cooperative',
        'certificate_method':'path_forward','cooperative_blocks_per_sm':4,
        'lower_cooperative_blocks_per_sm':8,'return_float64':False}
WARMUPS=5;REPEATS=51

def compare(output,mask,pore,owner,distance):
    label=cp.asnumpy(output[0]).ravel()[pore]
    dist=cp.asnumpy(output[1]).ravel()[pore]
    return {'owner_mismatches':int(np.count_nonzero(label!=owner)),
            'distance_mismatches':int(np.count_nonzero(dist!=distance)),
            'max_abs_distance_difference':float(np.max(np.abs(dist-distance)))}

def timing(fn):
    cp.cuda.Stream.null.synchronize();t=time.perf_counter();v=fn();cp.cuda.Stream.null.synchronize()
    return v,time.perf_counter()-t

def one(case,count,scale):
    path=ROOT/'out/roi'/f'{case}_n{count}_s{scale}.json'
    if path.exists():return json.loads(path.read_text())
    with np.load(ROOT/'inputs'/f'{case}.npz',allow_pickle=False) as z:mask=z['mask'].astype(bool)
    with np.load(ROOT/'inputs'/f'{case}__mask_graph_fps_n{count}__ownership.npz',allow_pickle=False) as z:sites=z['sites']
    native_shape=mask.shape
    coords=np.column_stack(np.unravel_index(sites,mask.shape))
    if scale==2:
        for axis in range(3):mask=np.repeat(mask,2,axis=axis)
        coords=2*coords+1
        sites=np.ravel_multi_index(coords.T,mask.shape)
    p,s,owner,distance,nb=strict_owner(mask,sites)
    if np.any(owner<0):raise AssertionError('unseeded pore component')
    mask_cp=cp.asarray(mask,dtype=cp.uint8);seeds_cp=cp.asarray(np.column_stack(np.unravel_index(s,mask.shape)),dtype=cp.int32)
    def full():return G.exact_frontier_dijkstra_gpu_6(mask_cp,seeds_cp,validate=False,return_float64=False)
    def roi():return G.certified_l1_roi_frontier_gpu_6(mask_cp,seeds_cp,validate=False,pore_voxel_count=len(p),**PARAMS)
    def roi_sm_query():return G_sm_query.certified_l1_roi_frontier_gpu_6(mask_cp,seeds_cp,validate=False,pore_voxel_count=len(p),**PARAMS)
    _,first_full=timing(full);_,first_roi=timing(roi)
    ref=G.exact_frontier_dijkstra_gpu_6(mask_cp,seeds_cp,validate=True,return_float64=False)
    candidate=G.certified_l1_roi_frontier_gpu_6(mask_cp,seeds_cp,validate=True,return_stats=True,profile_stages=True,pore_voxel_count=len(p),**PARAMS)
    comparison={'full_vs_CPU':compare(ref,mask,p,owner,distance),'ROI_vs_CPU':compare(candidate,mask,p,owner,distance)}
    comparison['roi_sm_query_vs_CPU']=compare(roi_sm_query(),mask,p,owner,distance)
    if any(v['owner_mismatches'] or v['distance_mismatches'] for v in comparison.values()):
        save(path.with_name(path.stem+'_FAILED.json'),comparison);raise AssertionError('ownership mismatch '+str(comparison))
    specialization=None
    if mask.shape==(64,64,64):
        bits,prep=timing(lambda:G.prepare_mask_line_bits64(mask_cp))
        special=G.certified_roi_jfa_gpu_6(mask_cp,seeds_cp,validate=True,return_stats=True,profile_stages=True,certificate_line_bits=bits,pore_voxel_count=len(p))
        specialization={'comparison':compare(special,mask,p,owner,distance),'cache_preparation_s':prep,'stats':special[2]}
        if specialization['comparison']['owner_mismatches'] or specialization['comparison']['distance_mismatches']:
            raise AssertionError('64-cube specialization mismatch')
    for _ in range(WARMUPS):full();roi();roi_sm_query()
    times={'full':[],'roi':[],'roi_sm_query':[]}
    for k in range(REPEATS):
        calls=[('full',full),('roi_sm_query',roi_sm_query),('roi',roi)]
        cycle=calls[k%3:]+calls[:k%3]
        if k%2:cycle=cycle[::-1]
        for name,fn in cycle:
            _,seconds=timing(fn);times[name].append(seconds)
    ratios=np.asarray(times['full'])/np.asarray(times['roi'])
    rng=np.random.default_rng(20260911)
    samples=np.median(ratios[rng.integers(0,REPEATS,size=(10000,REPEATS))],axis=1)
    row={'case':case,'sites':count,'scale':scale,'native_shape':list(native_shape),'shape':list(mask.shape),
         'pore_voxels':len(p),'mask':array_meta(mask),'site_ids':array_meta(s),'configuration':PARAMS,
         'comparison':comparison,'roi_stats':candidate[2],'specialization_64':specialization,
         'first_call_s':{'full':first_full,'roi':first_roi},'warmups':WARMUPS,'repeats':REPEATS,
         'times_s':times,'roi_sm_query_median_s':float(np.median(times['roi_sm_query'])),'full_over_roi_sm_query_median':float(np.median(np.array(times['full'])/np.array(times['roi_sm_query']))),'paired_speedup_median':float(np.median(ratios)),
         'paired_speedup_bootstrap_95_interval':np.quantile(samples,[.025,.975]).tolist(),
         'roi_faster_fraction':float(np.mean(ratios>1)),
         'median_full_s':float(np.median(times['full'])),'median_roi_s':float(np.median(times['roi']))}
    save(path,row)
    print('ROI',case,count,scale,list(mask.shape),'speedup',row['paired_speedup_median'],'certified',row['roi_stats']['certified_fraction'],flush=True)
    del mask_cp,seeds_cp,ref,candidate
    cp.get_default_memory_pool().free_all_blocks()
    return row

def main():
    check_input_manifest()
    properties=cp.cuda.runtime.getDeviceProperties(0)
    device=properties['name'];device=device.decode() if isinstance(device,bytes) else str(device)
    rows=[]
    for case in ['thin_wall','maze','bentheimer_crop','berea_heldout_64']:
        for count in [200,800]:
            for scale in [1,2]:rows.append(one(case,count,scale))
    meta={'host':socket.gethostname(),'device':device,'numpy':np.__version__,'cupy':cp.__version__,
          'module_difference':'ptv_ownership_gpu6.py caches the hardware SM count once per device; ptv_ownership_gpu6_sm_query.py queries it at each launch; all numerical kernels identical',
          'sm_query_module_sha256':hashlib.sha256((ROOT/'src/ptv_ownership_gpu6_sm_query.py').read_bytes()).hexdigest(),
          'module_sha256':hashlib.sha256((ROOT/'src/ptv_ownership_gpu6.py').read_bytes()).hexdigest(),
          'timing_boundary':'synchronized GPU-resident ownership call: exact lower bound, path certificate, closure, output allocation/materialization; warm JIT and memory pool; no file load, transfers, optional correctness audit, FV enumeration, Stokes assembly or solve',
          'matrix_path':'shape-general path_forward certificate; production 64-cube specialization verified separately where applicable',
          'refinement':'2x repeat of voxel masks in each axis; fixed mapped sites 2*coordinate+1; same voxel-union geometry, not new independent material samples'}
    save(ROOT/'out/roi.json',{'metadata':meta,'rows':rows})
    save(ROOT/'out/roi_complete.json',{'complete':len(rows)==16,'cases':len(rows),'host':socket.gethostname(),'device':device,'all_CPU_matches':all(all(v['owner_mismatches']==0 and v['distance_mismatches']==0 for v in r['comparison'].values()) for r in rows)})

if __name__=='__main__':main()
