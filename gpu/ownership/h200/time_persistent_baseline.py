"""Unmodified ownership methods, matched inputs and synchronized call boundary.

All oracle, GPU timing, statistics and result generation execute on one NVIDIA H200 node.
"""
from pathlib import Path
import csv, hashlib, json, os, platform, socket, time, traceback
import numpy as np
from common import ROOT, save, array_meta, strict_owner
import cupy as cp
import ptv_ownership_gpu6 as G
from hybrid_site_sources import load_particle_window_seed_flat

PARAMS={'block_size':256,'closure_block_size':64,'lower_bound_method':'jump_cooperative',
        'certificate_method':'path_forward','cooperative_blocks_per_sm':4,
        'lower_cooperative_blocks_per_sm':8,'return_float64':False}
PERSISTENT={'block_size':256,'blocks_per_sm':1,'return_float64':False}
METHODS=['host_frontier','persistent_frontier','roi']

def timed(fn):
    cp.cuda.Stream.null.synchronize()
    start=time.perf_counter();out=fn();cp.cuda.Stream.null.synchronize()
    return out,time.perf_counter()-start

def input_gate():
    expected=json.loads((ROOT/'input_hashes.json').read_text())
    checks=[]
    for path,digest in expected.items():
        actual=hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
        checks.append({'path':path,'expected_sha256':digest,'actual_sha256':actual,'match':digest==actual})
    save(ROOT/'out/input_gate.json',{'checks':checks,'passed':all(c['match'] for c in checks)})
    assert all(c['match'] for c in checks),'source/input manifest mismatch'
    return expected

def compare(output,mask,pore,owner,distance):
    label=cp.asnumpy(output[0]);dist=cp.asnumpy(output[1])
    flat_l=label.ravel()[pore];flat_d=dist.ravel()[pore]
    oi=np.flatnonzero(flat_l!=owner);di=np.flatnonzero(flat_d!=distance)
    solid=~mask
    result={'owner_mismatches':int(len(oi)),'distance_mismatches':int(len(di)),
            'solid_owner_mismatches':int(np.count_nonzero(label[solid]!=-1)),
            'solid_distance_mismatches':int(np.count_nonzero(~np.isposinf(dist[solid]))),
            'output_owner_dtype':str(label.dtype),'output_distance_dtype':str(dist.dtype),
            'output_shape_match':list(label.shape)==list(mask.shape) and list(dist.shape)==list(mask.shape),
            'owner_sha256':array_meta(label)['data_sha256'],'distance_sha256':array_meta(dist)['data_sha256']}
    result['passed']=(not len(oi) and not len(di) and not result['solid_owner_mismatches'] and
                      not result['solid_distance_mismatches'] and result['output_shape_match'] and
                      label.dtype==np.int32 and dist.dtype==np.float32)
    mismatches={'owner_pore_indices':pore[oi],'owner_expected':owner[oi],'owner_actual':flat_l[oi],
                'distance_pore_indices':pore[di],'distance_expected':distance[di],'distance_actual':flat_d[di]}
    return result,mismatches

def functions(mask_cp,seeds_cp,pore_count,bits=None):
    def full():return G.exact_frontier_dijkstra_gpu_6(mask_cp,seeds_cp,validate=False,return_float64=False)
    def persistent():return G.exact_persistent_frontier_gpu_6(mask_cp,seeds_cp,validate=False,**PERSISTENT)
    if bits is None:
        def roi():return G.certified_l1_roi_frontier_gpu_6(mask_cp,seeds_cp,validate=False,pore_voxel_count=pore_count,**PARAMS)
    else:
        def roi():return G.certified_roi_jfa_gpu_6(mask_cp,seeds_cp,block_size=256,closure_block_size=64,
                            validate=False,return_float64=False,certificate_line_bits=bits,pore_voxel_count=pore_count)
    return dict(zip(METHODS,[full,persistent,roi]))

def known_answers():
    tests=[]
    mask=np.ones((1,1,5),dtype=bool)
    tests.append(('equal_distance_tie',mask,np.array([0,4]),[0,0,0,1,1],[0,1,2,1,0]))
    mask=np.array([[[1,1,1],[0,0,1],[1,1,1]]],dtype=bool)
    tests.append(('obstacle_detour',mask,np.array([0]),[0]*7,[0,1,2,3,6,5,4]))
    mask=np.array([[[1,1,0,0,0,1,1]]],dtype=bool)
    tests.append(('disconnected_seeded_components',mask,np.array([0,6]),[0,0,1,1],[0,1,1,0]))
    rows=[]
    for name,mask,sites,expected_owner,expected_distance in tests:
        pore,s,owner,distance,_=strict_owner(mask,sites)
        assert np.array_equal(owner,expected_owner) and np.array_equal(distance,expected_distance),name+' oracle known-answer failure'
        mc=cp.asarray(mask,dtype=cp.uint8)
        sc=cp.asarray(np.column_stack(np.unravel_index(s,mask.shape)),dtype=cp.int32)
        comparison={};first={}
        for method,fn in functions(mc,sc,len(pore)).items():
            output,seconds=timed(fn);first[method]=seconds
            result,mismatch=compare(output,mask,pore,owner,distance);comparison[method]=result
            if not result['passed']:
                np.savez(ROOT/'out'/f'FAILED_{name}_{method}.npz',**mismatch)
        row={'case':name,'comparison':comparison,'first_validation_call_s':first,
             'persistent_stats':dict(G.EXACT_PERSISTENT_BFS6_LAST_STATS)}
        rows.append(row);save(ROOT/'out/known_answers.json',rows)
        assert all(v['passed'] for v in comparison.values()),name+' known-answer mismatch'
        assert row['persistent_stats']['backend']=='cooperative_persistent_frontier6','persistent fallback is not a measured cooperative baseline'
    print('KNOWN_ANSWERS_PASS',len(rows),flush=True)

def interval(values,seed=20260911,draws=20000):
    values=np.asarray(values);rng=np.random.default_rng(seed)
    boot=np.median(values[rng.integers(0,len(values),size=(draws,len(values)))],axis=1)
    return {'paired_median':float(np.median(values)),'bootstrap_95_interval':np.quantile(boot,[.025,.975]).tolist(),
            'numerator_slower_fraction':float(np.mean(values>1)),'bootstrap_samples':draws}

def measure(case_id,mask,sites,context,warmups,repeats,bits=None):
    pore,sites,owner,distance,_=strict_owner(mask,sites)
    assert np.all(owner>=0),'unseeded pore component: '+case_id
    mask_cp=cp.asarray(mask,dtype=cp.uint8)
    seeds_cp=cp.asarray(np.column_stack(np.unravel_index(sites,mask.shape)),dtype=cp.int32)
    # For paper rows reuse the same mask object paired with its prebuilt bit cache.
    if bits is not None:mask_cp,bits=bits
    calls=functions(mask_cp,seeds_cp,len(pore),bits)
    comparison={};first_validation={}
    for method,fn in calls.items():
        output,seconds=timed(fn);first_validation[method]=seconds
        result,mismatch=compare(output,mask,pore,owner,distance);comparison[method]=result
        if not result['passed']:
            np.savez(ROOT/'out'/f'FAILED_{case_id}_{method}.npz',**mismatch)
    persistent_stats=dict(G.EXACT_PERSISTENT_BFS6_LAST_STATS)
    save(ROOT/'out/gates'/f'{case_id}.json',{'comparison':comparison,'persistent_stats':persistent_stats})
    assert all(v['passed'] for v in comparison.values()),case_id+' CPU mismatch'
    assert persistent_stats['backend']=='cooperative_persistent_frontier6','persistent fallback unsupported'
    # Also exercise the production persistent fixed-point audit outside timing.
    validated=G.exact_persistent_frontier_gpu_6(mask_cp,seeds_cp,validate=True,return_stats=True,**PERSISTENT)
    validate_stats=validated[2]
    assert validate_stats['fixed_point_residuals']==0
    del validated,output
    for _ in range(warmups):
        for fn in calls.values():fn()
    samples={name:[] for name in METHODS};orders=[]
    for repeat in range(repeats):
        cycle=METHODS[repeat%3:]+METHODS[:repeat%3]
        if repeat%2:cycle=cycle[::-1]
        orders.append(cycle)
        for method in cycle:
            output,seconds=timed(calls[method]);samples[method].append(seconds)
            del output
    ratios={}
    for numerator,denominator in [('host_frontier','roi'),('persistent_frontier','roi'),('host_frontier','persistent_frontier')]:
        ratios[numerator+'_over_'+denominator]=interval(np.array(samples[numerator])/np.array(samples[denominator]))
    if bits is None:
        stats=G.certified_l1_roi_frontier_gpu_6(mask_cp,seeds_cp,validate=True,return_stats=True,
              profile_stages=True,pore_voxel_count=len(pore),**PARAMS)[2]
    else:
        stats=G.certified_roi_jfa_gpu_6(mask_cp,seeds_cp,validate=True,return_stats=True,profile_stages=True,
              block_size=256,closure_block_size=64,return_float64=False,certificate_line_bits=bits,pore_voxel_count=len(pore))[2]
    row={'case_id':case_id,**context,'shape':list(mask.shape),'pore_voxels':len(pore),'site_count':len(sites),
         'mask':array_meta(mask),'site_ids':array_meta(sites),'gpu_mask_dtype':'uint8','gpu_sites_dtype':'int32',
         'comparison':comparison,'persistent_stats':persistent_stats,'persistent_validation_stats':validate_stats,
         'first_validation_call_s':first_validation,'warmups':warmups,'repeats':repeats,'times_s':samples,
         'repeat_orders':orders,'medians_s':{k:float(np.median(v)) for k,v in samples.items()},'ratios':ratios,'roi_stats':stats}
    save(ROOT/'out/rows'/f'{case_id}.json',row)
    print('ROW',case_id,'persistent/roi',ratios['persistent_frontier_over_roi'],flush=True)
    return row

def reports(rows,metadata):
    with (ROOT/'out/summary.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['case_id','sites','host_ms','persistent_ms','roi_ms','persistent_over_roi','ci_low','ci_high','host_over_persistent'])
        for row in rows:
            ratio=row['ratios']['persistent_frontier_over_roi']
            writer.writerow([row['case_id'],row['site_count'],*[1000*row['medians_s'][m] for m in METHODS],ratio['paired_median'],*ratio['bootstrap_95_interval'],row['ratios']['host_frontier_over_persistent_frontier']['paired_median']])
    with (ROOT/'out/raw_repeats.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['case_id','repeat','position','method','seconds'])
        for row in rows:
            for repeat,order in enumerate(row['repeat_orders']):
                for position,method in enumerate(order):writer.writerow([row['case_id'],repeat,position,method,row['times_s'][method][repeat]])
    lines=['# Persistent-frontier comparison on one NVIDIA H200','',
           'All three retained methods were run without numerical-kernel changes. The independent CPU oracle and known-answer checks execute on the same node before any timing sample is recorded. The persistent method uses its documented defaults: 256 threads, one requested block per SM, capped by kernel occupancy. No hardware parameter search was performed.','',
           f"Job {metadata['job_id']}; host {metadata['host']}; GPU {metadata['device']}. Exact input/source hashes, backend details, output hashes and zero/mismatch counts are retained in JSON.",'',
           'Timing includes GPU-resident call allocation, initialization, method stages and output materialization, with synchronization immediately before and after every call. It excludes input loading/transfers, CPU validation, optional fixed-point validation, FV construction and Stokes. Warm JIT and allocator pool. The persistent wrapper retains its convergence/overflow checks, property/occupancy queries and scalar device-to-host reads. Thus this is a retained-implementation comparison, not isolated kernel timing.','',
           'The 16 geometry/scale rows use 5 warmups and 51 repeats per method; the five Table 2 trajectory prefixes use 10 warmups and 101 repeats. Three-method order rotates and reverses. Reported intervals are paired bootstrap intervals for the median within this run, not hardware-to-hardware or independent-run uncertainty. First validation calls are recorded separately and are not pure cold-start benchmarks because earlier cases warm shared kernels.','',
           'For the five paper prefixes only, the 64-cube mask-line cache of the Table 2 configuration is built once per mask before all three-method calls, reused by ROI, and excluded from warm ownership timings. Both first and warm cache preparation times are in metadata; no analogous cache is used by the two BFS methods. General matrix rows use the path-forward implementation without that bit cache.','',
           '| Case | Sites | Host BFS ms | Persistent BFS ms | ROI ms | Persistent / ROI (95% interval) |','|---|---:|---:|---:|---:|---:|']
    tex=['% Generated on one NVIDIA H200 from raw paired repetitions. Ratio > 1 favors ROI.',r'\begin{tabular}{lrrrrr}',r'\toprule',r'Case & Sites & Host BFS (ms) & Persistent BFS (ms) & ROI (ms) & Persistent/ROI \\',r'\midrule']
    for row in rows:
        ratio=row['ratios']['persistent_frontier_over_roi'];low,high=ratio['bootstrap_95_interval'];m=row['medians_s']
        lines.append(f"| {row['case_id']} | {row['site_count']} | {1000*m['host_frontier']:.3f} | {1000*m['persistent_frontier']:.3f} | {1000*m['roi']:.3f} | {ratio['paired_median']:.3f} [{low:.3f}, {high:.3f}] |")
        label=row['case_id'].replace('_',r'\_')
        tex.append(f"{label} & {row['site_count']} & {1000*m['host_frontier']:.3f} & {1000*m['persistent_frontier']:.3f} & {1000*m['roi']:.3f} & {ratio['paired_median']:.3f} [{low:.3f}, {high:.3f}] "+r'\\')
    tex.extend([r'\bottomrule',r'\end{tabular}'])
    favored=sum(row['ratios']['persistent_frontier_over_roi']['bootstrap_95_interval'][0]>1 for row in rows)
    lost=sum(row['ratios']['persistent_frontier_over_roi']['bootstrap_95_interval'][1]<1 for row in rows)
    lines.extend(['',f'Across 21 scoped rows, ROI is faster with its interval wholly above one in {favored} rows; persistent BFS is faster with its interval wholly below one in {lost} rows. The remaining {len(rows)-favored-lost} intervals include one.','',
                  'A persistent/ROI ratio below one is an ROI loss and must be reported as such. These results do not establish performance against every optimized external shortest-path implementation or universal algorithm superiority. They also do not attribute exact fractions of speedup to individual mechanisms, because implementations differ in work, synchronization and wrappers.',''])
    (ROOT/'out/RESULTS.md').write_text('\n'.join(lines))
    (ROOT/'out/publication_table.tex').write_text('\n'.join(tex)+'\n')

def main():
    (ROOT/'out').mkdir(exist_ok=True)
    input_manifest=input_gate()
    device_id=int(cp.cuda.runtime.getDevice());props=cp.cuda.runtime.getDeviceProperties(device_id)
    cooperative=int(cp.cuda.runtime.deviceGetAttribute(cp.cuda.runtime.cudaDevAttrCooperativeLaunch,device_id))
    device=props['name'];device=device.decode() if isinstance(device,bytes) else str(device)
    metadata={'host':socket.gethostname(),'job_id':os.environ['SLURM_JOB_ID'],'device':device,
              'python':platform.python_version(),'numpy':np.__version__,'cupy':cp.__version__,
              'driver_version':cp.cuda.runtime.driverGetVersion(),'runtime_version':cp.cuda.runtime.runtimeGetVersion(),
              'cooperative_launch':cooperative,'sm_count':int(props['multiProcessorCount']),
              'persistent_parameters':PERSISTENT,'matrix_roi_parameters':PARAMS,'input_manifest':input_manifest}
    save(ROOT/'out/metadata.json',metadata)
    assert cooperative==1,'Cooperative launch unsupported: no persistent benchmark recorded'
    known_answers()
    rows=[]
    for case in ['thin_wall','maze','bentheimer_crop','berea_heldout_64']:
        for count in [200,800]:
            for scale in [1,2]:
                with np.load(ROOT/'inputs'/f'{case}.npz') as archive:mask=archive['mask'].astype(bool)
                with np.load(ROOT/'inputs'/f'{case}__mask_graph_fps_n{count}__ownership.npz') as archive:sites=archive['sites']
                native=list(mask.shape);coords=np.column_stack(np.unravel_index(sites,mask.shape))
                if scale==2:
                    for axis in range(3):mask=np.repeat(mask,2,axis=axis)
                    sites=np.ravel_multi_index((2*coords+1).T,mask.shape)
                rows.append(measure(f'{case}_n{count}_s{scale}',mask,sites,{'matrix':'geometry','case':case,'scale':scale,'native_shape':native},5,51))
                cp.get_default_memory_pool().free_all_blocks()
    with np.load(ROOT/'../../../data/berea64/mask.npz') as archive:mask=archive['mask'].astype(bool)
    mc=cp.asarray(mask,dtype=cp.uint8)
    first_bits,first_cache=timed(lambda:G.prepare_mask_line_bits64(mc))
    del first_bits
    bits,warm_cache=timed(lambda:G.prepare_mask_line_bits64(mc))
    metadata['mask_cache']={'first_preparation_s':first_cache,'warm_preparation_s':warm_cache,'scope':'single 64-cube mask shared across all five Table 2 prefixes; excluded from warm ownership call'}
    for count,expected in [(10,45),(50,379),(100,900),(200,1831),(500,4452)]:
        sites,site_meta=load_particle_window_seed_flat(mask,ROOT/'../../../data/berea64/particle_tracks.csv.gz',particle_id_selection=f'0:{count-1}')
        assert len(sites)==expected,'Table 2 prefix site count mismatch'
        rows.append(measure(f'ptv{count}',mask,sites,{'matrix':'paper','particles':count,'site_metadata':site_meta},10,101,bits=(mc,bits)))
    reports(rows,metadata)
    save(ROOT/'out/results.json',{'metadata':metadata,'rows':rows})
    save(ROOT/'out/metadata.json',metadata)
    save(ROOT/'out/complete.json',{'complete':len(rows)==21,'case_count':len(rows),'all_CPU_matches':all(all(c['passed'] for c in row['comparison'].values()) for row in rows),
         'all_cooperative_persistent':all(row['persistent_stats']['backend']=='cooperative_persistent_frontier6' for row in rows),'job_id':os.environ['SLURM_JOB_ID'],'host':socket.gethostname()})
    print('BASELINE_COMPLETE',len(rows),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        save(ROOT/'out/failure.json',{'traceback':traceback.format_exc(),'host':socket.gethostname(),'job_id':os.environ.get('SLURM_JOB_ID')})
        raise
