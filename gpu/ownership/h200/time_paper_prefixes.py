"""The five Table 2 prefixes and call configuration, timed on one NVIDIA H200."""
from pathlib import Path
import hashlib,json,socket,time
import numpy as np
import cupy as cp
from common import ROOT,save,strict_owner,array_meta
import ptv_ownership_gpu6 as sm_cached
import ptv_ownership_gpu6_sm_query as sm_query
from hybrid_site_sources import load_particle_window_seed_flat

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
    manifest=json.loads((ROOT/'paper_inputs_manifest.json').read_text())
    for name,item in manifest['files'].items():assert digest(ROOT/name)==item['sha256'],name
    with np.load(ROOT/'../../../data/berea64/mask.npz') as z:mask=z['mask'].astype(bool)
    p=np.flatnonzero(mask);mc=cp.asarray(mask,dtype=cp.uint8)
    sm_cached.prepare_mask_line_bits64(mc);cp.cuda.Stream.null.synchronize()
    t=time.perf_counter();bits=sm_cached.prepare_mask_line_bits64(mc);cp.cuda.Stream.null.synchronize();cache_s=time.perf_counter()-t
    rows=[]
    for count,expected in [(10,45),(50,379),(100,900),(200,1831),(500,4452)]:
        sites,meta=load_particle_window_seed_flat(mask,ROOT/'../../../data/berea64/particle_tracks.csv.gz',particle_id_selection=f'0:{count-1}')
        assert len(sites)==expected
        p,s,owner,d,nb=strict_owner(mask,sites);sc=cp.asarray(np.column_stack(np.unravel_index(s,mask.shape)),dtype=cp.int32)
        def full():return sm_query.exact_frontier_dijkstra_gpu_6(mc,sc,validate=False,return_float64=False)
        def roi(module):return module.certified_roi_jfa_gpu_6(mc,sc,block_size=256,closure_block_size=64,validate=False,return_float64=False,certificate_line_bits=bits,pore_voxel_count=len(p))
        calls=[('full',full),('roi_sm_query',lambda:roi(sm_query)),('roi_sm_cached',lambda:roi(sm_cached))]
        comparison={}
        for name,fn in calls:
            out=fn();labels=cp.asnumpy(out[0]).ravel()[p];distance=cp.asnumpy(out[1]).ravel()[p]
            comparison[name]={'owner_mismatches':int(np.count_nonzero(labels!=owner)),'distance_mismatches':int(np.count_nonzero(distance!=d))}
        assert all(v['owner_mismatches']==0 and v['distance_mismatches']==0 for v in comparison.values())
        for _ in range(10):
            for name,fn in calls:fn()
        samples={name:[] for name,_ in calls}
        for repeat in range(101):
            cycle=calls[repeat%3:]+calls[:repeat%3]
            if repeat%2:cycle=cycle[::-1]
            for name,fn in cycle:
                cp.cuda.Stream.null.synchronize();t=time.perf_counter();fn();cp.cuda.Stream.null.synchronize();samples[name].append(time.perf_counter()-t)
        ratios={};rng=np.random.default_rng(20260717)
        for method in ['roi_sm_query','roi_sm_cached']:
            values=np.array(samples['full'])/np.array(samples[method]);boot=np.median(values[rng.integers(0,101,size=(20000,101))],axis=1)
            ratios[method]={'paired_median':float(np.median(values)),'bootstrap_95_interval':np.quantile(boot,[.025,.975]).tolist()}
        row={'particles':count,'sites':len(s),'site_ids':array_meta(s),'site_metadata':meta,'comparisons':comparison,'times_s':samples,
             'medians_s':{k:float(np.median(v)) for k,v in samples.items()},'full_over_roi':ratios}
        save(ROOT/'out/paper_roi'/f'ptv{count}.json',row);rows.append(row)
        print('PAPER_ROI',count,expected,row['medians_s'],ratios,flush=True)
    save(ROOT/'out/paper_roi.json',{'host':socket.gethostname(),'rows':rows,'warmups':10,'repeats':101,'bootstrap_samples':20000,
         'warm_line_cache_build_s':cache_s,'input_manifest':manifest,'device':'NVIDIA H200',
         'sm_query_module_sha256':digest(ROOT/'src/ptv_ownership_gpu6_sm_query.py'),'fixed_module_sha256':digest(ROOT/'src/ptv_ownership_gpu6.py'),
         'scope':'Original five trajectory prefixes, exact original loader and source input hashes; original 64-cube call configuration; compare original and SM-count-cached implementation on the same H200. No kernel or method retuning.'})
    save(ROOT/'out/paper_roi_complete.json',{'complete':len(rows)==5,'all_CPU_matches':True,'job_id':__import__('os').environ['SLURM_JOB_ID']})

if __name__=='__main__':main()
