from pathlib import Path
from collections import deque
import hashlib, heapq, io, json, os, socket, sys
import numpy as np

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'src'))
os.environ['PVFV_STUDIES_DIR']=str(ROOT/'src')
import study_common as ec
INF=int(ec._INF)

def save(path,data):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.part')
    def native(value):
        if isinstance(value,np.generic):return value.item()
        raise TypeError('Unsupported JSON value: '+type(value).__name__)
    temp.write_text(json.dumps(data,indent=2,allow_nan=False,default=native),encoding='utf-8')
    temp.replace(p)

def array_meta(a):
    a=np.ascontiguousarray(a)
    return {'shape':list(a.shape),'dtype':str(a.dtype),'data_sha256':hashlib.sha256(a.tobytes()).hexdigest()}

def heap_reference(neighbours,site_nodes):
    """Independent lexicographic shortest-path oracle; no GPU code is reused."""
    n=len(neighbours);d=np.full(n,INF,dtype=np.int64);owner=np.full(n,-1,dtype=np.int64);q=[]
    for label,node in enumerate(site_nodes):
        d[node]=0;owner[node]=label;heapq.heappush(q,(0,label,int(node)))
    while q:
        dist,label,node=heapq.heappop(q)
        if dist!=d[node] or label!=owner[node]:continue
        for other in neighbours[node]:
            if other<0:continue
            if dist+1<d[other] or (dist+1==d[other] and label<owner[other]):
                d[other]=dist+1;owner[other]=label
                heapq.heappush(q,(dist+1,label,int(other)))
    return d,owner

def strict_owner(mask,sites,periodic_x=False):
    pore,neighbours=ec.pore_neighbour_table(mask,periodic_x=periodic_x)
    sites=np.unique(np.asarray(sites,dtype=np.int64))
    lookup=np.full(mask.size,-1,dtype=np.int64);lookup[pore]=np.arange(len(pore))
    if np.any(lookup[sites]<0):raise ValueError('solid site')
    d,o=heap_reference(neighbours,lookup[sites])
    return pore,sites,o,d,neighbours

def check_input_manifest():
    manifest=json.loads((ROOT/'inputs_manifest.json').read_text())
    checks=[]
    for name,record in manifest['source_files'].items():
        actual=hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()
        checks.append({'kind':'source','name':name,'match':actual==record['sha256']})
    for name,digest in manifest['driver_sha256'].items():
        checks.append({'kind':'driver','name':name,'match':hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest})
    for case,record in manifest['masks'].items():
        with np.load(ROOT/'inputs'/f'{case}.npz',allow_pickle=False) as z:
            actual=array_meta(z['mask'])
        expected=record['members']['mask.npy']
        checks.append({'kind':'mask','name':case,'match':actual['data_sha256']==expected['data_sha256'] and actual['shape']==list(expected['shape'])})
    for record in manifest['ownership_fields']:
        with np.load(ROOT/'inputs'/record['filename'],allow_pickle=False) as z:actual=array_meta(z['sites'])
        checks.append({'kind':'sites','name':record['filename'],'match':actual['data_sha256']==record['members']['sites.npy']['data_sha256']})
    save(ROOT/'out/gates'/('input_gate_'+os.environ.get('SLURM_JOB_ID','unscheduled')+'.json'),{'host':socket.gethostname(),'checks':checks,'passed':all(x['match'] for x in checks),'source_export':manifest['symbolic_export']})
    if not all(x['match'] for x in checks):raise AssertionError('Input/source hash mismatch')
    return manifest
