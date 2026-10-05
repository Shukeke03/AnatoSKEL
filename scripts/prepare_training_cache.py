"""Create a bounded, train-only physical-resampling cache for repeated optimization."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import torch

from anatoskel.config import load_config
from anatoskel.data import coarse_resample, load_scan, read_manifest
from anatoskel.data.scans import ensure_label_bounds
from anatoskel.training.pipeline import coarse_scan


def prepare_one(task):
    record, destination, manifest_hash, data, threads = task
    torch.set_num_threads(threads)
    path = Path(destination) / f"{record.scan_id}.pt"
    if path.exists():
        old = torch.load(path, weights_only=True, mmap=True)
        if old['manifest_sha256'] == manifest_hash and old['data_config'] == data:
            return {'scan_id':record.scan_id, 'status':'cached', 'bytes':path.stat().st_size}
        raise ValueError(f"Stale cache: {path}")
    start = time.perf_counter()
    native = load_scan(record)
    coarse = coarse_scan(native, data)
    scan = coarse_resample(native, data['local_spacing_mm'])
    del native
    scan['volume'] = scan['volume'].half()
    scan['labels'] = scan['labels'].to(torch.uint8)
    coarse['labels'] = coarse['labels'].to(torch.uint8)
    # Cache only a lookup of tight bounds; no mask/point cloud per full volume.
    ensure_label_bounds(scan)
    payload = {'manifest_sha256':manifest_hash, 'data_config':data, 'scan':scan, 'coarse':coarse,
               'policy':'full_FOV_common_physical_spacing; fp16_CT_storage; uint8_labels'}
    temporary = path.with_suffix('.tmp')
    torch.save(payload, temporary)
    temporary.replace(path)
    return {'scan_id':record.scan_id, 'seconds':time.perf_counter()-start,
            'bytes':path.stat().st_size, 'shape_dhw':list(scan['labels'].shape)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    if args.workers < 1 or args.threads < 1:
        parser.error('workers and threads must be positive')
    destination=Path(args.output)
    destination.mkdir(parents=True,exist_ok=True)
    data=load_config(args.config)['data']
    manifest_hash=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()
    records=[r for r in read_manifest(args.manifest) if r.split=='train']
    tasks=[(r,str(destination),manifest_hash,data,args.threads)for r in records]
    total_bytes=0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i,result in enumerate(pool.map(prepare_one,tasks),1):
            total_bytes+=result['bytes']
            print(json.dumps(dict(result,completed=i,total=len(tasks))),flush=True)
    report={'status':'complete','scans':len(records),'bytes':total_bytes,
            'manifest_sha256':manifest_hash,'data_config':data,'split':'train'}
    (destination/'cache.json').write_text(json.dumps(report,indent=2))
    print('CACHE_COMPLETE',flush=True)


if __name__=='__main__':
    main()
