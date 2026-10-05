"""Unattended first real-data run: prior, training, held-out tests, one Git demo.

No raw CTs, checkpoints, caches or full per-case result trees are published.
The example CT is fixed before training from published test centroid metadata,
and inference receives a CT-only manifest; it cannot use ground-truth ROIs.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

from anatoskel.config import load_config
from anatoskel.data import read_manifest
from anatoskel.training.checkpoint import load_checkpoint


def stamp():
    return datetime.now(timezone.utc).isoformat()


def choose_example(manifest, destination):
    root=Path(manifest).resolve().parent
    rows=[json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip()]
    candidates=[]
    for row in sorted(rows,key=lambda r:r['scan_id']):
        if row['split']!='test':
            continue
        labels=set()
        for path in row.get('centroid_annotations',[]):
            for item in json.loads((root/path).read_text()):
                if 'label' in item:
                    labels.add(item['label'])
        if len(labels & {20,21,22,23,24,25})>=4:
            candidates.append(row)
    if not candidates:
        raise ValueError('No predeclared test example with four lumbar centroids')
    row=candidates[0]
    selected={k:row[k] for k in ('patient_id','scan_id','split','ct')}
    selected['ct']=str((root/selected['ct']).resolve())
    if 'ct_spatial_units_override' in row:
        selected['ct_spatial_units_override']=row['ct_spatial_units_override']
    destination.write_text(json.dumps(selected)+'\n')
    return {'scan_id':row['scan_id'], 'patient_id':row['patient_id'],
            'selection':'first sorted official test scan with >=4 published lumbar centroids; fixed before optimization',
            'inference_inputs':'CT and affine only; no labels, GT centroids or GT ROIs'}


def aggregate_metrics(path):
    report=json.loads(Path(path).read_text())
    grouped=defaultdict(list)
    for bone in report['bones']:
        grouped[str(bone['label_id'])].append(bone['dice'])
    return {'mean_dice':report['mean_dice'],'evaluated_bones':len(report['bones']),
            'scans_with_lumbar_ground_truth':len({b['scan_id'] for b in report['bones']}),
            'detected_gt_bones':sum(b['detected'] for b in report['bones']),
            'per_label_dice':{k:float(np.mean(v))for k,v in grouped.items()},
            'missed_bones_score_zero':True,'predicted_roi':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/autodl_5090.yaml')
    parser.add_argument('--manifest',default='data/verse/processed/manifest.jsonl')
    parser.add_argument('--output',default='outputs/first5090')
    parser.add_argument('--prior-epochs',type=int,default=100)
    parser.add_argument('--publish-demo',action='store_true')
    args=parser.parse_args()
    settings=load_config(args.config)
    run=Path(args.output).resolve()
    run.mkdir(parents=True,exist_ok=True)
    (run/'logs').mkdir(exist_ok=True)
    records=read_manifest(args.manifest)
    total_steps=settings['train']['epochs']*sum(r.split=='train'for r in records)
    status={'status':'running','started_utc':stamp(),'config':args.config,
            'planned_steps':total_steps,'gpu_batch_scan':1,
            'gpu_batch_local_bones_max':settings['train']['max_bones_per_step'],
            'manifest_sha256':hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest()}
    example=choose_example(args.manifest,run/'example_ct_only.jsonl')
    (run/'example_selection.json').write_text(json.dumps(example,indent=2))
    status['example']=example

    def write_status(**values):
        status.update(values,updated_utc=stamp())
        temporary=run/'status.tmp'
        temporary.write_text(json.dumps(status,indent=2))
        temporary.replace(run/'status.json')

    def command(stage,arguments):
        write_status(stage=stage)
        print(f'PHASE {stage} {stamp()}',flush=True)
        started=time.monotonic()
        env=None
        with (run/'logs'/f'{stage}.log').open('a') as log:
            process=subprocess.Popen(arguments,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                     text=True,bufsize=1,env=env)
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line,end='',flush=True)
                if stage=='train':
                    try:
                        record=json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(record,dict) and 'step' in record and 'coarse_loss' in record:
                        elapsed=time.monotonic()-started
                        base=status.get('resumed_step',0)
                        seconds=elapsed/max(1,record['step']-base)
                        write_status(step=record['step'],epoch=record['epoch']+1,
                                     coarse_loss=record['coarse_loss'],fine_loss=record['fine_loss'],
                                     fine_bones=record['fine_bones'],seconds_per_scan_running_mean=seconds,
                                     train_eta_hours=(total_steps-record['step'])*seconds/3600)
            if process.wait()!=0:
                raise RuntimeError(f'{stage} failed: inspect {run / "logs" / (stage + ".log")}')
        status.setdefault('phase_seconds',{})[stage]=time.monotonic()-started
        write_status()

    python=sys.executable
    cli=[python,'-u','-m','anatoskel.cli']
    try:
        write_status(stage='cache')
        cache=Path(settings['train']['cache_dir'])
        if not (cache/'cache.json').exists():
            # Reuse the independently started cache job; never race its file writes.
            while subprocess.run(['tmux','has-session','-t','anatoskel-cache'],
                                 stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:
                print('PHASE cache waiting for active anatoskel-cache job',flush=True)
                time.sleep(20)
            if not (cache/'cache.json').exists():
                command('cache',[python,'-u','scripts/prepare_training_cache.py',
                    '--manifest',args.manifest,'--config',args.config,'--output',str(cache),
                    '--workers','3','--threads','4'])
        command('doctor',cli+['doctor','--require-cuda'])
        atlas=run/'atlas.json'
        command('atlas',cli+['build-atlas','--manifest',args.manifest,'--output',str(atlas),
                             '--cache-dir',str(cache),'--label-ids',*map(str,settings['train']['fine_label_ids'])])
        prior=run/'prior'/'prior.pt'
        if not prior.exists():
            command('prior',cli+['train-prior','--config',args.config,'--manifest',args.manifest,
                '--atlas',str(atlas),'--epochs',str(args.prior_epochs),'--device','cuda',
                '--output',str(run/'prior')])
        command('prior_audit',[python,'scripts/validate_training_checkpoint.py',str(prior),
            '--device','cuda','--min-steps','10','--require-atlas','--amp',
            '--report',str(run/'prior_audit.json')])
        audit=load_checkpoint(prior)['real_training_sample_sign_audit']
        for label in settings['train']['fine_label_ids']:
            entry=audit.get(str(label))
            if entry is None or entry['bones_with_zero_crossing']!=entry['bones']:
                raise ValueError(f'Prior class {label} has missing/noncrossing real training bone fields')
        model=run/'train'/'model.pt'
        resume=['--resume',str(model)] if model.exists() else []
        write_status(resumed_step=load_checkpoint(model)['step'] if model.exists() else 0)
        command('train',cli+['train','--config',args.config,'--manifest',args.manifest,
            '--prior',str(prior),'--device','cuda','--output',str(run/'train'),*resume])
        command('model_audit',[python,'scripts/validate_training_checkpoint.py',str(model),
            '--device','cuda','--min-steps',str(total_steps),'--require-resumable','--amp',
            '--report',str(run/'model_audit.json')])
        for split in ('val','test'):
            command(split,cli+['evaluate','--checkpoint',str(model),'--manifest',args.manifest,
                '--split',split,'--device','cuda','--output',str(run/f'{split}.json')])
        command('example',cli+['infer','--checkpoint',str(model),'--manifest',str(run/'example_ct_only.jsonl'),
            '--split','test','--device','cuda','--output',str(run/'example')])
        source=run/'example'/example['scan_id']
        obj=source/'bones.obj'
        vertices=faces=0
        with obj.open() as stream:
            for line in stream:
                vertices+=line.startswith('v ')
                faces+=line.startswith('f ')
        summary={'run':'first5090','model':'AnatoSKEL','trained_on_real_CT':True,
            'epochs':settings['train']['epochs'],'steps':total_steps,'prior_epochs':args.prior_epochs,
            'example':dict(example,vertices=vertices,faces=faces,nonempty_model_prediction=faces>0),
            'validation':aggregate_metrics(run/'val.json'),'test':aggregate_metrics(run/'test.json'),
            'units':'mm','coordinate_system':'RAS','clinical_validation':False,
            'phase_seconds':status['phase_seconds'],'finished_utc':stamp(),
            'prediction_source':'learned model with predicted ROIs; no ground-truth mesh substitution'}
        (run/'summary.json').write_text(json.dumps(summary,indent=2))
        if args.publish_demo:
            command('git_sync',['git','fetch','ssh://git@ssh.github.com:443/Shukeke03/AnatoSKEL.git','main'])
            command('git_ff',['git','merge','--ff-only','FETCH_HEAD'])
            demo=Path('examples/first5090')
            demo.mkdir(parents=True,exist_ok=True)
            shutil.copy2(obj,demo/'bones.obj')
            metadata=json.loads((source/'bones.json').read_text())
            # Keep geometrical metadata, remove machine-specific paths.
            metadata.get('source',{}).pop('ct_path',None)
            metadata.get('source',{}).pop('checkpoint',None)
            (demo/'bones.json').write_text(json.dumps(metadata,indent=2))
            shutil.copy2(run/'summary.json',demo/'metrics.json')
            (demo/'README.md').write_text(
                '# First real CT model prediction\n\n'
                f'Fixed held-out case: `{example["scan_id"]}`. '
                f'Model prediction: {vertices} vertices, {faces} faces.\n\n'
                'OBJ coordinates are RAS millimetres; keep bones.json with the OBJ. '
                'metrics.json contains aggregate validation/test results. '
                'The case was fixed before optimization, not selected by its output quality. '
                'No CT, weights, cache or full result tree is included.\n\n'
                'This is an initial research experiment; anatomical pose axes and clinical '
                'accuracy have not been validated. Empty predictions, if present, are reported '
                'honestly and never replaced by a ground-truth mesh.\n\n'
                'Data attribution: [VerSe](https://github.com/anjany/verse), '
                '[dataset paper](https://doi.org/10.1016/j.media.2021.102166). '
                'This derived example is provided under '
                '[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).\n')
            if (demo/'bones.obj').stat().st_size>40*1024**2:
                raise ValueError('Demo OBJ exceeds the 40 MiB example limit; keep it on server')
            allowed=['bones.obj','bones.json','metrics.json','README.md']
            command('git_stage',['git','add','--',*[str(demo/name)for name in allowed]])
            staged=subprocess.check_output(['git','diff','--cached','--name-only'],text=True).splitlines()
            if not set(staged)<=set(str(demo/name)for name in allowed):
                raise ValueError('Unexpected staged paths; refusing to publish training outputs')
            command('git_commit',['git','commit','-m','Add first RTX 5090 held-out bone prediction and summary'])
            command('git_push',['git','push','origin','HEAD:main'])
            write_status(published_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
        write_status(status='complete',stage='complete',finished_utc=stamp(),
                     example_nonempty=faces>0,summary=str(run/'summary.json'))
        print('JOB_COMPLETE',flush=True)
    except Exception as error:
        write_status(status='failed',error=str(error))
        print(f'JOB_FAILED {error}',flush=True)
        raise


if __name__=='__main__':
    main()
