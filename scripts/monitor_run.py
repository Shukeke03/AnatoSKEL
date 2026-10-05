"""Compact console status; suitable for watch -n 5."""
import json
from pathlib import Path
import subprocess
import sys

root=Path('outputs/first5090')
status_path=root/'status.json'
status=json.loads(status_path.read_text()) if status_path.exists() else {'stage':'setting_up'}
print('AnatoSKEL:',status.get('status','running'),'stage=',status.get('stage'))
if 'step' in status:
    print(f"Epoch {status['epoch']}/100; step {status['step']}/{status['planned_steps']}; "
          f"local bones {status.get('fine_bones')}")
    print(f"Coarse loss {status.get('coarse_loss'):.4f}; fine loss {status.get('fine_loss'):.4f}; "
          f"training ETA {status.get('train_eta_hours',0):.2f} h (tests follow)")
if status.get('stage')=='cache':
    files=list(Path('data/verse/training-cache-v1').glob('*.pt'))
    print(f'Prepared scan caches: {len(files)}/141')
if status.get('stage')=='prior':
    prior_log=root/'logs'/'prior.log'
    if prior_log.exists():
        with prior_log.open('rb') as stream:
            stream.seek(max(0,prior_log.stat().st_size-4000))
            prior_lines=stream.read().decode(errors='replace').splitlines()
        for line in reversed(prior_lines):
            try:
                record=json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(record,dict):
                continue
            if record.get('stage')=='prior_samples':
                print(f"Prior CPU preparation: {record['scan']}/{record['scans']} scans; "
                      f"{record['bones']} bone samples. GPU optimization follows.")
                break
            if record.get('stage')=='prior_train':
                print(f"Prior GPU training: epoch {record['epoch']}/100; "
                      f"step {record['step']}; loss {record['loss']:.4f}")
                break
for name in ('run','cache'):
    path=root/'logs'/f'{name}.log'
    if path.exists():
        with path.open('rb') as stream:
            stream.seek(max(0,path.stat().st_size-4000))
            lines=stream.read().decode(errors='replace').splitlines()
        print(name+':', '\n'.join(lines[-2:]))
sys.stdout.flush()
subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw',
                '--format=csv,noheader'])
