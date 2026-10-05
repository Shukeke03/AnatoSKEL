"""Compact console status; suitable for watch -n 5."""
import json
from pathlib import Path
import subprocess

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
for name in ('run','cache'):
    path=root/'logs'/f'{name}.log'
    if path.exists():
        with path.open('rb') as stream:
            stream.seek(max(0,path.stat().st_size-4000))
            lines=stream.read().decode(errors='replace').splitlines()
        print(name+':', '\n'.join(lines[-2:]))
subprocess.run(['nvidia-smi','--query-gpu=name,utilization.gpu,memory.used,memory.total,power.draw',
                '--format=csv,noheader'])
