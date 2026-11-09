"""Execute the six predeclared candidate evaluations, retaining every record."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import subprocess
import sys

ROOT=Path('/workspace/b3_generation')
OUT=ROOT/'results/b3/generation_followup_v1'
EXPECTED_RUNNER='1ab53cf1ddf42012e3b41c4a380690fb4fa587762df0274015972d4c86c00d69'
EXPECTED_CONFIG='785f600ad422910847c35dd1b9d82d5a7a2315fe798dc56a9b745c7aad9264c6'
IMAGE_DIGEST='sha256:d8d39b59e909d2378ac4feeb191f7e7b6f1342477dc66b7c47cec89e9985ad8a'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

if __name__=='__main__':
    os.chdir(ROOT)
    assert sha(ROOT/'src/suture/b3_generation_followup.py')==EXPECTED_RUNNER
    assert sha(ROOT/'configs/b3_generation_followup_v1.json')==EXPECTED_CONFIG
    OUT.mkdir(parents=True,exist_ok=True)
    for name in ['candidate_input_receipts.json','fr_candidate_provenance.json','zh_candidate_provenance.json','download_manifest.json','environment.lock','backend_smoke.json']:
        shutil.copyfile(ROOT/name,OUT/name)
    receipts=json.loads((ROOT/'candidate_input_receipts.json').read_text())
    state={'status':'RUNNING','started_at_utc':datetime.now(timezone.utc).isoformat(),
           'runner_sha256':EXPECTED_RUNNER,'config_sha256':EXPECTED_CONFIG,
           'image_digest':IMAGE_DIGEST,'candidate_order':[],'completed':[]}
    environment=os.environ.copy()
    environment.update({'PYTHONPATH':str(ROOT/'src'),'VLLM_NO_USAGE_STATS':'1','VLLM_WORKER_MULTIPROC_METHOD':'spawn','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1','PYTHONUNBUFFERED':'1'})
    for language in ['fr','zh']:
        for candidate in ['host_baseline','selected','published_reference']:
            model=receipts['languages'][language]['candidates'][candidate]['model_path']
            command=[sys.executable,'scripts/run_b3_generation_followup.py','run',
                     '--language',language,'--candidate',candidate,'--model-dir',model,
                     '--tokenizer-dir',str(ROOT/'models'/language),
                     '--manifest',str(ROOT/f'results/b3/lighton_qwen3_8b/{language}_host__en_donor/data/MGSM_rev2_test.jsonl'),
                     '--lid-model',str(ROOT/'models/lid.176.ftz'),'--output-root',str(OUT),
                     '--container-image-digest',IMAGE_DIGEST]
            label=language+'/'+candidate
            state['candidate_order'].append({'label':label,'command':command})
            state['current_candidate']=label
            (OUT/'execution_manifest.json').write_text(json.dumps(state,indent=2))
            print('QUEUE_START',label,datetime.now(timezone.utc).isoformat(),flush=True)
            with (OUT/(language+'_'+candidate+'.log')).open('w') as handle:
                result=subprocess.run(command,env=environment,stdout=handle,stderr=subprocess.STDOUT,timeout=5400)
            if result.returncode:
                state.update({'status':'FAILED','exit_code':result.returncode})
                (OUT/'execution_manifest.json').write_text(json.dumps(state,indent=2))
                raise SystemExit(result.returncode)
            state['completed'].append(label)
            (OUT/'execution_manifest.json').write_text(json.dumps(state,indent=2))
            print('QUEUE_COMPLETE',label,datetime.now(timezone.utc).isoformat(),flush=True)
    state.update({'status':'PASS_SIX_CANDIDATES','finished_at_utc':datetime.now(timezone.utc).isoformat()})
    (OUT/'execution_manifest.json').write_text(json.dumps(state,indent=2))
    print('ALL_SIX_CANDIDATES_COMPLETE',flush=True)
