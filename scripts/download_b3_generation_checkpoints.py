"""Fetch only public, immutable checkpoints for the separate generation study."""
from pathlib import Path
from huggingface_hub import snapshot_download
from datetime import datetime, timezone
import json

root = Path('/workspace/b3_generation')
contract = json.loads((root/'configs/b3_lighton_qwen3_8b_v1.json').read_text())
specs = [('en', contract['models']['donor'])]
for host in contract['models']['hosts']:
    specs.extend([(host['language'], host), (host['language']+'_published',host['published_swap'])])
report = {'started_at_utc': datetime.now(timezone.utc).isoformat(), 'checkpoints': {}}
for slug, spec in specs:
    target = root/'models'/slug
    print('DOWNLOAD',slug,spec['id'],spec['revision'],flush=True)
    snapshot_download(repo_id=spec['id'], revision=spec['revision'], local_dir=str(target),
                      allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','README.md'], max_workers=4)
    report['checkpoints'][slug] = {'id':spec['id'],'revision':spec['revision'],'path':str(target)}
    (root/'download_manifest.json').write_text(json.dumps(report,indent=2))
    print('COMPLETE',slug,flush=True)
report['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
(root/'download_manifest.json').write_text(json.dumps(report,indent=2))
print('ALL_DOWNLOADS_COMPLETE',flush=True)
