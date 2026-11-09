"""Wait for the unchanged six-run queue, aggregate, and package its evidence."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = Path('/workspace/b3_generation')
OUT = ROOT / 'results/b3/generation_followup_v1'
STATUS = ROOT / 'finalization_status.json'


def write_status(**fields):
    STATUS.write_text(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), **fields}, indent=2) + '\n')


if __name__ == '__main__':
    os.chdir(ROOT)
    write_status(status='WAITING_FOR_SIX_CANDIDATES')
    while True:
        state = json.loads((OUT / 'execution_manifest.json').read_text())
        if state['status'] == 'PASS_SIX_CANDIDATES':
            break
        if state['status'] != 'RUNNING':
            write_status(status='QUEUE_DID_NOT_PASS', queue_status=state['status'])
            raise SystemExit(1)
        time.sleep(30)
    environment = os.environ.copy()
    environment['PYTHONPATH'] = str(ROOT / 'src')
    for language in ['fr', 'zh']:
        for candidate in ['host_baseline', 'selected', 'published_reference']:
            path = OUT / language / candidate
            run = json.loads((path / 'run.json').read_text())
            assert run['status'] == 'PASS_GENERATION' and run['completed_records'] == 250
            assert len((path / 'generations.jsonl').read_text().splitlines()) == 250
        summary = OUT / language / 'summary.json'
        if not summary.exists():
            write_status(status='AGGREGATING', language=language)
            with (ROOT / f'aggregate_{language}.stdout.log').open('w') as stream:
                subprocess.run([sys.executable, 'scripts/run_b3_generation_followup.py', 'aggregate', '--language', language], env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
        assert json.loads(summary.read_text())['status'] == 'PASS_AGGREGATE'
    orchestration = OUT / 'orchestration'
    orchestration.mkdir(exist_ok=True)
    for name in ['run_generation_queue.py', 'finalize_generation.py', 'generation_queue.log', 'aggregate_fr.stdout.log', 'aggregate_zh.stdout.log']:
        source = ROOT / name
        if source.exists():
            shutil.copyfile(source, orchestration / name)
    hashes = {str(p.relative_to(OUT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'remote_bundle_files.json'}
    (OUT / 'remote_bundle_files.json').write_text(json.dumps({'status': 'PASS_PACKAGE_COUNTS', 'n_generation_rows': 1500, 'files_sha256': hashes}, indent=2) + '\n')
    bundle = Path('/workspace/b3_generation_final.tgz')
    write_status(status='PACKAGING')
    with tarfile.open(bundle, 'w:gz') as archive:
        archive.add(OUT, arcname='results/b3/generation_followup_v1')
    write_status(status='PASS_FINAL_BUNDLE', bundle=str(bundle), bytes=bundle.stat().st_size, sha256=hashlib.sha256(bundle.read_bytes()).hexdigest(), n_generation_rows=1500)
