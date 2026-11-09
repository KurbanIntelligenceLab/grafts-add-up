"""Bind the generation inputs to downloaded commits, LFS hashes and exact swaps."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json

def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda:handle.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--workspace',type=Path,default=Path('/workspace/b3_generation'))
    root=parser.parse_args().workspace.resolve()
    contract=json.loads((root/'configs/b3_lighton_qwen3_8b_v1.json').read_text())
    report={'created_at_utc':datetime.now(timezone.utc).isoformat(),'source_contract_sha256':sha256(root/'configs/b3_lighton_qwen3_8b_v1.json'),'languages':{}}
    for host in contract['models']['hosts']:
        language=host['language']; receipt=json.loads((root/(language+'_candidate_provenance.json')).read_text())
        assert receipt['official_published_equals_reconstructed_window'] is True
        assert receipt['selected_saved_equals_exact_mapping'] is True
        rows={}
        for name,slug,spec in [('host_baseline',language,host),('selected',language+'_selected',None),('published_reference',language+'_published',host['published_swap'])]:
            directory=root/'models'/slug
            core=[p for p in directory.iterdir() if p.is_file() and (p.name.endswith('.safetensors') or p.name in ['config.json','generation_config.json','model.safetensors.index.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','merges.txt','vocab.json','added_tokens.json','chat_template.jinja'])]
            hashes={}; verified_lfs=[]
            for path in sorted(core):
                digest=sha256(path);hashes[path.name]=digest
                if spec:
                    metadata=directory/'.cache/huggingface/download'/(path.name+'.metadata')
                    commit,etag,*_=metadata.read_text().splitlines()
                    assert commit==spec['revision'],(slug,path.name,'commit mismatch')
                    if len(etag)==64:
                        assert digest==etag,(slug,path.name,'LFS SHA mismatch')
                        verified_lfs.append(path.name)
                    else:
                        assert len(etag)==40,(slug,path.name,'unexpected etag')
                        blob=path.read_bytes()
                        assert hashlib.sha1(b'blob '+str(len(blob)).encode()+b'\0'+blob).hexdigest()==etag,(slug,path.name,'Git blob mismatch')
            index=json.loads((directory/'model.safetensors.index.json').read_text())['weight_map']
            assert set(index.values()).issubset(hashes)
            rows[name]={'model_path':str(directory),'model_id':spec['id'] if spec else 'exact_selected_layer_swap',
                        'revision':spec['revision'] if spec else None,'core_file_sha256':hashes,
                        'download_commit_and_blob_hashes_verified':True if spec else None,
                        'lfs_weight_files_verified':verified_lfs}
        report['languages'][language]={'candidates':rows,'tensor_mapping_receipt_sha256':sha256(root/(language+'_candidate_provenance.json')),
                                      'tensor_mapping_receipt_file':language+'_candidate_provenance.json',
                                      'data_manifest_sha256':sha256(root/f'results/b3/lighton_qwen3_8b/{language}_host__en_donor/data/MGSM_rev2_test.jsonl')}
        print('INPUT_RECEIPTS_VERIFIED',language,flush=True)
    destination=root/'candidate_input_receipts.json'
    destination.write_text(json.dumps(report,indent=2))
    print('ALL_INPUT_RECEIPTS_VERIFIED',flush=True)
