"""Prepare exact whole-layer swaps on CPU and audit official reference tensors."""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import shutil
import gc

ROOT = Path('/workspace/b3_generation')

def digest_tensor(tensor):
    data=tensor.detach().cpu().contiguous()
    h=hashlib.sha256()
    h.update(str(data.dtype).encode()); h.update(str(tuple(data.shape)).encode())
    h.update(memoryview(data.view(__import__('torch').uint8).numpy()))
    return h.hexdigest()

def checkpoint_hashes(directory):
    from safetensors import safe_open
    index=json.loads((directory/'model.safetensors.index.json').read_text())['weight_map']
    out={}
    for shard in sorted(set(index.values())):
        with safe_open(str(directory/shard),framework='pt',device='cpu') as handle:
            for name in handle.keys():
                assert name not in out
                out[name]=digest_tensor(handle.get_tensor(name))
    assert set(out)==set(index)
    return out

def layer_of(name):
    parts=name.split('.')
    return int(parts[2]) if len(parts)>3 and parts[:2]==['model','layers'] else None

def aggregate(hashes):
    return hashlib.sha256(json.dumps(hashes,sort_keys=True,separators=(',',':')).encode()).hexdigest()

if __name__=='__main__':
    import torch
    from transformers import AutoModelForCausalLM
    parser=argparse.ArgumentParser();parser.add_argument('--language',required=True,choices=['fr','zh'])
    language=parser.parse_args().language
    contract=json.loads((ROOT/'configs/b3_lighton_qwen3_8b_v1.json').read_text())
    host_spec=next(x for x in contract['models']['hosts'] if x['language']==language)
    selected=list(range(13,16)) if language=='fr' else list(range(20,32))
    published=list(range(host_spec['published_swap']['window_inclusive'][0],host_spec['published_swap']['window_inclusive'][1]+1))
    source=ROOT/'models'/language; donor_dir=ROOT/'models/en'
    destination=ROOT/'models'/(language+'_selected')
    report_path=ROOT/(language+'_candidate_provenance.json')
    if destination.exists():
        raise RuntimeError('Selected destination already exists; inspect prior provenance before resuming')
    print('HASH_SOURCE_CHECKPOINTS',language,flush=True)
    host_hash=checkpoint_hashes(source); donor_hash=checkpoint_hashes(donor_dir)
    official_hash=checkpoint_hashes(ROOT/'models'/(language+'_published'))
    assert set(host_hash)==set(donor_hash)==set(official_hash)
    expected_pub={name:(donor_hash[name] if layer_of(name) in published else digest) for name,digest in host_hash.items()}
    mismatches=[name for name in host_hash if official_hash[name]!=expected_pub[name]]
    report={'language':language,'created_at_utc':datetime.now(timezone.utc).isoformat(),
            'selected_window_zero_based_inclusive':[selected[0],selected[-1]],
            'published_window_zero_based_inclusive':[published[0],published[-1]],
            'host':host_spec,'donor':contract['models']['donor'],
            'source_aggregate_tensor_sha256':{'host':aggregate(host_hash),'donor':aggregate(donor_hash),'official_published':aggregate(official_hash)},
            'official_published_equals_reconstructed_window':not mismatches,
            'official_published_mismatched_tensor_names':mismatches,
            'tensor_hashes':{'host':host_hash,'donor':donor_hash,'official_published':official_hash}}
    report_path.write_text(json.dumps(report,indent=2))
    print('OFFICIAL_PUBLISHED_PARITY',language,len(mismatches),'mismatches',flush=True)
    if mismatches:
        raise RuntimeError('Official published checkpoint differs from declared layer mapping; inspect before generation')
    print('LOAD_CPU_MODELS',flush=True)
    donor=AutoModelForCausalLM.from_pretrained(str(donor_dir),dtype=torch.bfloat16,local_files_only=True,trust_remote_code=False)
    host=AutoModelForCausalLM.from_pretrained(str(source),dtype=torch.bfloat16,local_files_only=True,trust_remote_code=False)
    assert {name:digest_tensor(tensor) for name,tensor in donor.state_dict().items()}==donor_hash
    assert {name:digest_tensor(tensor) for name,tensor in host.state_dict().items()}==host_hash
    for layer in selected:
        host.model.layers[layer].load_state_dict(donor.model.layers[layer].state_dict(),strict=True)
    expected_selected={name:(donor_hash[name] if layer_of(name) in selected else digest) for name,digest in host_hash.items()}
    assert {name:digest_tensor(tensor) for name,tensor in host.state_dict().items()}==expected_selected
    host.save_pretrained(str(destination),safe_serialization=True,max_shard_size='5GB')
    for filename in ['tokenizer.json','tokenizer_config.json','special_tokens_map.json','merges.txt','vocab.json','added_tokens.json','chat_template.jinja']:
        if (source/filename).is_file():shutil.copyfile(source/filename,destination/filename)
    del donor,host;gc.collect()
    saved_hashes=checkpoint_hashes(destination)
    assert saved_hashes==expected_selected
    report['selected_saved_equals_exact_mapping']=True
    report['source_aggregate_tensor_sha256']['selected']=aggregate(saved_hashes)
    report['tensor_hashes']['selected']=saved_hashes
    report['finished_at_utc']=datetime.now(timezone.utc).isoformat()
    report_path.write_text(json.dumps(report,indent=2))
    selected_provenance={'language':language,'graft_layers':selected,
                         'host_revision':host_spec['revision'],
                         'donor_revision':contract['models']['donor']['revision'],
                         'exact_mapping_verified':True,
                         'aggregate_tensor_sha256':aggregate(saved_hashes),
                         'host_model_id':host_spec['id'],
                         'donor_model_id':contract['models']['donor']['id']}
    (destination/'suture_graft_provenance.json').write_text(json.dumps(selected_provenance,indent=2))
    print('PREPARE_COMPLETE',language,flush=True)
