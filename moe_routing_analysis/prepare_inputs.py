#!/usr/bin/env python3
"""Freeze JSONL inputs into token IDs for checkpoint-matched routing analysis.

Input records: {id,domain,text}, {id,domain,messages}, or {id,domain,input_ids}.
Use fixed text/reference answers, not checkpoint-dependent generated responses.
"""
import argparse
import json
import random
import hashlib
from pathlib import Path
from routing_core import canonical_hash,read_inputs


def prepare(source,destination,max_length,limit=None,per_domain=None,seed=42,tokenizer=None):
    if max_length < 1 or (limit is not None and limit < 1) or (per_domain is not None and per_domain < 1):
        raise ValueError('Lengths and sample limits must be positive')
    source,destination=Path(source),Path(destination)
    if source.resolve()==destination.resolve():
        raise ValueError('Input and output must differ')
    if destination.exists() or destination.with_suffix(destination.suffix+'.meta.json').exists():
        raise ValueError('Prepared output already exists; choose a new path')
    records=[];seen=set();truncated=0
    for line_no,line in enumerate(source.read_text(encoding='utf-8').splitlines(),1):
        if not line.strip():continue
        x=json.loads(line)
        ident=x.get('id',f'record_{line_no:06d}');domain=x.get('domain')
        if not isinstance(ident,str) or not ident or ident in seen:
            raise ValueError(f'Line {line_no}: invalid or duplicate id')
        if not isinstance(domain,str) or not domain:
            raise ValueError(f'Line {line_no}: domain is required')
        seen.add(ident)
        kinds=[key for key in ('input_ids','text','messages') if key in x]
        if len(kinds)!=1:raise ValueError(f'Line {line_no}: provide exactly one of text/messages/input_ids')
        if kinds[0]=='input_ids':
            ids=x['input_ids']
        elif tokenizer is None:
            raise ValueError('--tokenizer is required for text or messages')
        elif kinds[0]=='text':
            if not isinstance(x['text'],str):raise ValueError('text must be a string')
            ids=tokenizer.encode(x['text'],add_special_tokens=False)
        else:
            ids=tokenizer.apply_chat_template(x['messages'],tokenize=True,add_generation_prompt=False)
        if not isinstance(ids,list) or not ids or any(type(i) is not int or i<0 for i in ids):
            raise ValueError(f'Line {line_no}: invalid or empty token sequence')
        truncated+=len(ids)>max_length
        records.append(dict(id=ident,domain=domain,input_ids=ids[:max_length]))
    rng=random.Random(seed)
    rng.shuffle(records)
    if per_domain is not None:
        keep=[];sizes={}
        for x in records:
            domain=x['domain'];sizes.setdefault(domain,0)
            if sizes[domain]<per_domain:keep.append(x);sizes[domain]+=1
        short={k:v for k,v in sizes.items() if v<per_domain}
        if short:raise ValueError(f'Insufficient records for --per-domain {per_domain}: {short}')
        records=keep
    if limit is not None:records=records[:limit]
    if not records:raise ValueError('No inputs selected')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with destination.open('w',encoding='utf-8') as f:
        for x in records:f.write(json.dumps(x,ensure_ascii=False)+'\n')
    read_inputs(destination,max_length)
    stats={}
    for x in records:
        d=stats.setdefault(x['domain'],dict(samples=0,tokens=0));d['samples']+=1;d['tokens']+=len(x['input_ids'])
    metadata=dict(schema=1,input_sha256=canonical_hash(records),max_length=max_length,seed=seed,
                  source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                  domains=stats,source_records_truncated=truncated,
                  tokenizer=getattr(tokenizer,'name_or_path','pretokenized IDs'),
                  note='Each record is one independent sequence; tokens are truncated from the right; no document packing.')
    destination.with_suffix(destination.suffix+'.meta.json').write_text(json.dumps(metadata,indent=2))
    print(json.dumps(metadata,indent=2));return records


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',default="./four_domain_620.jsonl")
    p.add_argument('--output',default="./four_domain_620_tokenized.jsonl")
    p.add_argument('--tokenizer',default="/dpc-zhouy/zhouy/ckpts/Qwen3-30B-A3B")
    p.add_argument('--max-length',type=int,default=50)
    p.add_argument('--limit',type=int)
    p.add_argument('--per-domain',type=int)
    p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();tokenizer=None
    if a.tokenizer:
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(a.tokenizer,local_files_only=True,trust_remote_code=False)
    prepare(a.input,a.output,a.max_length,a.limit,a.per_domain,a.seed,tokenizer)


if __name__=='__main__':main()
