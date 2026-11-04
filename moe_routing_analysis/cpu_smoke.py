#!/usr/bin/env python3
"""Two-process CPU smoke test of hooks, input partitioning, and reports.

Not a substitute for a native Ascend/HCCL model run. Example:
python cpu_smoke.py --out /tmp/routing_smoke
Optional Gloo: torchrun --standalone --nproc_per_node 2 cpu_smoke.py --gloo --out /tmp/gloo_smoke
"""
import argparse
import json
from pathlib import Path
import torch
import torch.distributed as dist
from routing_core import RouterCollector,canonical_hash,scheduling,save_rank


class ToyRouter(torch.nn.Module):
    def __init__(self,experts=8,topk=2,layer=0):
        super().__init__();self.experts=experts;self.topk=topk;self.layer=layer
    def forward(self,x):
        x=x.reshape(-1,self.experts).roll(self.layer, dims=-1)
        bias=torch.arange(self.experts,device=x.device).float()/100
        logits=x*3+bias
        values,idx=logits.topk(self.topk,dim=-1)
        mass=values.softmax(-1)
        scores=torch.zeros_like(logits).scatter(1,idx,mass)
        route=torch.zeros_like(logits,dtype=torch.bool).scatter(1,idx,True)
        return scores,route


class ToyModel(torch.nn.Module):
    def __init__(self,layers=3,experts=8,topk=2):
        super().__init__();self.experts=experts
        self.decoder=torch.nn.Module();self.decoder.layers=torch.nn.ModuleList()
        for l in range(layers):
            layer=torch.nn.Module();layer.mlp=torch.nn.Module();layer.mlp.router=ToyRouter(experts,topk,l)
            self.decoder.layers.append(layer)
    def forward(self,tokens):
        x=torch.nn.functional.one_hot(tokens[0]%self.experts,num_classes=self.experts).float()[:,None,:]
        for layer in self.decoder.layers:
            layer.mlp.router(x)
        return x


def write_worker(out,rank,world):
    records=[dict(id=f'sample{i}',domain=['science','general'][i%2],input_ids=list(range(i+2))) for i in range(5)]
    model=ToyModel().eval();collector=RouterCollector(model,3,8,2,torch.device('cpu'))
    plan=scheduling(len(records),world,rank);batches=[];length=12
    with torch.no_grad():
        for idx in plan:
            ids=records[idx]['input_ids'] if idx>=0 else []
            tokens=torch.zeros(1,length,dtype=torch.long)
            if ids:tokens[0,:len(ids)]=torch.tensor(ids)
            collector.begin(torch.arange(length)<len(ids));model(tokens);batches.append(collector.finish())
    collector.close();save_rank(Path(out)/f'rank_{rank:05d}.npz',rank,rank,plan,batches,3,8)
    return records


def write_manifest(out,records,world):
    manifest=dict(schema=1,completed=True,input_sha256=canonical_hash(records),checkpoint='toy',checkpoint_step=1,
                  num_layers=3,num_experts=8,topk=2,ep=world,world_size=world,sequence_length=12,
                  routing_type='toy',norm_topk_prob=True,
                  router_score_function='softmax',expert_bias=False,router_scaling_factor=None,
                  samples=[dict(id=r['id'],domain=r['domain'],tokens=len(r['input_ids'])) for r in records])
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    from summarize_routing import summarize
    summarize(out,out/'summary')


def smoke(out,use_gloo=False):
    out=Path(out)
    if use_gloo:
        dist.init_process_group('gloo')
        rank,world=dist.get_rank(),dist.get_world_size()
        if world!=2:raise ValueError('Run the Gloo smoke test with two processes')
        if rank==0:
            if out.exists() and any(out.iterdir()):raise ValueError('Smoke output must be new/empty')
            out.mkdir(parents=True,exist_ok=True)
        dist.barrier();records=write_worker(out,rank,world);dist.barrier()
        if rank==0:write_manifest(out,records,world)
        dist.barrier();dist.destroy_process_group()
    else:
        import subprocess,sys
        if out.exists() and any(out.iterdir()):raise ValueError('Smoke output must be new/empty')
        out.mkdir(parents=True,exist_ok=True)
        jobs=[subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--out',str(out),
                               '--worker-rank',str(r)]) for r in range(2)]
        codes=[job.wait() for job in jobs]
        if any(codes):raise RuntimeError(f'CPU workers failed: {codes}')
        records=[dict(id=f'sample{i}',domain=['science','general'][i%2],input_ids=list(range(i+2))) for i in range(5)]
        write_manifest(out,records,2)
    print('PASS: two-process CPU collection and merged reports',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True)
    p.add_argument('--gloo',action='store_true',help='Optional: launch with torchrun for a real Gloo barrier test')
    p.add_argument('--worker-rank',type=int,help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.worker_rank is not None:write_worker(a.out,a.worker_rank,2)
    else:smoke(a.out,a.gloo)
