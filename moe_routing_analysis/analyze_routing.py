#!/usr/bin/env python3
"""Native MindSpeed/Megatron fixed-input routing analysis; launch with torchrun.

Uses the repository's model_provider/get_model/load_checkpoint APIs. Requires
TP=PP=CP=ETP=1, B=1, WORLD_SIZE=EP, full expert-parallel group, torch checkpoints.
"""
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np

# Must execute before imports of Megatron or any feature/spec modules.
REPO = Path(os.environ.get('MINDSPEED_REPO', '.')).resolve()
if not (REPO/'pretrain_gpt.py').is_file():
    raise RuntimeError('Set MINDSPEED_REPO to your MindSpeed-LLM checkout')
sys.path.insert(0, str(REPO))
from mindspeed_llm import megatron_adaptor  # noqa: E402, F401
import torch  # noqa: E402
from megatron.training import get_args, get_model  # noqa: E402
from megatron.training.initialize import initialize_megatron  # noqa: E402
from megatron.training.checkpointing import load_checkpoint  # noqa: E402
from megatron.core import parallel_state  # noqa: E402
from routing_core import RouterCollector, canonical_hash, read_inputs, scheduling, save_rank  # noqa: E402

REVIEWED_COMMIT = 'f2d5e348862885885a909bb5a50995b88c90d5a5'


def extra_args(parser):
    group = parser.add_argument_group('fixed-input routing analysis')
    group.add_argument('--route-input', required=True)
    group.add_argument('--route-output', required=True)
    return parser


def verify(args):
    for key in ['tensor_model_parallel_size', 'pipeline_model_parallel_size',
                'context_parallel_size', 'expert_tensor_parallel_size', 'micro_batch_size']:
        if getattr(args, key, 1) != 1:
            raise ValueError(f'{key} must be 1 for this verified collector layout')
    if torch.distributed.get_world_size() != args.expert_model_parallel_size:
        raise ValueError('WORLD_SIZE must equal EP (one complete expert-parallel group)')
    if not args.load or getattr(args, 'ckpt_step', None) is None:
        raise ValueError('--load checkpoint-root and --ckpt-step are both required')
    if args.ckpt_format != 'torch' or not args.no_load_optim or not args.no_load_rng:
        raise ValueError('Use --ckpt-format torch --no-load-optim --no-load-rng')
    if args.micro_batch_size != 1 or args.num_experts % args.expert_model_parallel_size:
        raise ValueError('Requires B=1 and experts divisible by EP')
    for key in ['moe_pad_expert_input_to_capacity', 'moe_fb_overlap', 'moe_tp_extend_ep',
                'fix_router', 'enable_hf2mg_convert', 'use_checkpoint_args']:
        if getattr(args, key, False):
            raise ValueError(f'{key} is unsupported for this analysis path')
    if getattr(args, 'moe_expert_capacity_factor', None) is not None:
        raise ValueError('Capacity-limited routing is unsupported; requested and accepted loads would differ')
    if getattr(args, 'mtp_num_layers', None):
        raise ValueError('MTP routers require a separate collection layout')
    if args.num_experts is None or args.moe_router_topk <= 1:
        raise ValueError('Requires an MoE with top-k > 1')
    if getattr(args, 'virtual_pipeline_model_parallel_size', None):
        raise ValueError('Virtual pipeline parallelism is unsupported')
    if not 1 <= args.seq_length <= args.max_position_embeddings:
        raise ValueError('Invalid evaluation sequence length')
    directory = Path(args.load)/f'iter_{args.ckpt_step:07d}'
    expected = {f'mp_rank_00_{r:03d}' for r in range(args.expert_model_parallel_size)}
    actual = {p.name for p in directory.glob('mp_rank_*') if (p/'model_optim_rng.pt').is_file()}
    if actual != expected:
        raise ValueError(f'Checkpoint shards do not match EP={len(expected)}: {directory}')


def distributed_hash_check(digest, device):
    raw = list(bytes.fromhex(digest))
    low = torch.tensor(raw, dtype=torch.int32, device=device)
    high = low.clone()
    torch.distributed.all_reduce(low, op=torch.distributed.ReduceOp.MIN)
    torch.distributed.all_reduce(high, op=torch.distributed.ReduceOp.MAX)
    if not torch.equal(low, high):
        raise ValueError('Ranks read different prepared input data')


def main():
    initialize_megatron(extra_args_provider=extra_args,
                        args_defaults={'no_load_rng': True, 'no_load_optim': True})
    args = get_args()
    verify(args)
    rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
    expert_rank = parallel_state.get_expert_model_parallel_rank()
    # MindSpeed's adaptor redirects torch.cuda device operations to Ascend.
    device = torch.device('npu', torch.npu.current_device())
    records = read_inputs(args.route_input, args.seq_length, args.padded_vocab_size)
    digest = canonical_hash(records)
    distributed_hash_check(digest, device)
    out = Path(args.route_output).resolve()
    if rank == 0:
        if out.exists() and any(out.iterdir()):
            raise ValueError(f'Output directory must be new or empty: {out}')
        out.mkdir(parents=True, exist_ok=True)
        print(f'Loading checkpoint step {args.ckpt_step}; {len(records)} distinct inputs; {world} ranks', flush=True)
    torch.distributed.barrier()
    if not out.is_dir():
        raise ValueError('Output path must be shared and visible on every node')

    # Import only after initialize_megatron: qwen3_spec reads global args.
    from pretrain_gpt import model_provider
    models = get_model(model_provider, wrap_with_ddp=False)
    if len(models) != 1:
        raise ValueError('Expected one local model chunk')
    loaded = load_checkpoint(models, None, None, 'load')
    loaded_step = int(loaded[0] if isinstance(loaded, tuple) else loaded)
    if loaded_step != args.ckpt_step:
        raise ValueError(f'Requested step {args.ckpt_step}, loader returned {loaded_step}')
    model = models[0]
    raw = model
    while hasattr(raw, 'module'):
        raw = raw.module
    if not hasattr(raw, 'decoder') or not hasattr(raw, 'post_process'):
        raise ValueError('Expected Megatron Core GPTModel')
    model.eval()
    model.requires_grad_(False)
    # Decoder and all expert layers still run. Skip unnecessary vocabulary logits.
    raw.post_process = False
    collector = RouterCollector(model, args.num_layers, args.num_experts, args.moe_router_topk, device)
    plan = scheduling(len(records), world, rank)
    batches = []
    length = args.seq_length
    positions = torch.arange(length, device=device).unsqueeze(0)
    causal = torch.triu(torch.ones((length, length), dtype=torch.bool, device=device), diagonal=1)[None,None]
    try:
        with torch.no_grad():
            for step, index in enumerate(plan):
                ids = records[index]['input_ids'] if index >= 0 else []
                # Real tokens are a prefix; padding is a suffix.
                # Pad ID 0 is a real vocabulary ID; its rows are excluded by valid_mask.
                tokens = torch.zeros((1,length), dtype=torch.long, device=device)
                if ids:
                    tokens[0,:len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
                valid = torch.arange(length, device=device) < len(ids)
                collector.begin(valid)
                output = model(tokens, positions, causal)
                del output
                batches.append(collector.finish())
                if rank == 0:
                    print(f'Forward round {step+1}/{len(plan)} completed', flush=True)
        save_rank(out/f'rank_{rank:05d}.npz', rank, expert_rank, plan, batches,
                  args.num_layers, args.num_experts)
    finally:
        collector.close()
    torch.distributed.barrier()
    if rank == 0:
        try:
            commit = subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'], text=True).strip()
        except (OSError,subprocess.CalledProcessError):
            commit = 'unknown'
        manifest = dict(schema=1, completed=True, input_sha256=digest,
                        checkpoint=str(Path(args.load).resolve()), checkpoint_step=loaded_step,
                        repo_commit=commit, reviewed_commit=REVIEWED_COMMIT,
                        num_layers=args.num_layers, num_experts=args.num_experts,
                        topk=args.moe_router_topk, ep=args.expert_model_parallel_size,
                        world_size=world, sequence_length=length,
                        router_names=collector.router_names,
                        routing_type=args.moe_router_load_balancing_type,
                        norm_topk_prob=bool(args.norm_topk_prob),
                        router_score_function=getattr(args,'moe_router_score_function','softmax'),
                        expert_bias=bool(getattr(args,'moe_router_enable_expert_bias',False)),
                        router_scaling_factor=getattr(args,'moe_router_topk_scaling_factor',None),
                        semantics='eval; no_grad; fixed teacher-forced inputs; input padding excluded; no generation',
                        samples=[{'id':r['id'],'domain':r['domain'],'tokens':len(r['input_ids'])} for r in records])
        (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        print(f'Completed. Summarize with: python summarize_routing.py --run {out}',flush=True)
    torch.distributed.barrier()
    torch.distributed.destroy_process_group()


if __name__ == '__main__':
    main()
