#!/usr/bin/env python3
"""Compare native torch MindSpeed/Megatron MoE checkpoints (TP=PP=ETP=1).

Dependencies: torch numpy pandas matplotlib. Run in your training environment.
Only load checkpoints you trust: Megatron args require pickle (weights_only=False).
No NPU, distributed initialization, or full-model construction is needed.
Supported expert layouts: GroupedMLP weight1/weight2; SequentialMLP local_experts;
TE grouped linear_fc1/linear_fc2 weightN. Unknown layouts fail explicitly.
"""
import argparse
import gc
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import time
from datetime import datetime

import sys
sys.path.append('/dpc-zhouy/zhouy/MindSpeed-LLM')

def load(path):
    path = str(path)  # Required by torch_npu when mmap=True
    try:
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
            mmap=True,
        )
    except RuntimeError as e:
        if "mmap" not in str(e):
            raise
        print(
            "Legacy serialization: loading entire shard into CPU RAM:",
            path,
            flush=True,
        )
        return torch.load(
            path,
            map_location="cpu",
            weights_only=False,
        )


def discover(root, ep):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f'Not a checkpoint iteration directory: {root}')
    files = list(root.glob('mp_rank_*/model_optim_rng.pt'))
    found = {}
    for path in files:
        m = re.fullmatch(r'mp_rank_(\d+)_(\d+)', path.parent.name)
        if not m or int(m[1]) != 0:
            raise ValueError(f'Unsupported rank directory {path.parent.name}; requires TP=PP=1')
        rank = int(m[2])
        if rank in found:
            raise ValueError(f'Duplicate EP shard {rank}')
        found[rank] = path
    if set(found) != set(range(ep)):
        raise ValueError(f'Expected EP shards 0..{ep-1}; found {sorted(found)}. '
                         'Pass iter_XXXXXXX, not its parent. Distributed torch_dist is unsupported.')
    return found


def flat(d, prefix=''):
    for k, v in d.items():
        name = prefix + str(k)
        if isinstance(v, dict):
            yield from flat(v, name + '.')
        elif torch.is_tensor(v):
            yield name, v


def validate_args(ckpt, cfg):
    args = ckpt.get('args')
    def get(k):
        return args.get(k) if isinstance(args, dict) else getattr(args, k, None)
    for k, expected in [('tensor_model_parallel_size', 1), ('pipeline_model_parallel_size', 1),
                        ('expert_tensor_parallel_size', 1), ('expert_model_parallel_size', cfg.ep),
                        ('num_experts', cfg.experts), ('num_layers', cfg.layers),
                        ('hidden_size', cfg.hidden), ('moe_ffn_hidden_size', cfg.ffn)]:
        actual = get(k)
        if actual is not None and actual != expected:
            raise ValueError(f'Checkpoint {k}={actual}; requested {expected}')
    if 'model' not in ckpt or any(re.fullmatch(r'model\d+', str(k)) for k in ckpt):
        raise ValueError('Expected a single model state dictionary; virtual PP is unsupported')


def extract(ckpt, cfg):
    validate_args(ckpt, cfg)
    result = {}
    local = cfg.experts // cfg.ep
    for key, w in flat(ckpt['model']):
        if '.experts.' not in key or not w.is_floating_point():
            continue
        layer = re.search(r'(?:^|\.)layers\.(\d+)\.', key)
        if not layer:
            raise ValueError(f'Cannot identify layer: {key}')
        layer = int(layer[1])  # Megatron state-dict layer IDs are zero-based with PP=1.
        tail = key.split('.experts.', 1)[1]
        pieces = []
        if tail == 'weight1':
            if tuple(w.shape) != (cfg.hidden, local * 2 * cfg.ffn):
                raise ValueError(f'Unexpected grouped FC1 shape {key}: {tuple(w.shape)}')
            grouped = w.view(local, cfg.hidden, 2*cfg.ffn)
            pieces = [(i, 'fc1', grouped[i]) for i in range(local)]
        elif tail == 'weight2':
            if tuple(w.shape) != (local * cfg.ffn, cfg.hidden):
                raise ValueError(f'Unexpected grouped FC2 shape {key}: {tuple(w.shape)}')
            pieces = [(i, 'fc2', w[i*cfg.ffn:(i+1)*cfg.ffn, :]) for i in range(local)]
        else:
            m = re.fullmatch(r'local_experts\.(\d+)\.linear_fc([12])\.weight', tail)
            if not m:
                m = re.fullmatch(r'linear_fc([12])\.weight(\d+)', tail)
                if m:
                    pieces = [(int(m[2]), 'fc'+m[1], w)]
            else:
                pieces = [(int(m[1]), 'fc'+m[2], w)]
            if not pieces:
                raise ValueError(f'Unsupported expert tensor: {key}, shape={tuple(w.shape)}')
        for i, part, tensor in pieces:
            expected_n = cfg.hidden * cfg.ffn * (2 if part == 'fc1' else 1)
            if i >= local or tensor.numel() != expected_n:
                raise ValueError(f'Invalid expert tensor: {key}')
            ident = (layer, i, part)
            if ident in result:
                raise ValueError(f'Duplicate expert parameter: {ident}')
            result[ident] = tensor
    expected = {(l, i, p) for l in range(cfg.layers) for i in range(local) for p in ('fc1', 'fc2')}
    if set(result) != expected:
        raise ValueError(f'Expert coverage mismatch: missing={sorted(expected-set(result))[:12]}, '
                         f'extra={sorted(set(result)-expected)[:12]}. Check architecture/layout.')
    return result


def moments(a, b, chunk):
    # Exact metrics over all parameters; FP64 accumulation, FP32 subtraction.
    # A contiguous copy is made for only one expert matrix at a time.
    a, b = a.contiguous().view(-1), b.contiguous().view(-1)
    sums = np.zeros(8, dtype=np.float64)
    for start in range(0, a.numel(), chunk):
        x = a[start:start+chunk].float().double()
        y = b[start:start+chunk].float().double()
        if not torch.isfinite(x).all() or not torch.isfinite(y).all():
            raise ValueError('Non-finite expert weights detected')
        d = y-x
        sums += np.array([x.numel(), x.sum().item(), y.sum().item(),
                          (x*x).sum().item(), (y*y).sum().item(),
                          (x*y).sum().item(), (d*d).sum().item(), d.abs().sum().item()])
    return sums


def summarize(s):
    n, sx, sy, xx, yy, xy, dd, ad = s
    nx, ny = np.sqrt(xx), np.sqrt(yy)
    return dict(n=int(n), mean_a=sx/n, mean_b=sy/n,
                std_a=np.sqrt(max(0, xx/n-(sx/n)**2)),
                std_b=np.sqrt(max(0, yy/n-(sy/n)**2)),
                norm_a=nx, norm_b=ny, rmse=np.sqrt(dd/n), mean_abs_delta=ad/n,
                relative_l2=np.sqrt(dd)/nx if nx else np.nan,
                cosine=xy/(nx*ny) if nx*ny else np.nan)


def sample(tensor, size, seed):
    # Identical random coordinates at both steps, sampled with replacement.
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, tensor.numel(), size=size)
    coords = np.unravel_index(idx, tuple(tensor.shape))
    return tensor[tuple(torch.from_numpy(x) for x in coords)].float().numpy()


def plot_heatmaps(df, out):
    data = df[df.part == 'all']
    fig, axes = plt.subplots(2, 2, figsize=(20, 12), constrained_layout=True)
    for ax, metric, title in zip(axes.flat, ['relative_l2','cosine','std_b','rmse'],
                               ['Relative L2 change','Cosine similarity','Later weight std','Weight-change RMSE']):
        grid = data.pivot(index='layer', columns='expert', values=metric)
        im = ax.imshow(grid.values, aspect='auto', interpolation='nearest', cmap='viridis')
        ax.set(title=title, xlabel='Global expert ID (zero-based)', ylabel='Layer ID (zero-based)')
        fig.colorbar(im, ax=ax)
    fig.savefig(out/'expert_heatmaps.png', dpi=180)
    plt.close(fig)


def plot_histograms(samples, cfg, out):
    # One PDF page per layer, every expert; histogram/quantiles are sampled.
    with PdfPages(out/'all_expert_distributions.pdf') as pdf:
        for layer in range(cfg.layers):
            cols = 8
            rows = (cfg.experts + cols-1)//cols
            fig, axes = plt.subplots(rows, cols, figsize=(24, rows*1.6), squeeze=False)
            for expert, ax in enumerate(axes.flat):
                if expert >= cfg.experts:
                    ax.set_visible(False)
                    continue
                a, b = samples[(layer, expert)]
                lo, hi = min(a.min(),b.min()), max(a.max(),b.max())
                if lo == hi:
                    lo, hi = lo-1e-6, hi+1e-6
                edges = np.linspace(lo, hi, 51)
                ax.hist(a, bins=edges, density=True, histtype='step', color='#2471a3', label='Earlier')
                ax.hist(b, bins=edges, density=True, histtype='step', color='#d35400', label='Later')
                ax.set_title(f'Expert {expert}', fontsize=9)
                ax.tick_params(labelsize=6)
                if expert == 0:
                    ax.legend(fontsize=7)
            fig.suptitle(f'Layer {layer}: sampled expert weight distributions (FC1 + FC2)', fontsize=16)
            fig.tight_layout(rect=(0,0,1,.985))
            pdf.savefig(fig)
            plt.close(fig)


def compare(cfg):
    if cfg.experts % cfg.ep:
        raise ValueError('experts must be divisible by EP')
    if cfg.samples < 3 or cfg.chunk < 1:
        raise ValueError('samples >= 3 and chunk >= 1 required')
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    print(cfg.a)
    print(cfg.b)
    pa, pb = discover(cfg.a, cfg.ep), discover(cfg.b, cfg.ep)
    rows, sampled, quantiles = [], {}, []
    steps = []
    for rank in range(cfg.ep):
        print(f'Comparing EP shard {rank+1}/{cfg.ep}', flush=True)
        ca, cb = load(pa[rank]), load(pb[rank])
        step = (ca.get('iteration'), cb.get('iteration'))
        if steps and step != steps[0]:
            raise ValueError('Inconsistent checkpoint iterations across shards')
        steps.append(step)
        ea, eb = extract(ca, cfg), extract(cb, cfg)
        for layer in range(cfg.layers):
            for i in range(cfg.experts // cfg.ep):
                expert = rank*(cfg.experts//cfg.ep)+i
                total = np.zeros(8)
                sa, sb = [], []
                for part in ('fc1','fc2'):
                    a, b = ea[(layer,i,part)], eb[(layer,i,part)]
                    if a.shape != b.shape:
                        raise ValueError(f'Shape changed at {(layer,expert,part)}')
                    s = moments(a,b,cfg.chunk)
                    total += s
                    rows.append(dict(layer=layer,expert=expert,part=part,**summarize(s)))
                    # FC1 has twice as many parameters, so use 2:1 sample counts.
                    count = 2*cfg.samples//3 if part == 'fc1' else cfg.samples-2*cfg.samples//3
                    seed = cfg.seed + (layer*cfg.experts+expert)*2 + (part=='fc2')
                    sa.append(sample(a,count,seed)); sb.append(sample(b,count,seed))
                rows.append(dict(layer=layer,expert=expert,part='all',**summarize(total)))
                sa, sb = np.concatenate(sa), np.concatenate(sb)
                sampled[(layer,expert)] = (sa,sb)
                for label, values in [('a',sa),('b',sb)]:
                    q = np.quantile(values,[.01,.05,.5,.95,.99])
                    quantiles.append(dict(layer=layer,expert=expert,checkpoint=label,
                                          **dict(zip(['q01','q05','q50','q95','q99'],q))))
        del ea, eb, ca, cb
        gc.collect()
    df = pd.DataFrame(rows).sort_values(['layer','expert','part'])
    df.to_csv(out/'expert_metrics.csv',index=False)
    pd.DataFrame(quantiles).to_csv(out/'sampled_quantiles.csv',index=False)
    plot_heatmaps(df,out)
    if cfg.hist_pages:
        plot_histograms(sampled,cfg,out)
    metadata = dict(vars(cfg), iterations=steps[0], exact_metrics=True,
                    histogram_and_quantiles='random coordinate samples; FC1:FC2 approximately 2:1',
                    script_source='YueZhou-oh/MindSpeed-LLM/scillm_30b_a3b_pretrain_stage1.sh',
                    scope='native torch checkpoints; identical TP=PP=ETP=1, EP and expert indexing')
    (out/'comparison_metadata.json').write_text(json.dumps(metadata,indent=2))
    print(df[df.part=='all'].nlargest(10,'relative_l2').to_string(index=False))
    print('Results:',out.resolve())
    return df


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--a',required=True,help='Earlier iter_XXXXXXX directory')
    p.add_argument('--b',required=True,help='Later iter_XXXXXXX directory')
    p.add_argument('--out',default=f'expert_comparison_{time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())}')
    p.add_argument('--ep',type=int,default=16)
    p.add_argument('--experts',type=int,default=128)
    p.add_argument('--layers',type=int,default=48)
    p.add_argument('--hidden',type=int,default=2048)
    p.add_argument('--ffn',type=int,default=768)
    p.add_argument('--samples',type=int,default=4096,help='Sampled coordinates per expert')
    p.add_argument('--chunk',type=int,default=262144)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--hist-pages',action='store_true',help='Also write all-expert histogram PDF (48 pages)')
    return p


if __name__ == '__main__':
    compare(parser().parse_args())

'''
python compare_moe_experts.py --a /dpc-zhouy/zhouy/ckpts/SciLLM-base-30B-A3B/iter_0002000 --b /dpc-zhouy/zhouy/ckpts/SciLLM-base-30B-A3B/iter_0062000 --hist-pages >>compare_moe_weights.log 2>&1 < /dev/null &
'''
