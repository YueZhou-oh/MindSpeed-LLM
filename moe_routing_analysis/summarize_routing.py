#!/usr/bin/env python3
"""Standalone report generator for the MindSpeed routing collector's schema-1 NPZs.

Dependencies: numpy, pandas, matplotlib. No Torch, NPU, MindSpeed, routing_core,
or routing_plots import is needed. Counts are merged before metrics are computed.
Use --baseline only with identical inputs and compatible routing settings.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

REPORT_VERSION = 'routing-report-3-single-file'


def scheduling(n, world, rank):
    if n < 1 or world < 1 or not 0 <= rank < world:
        raise ValueError('Invalid scheduling arguments')
    return [i if i < n else -1 for i in range(rank, math.ceil(n/world)*world, world)]


def metrics(counts, gate_mass=None, topk=None):
    c = np.asarray(counts, dtype=np.float64)
    if c.ndim != 1 or c.size < 2 or not np.isfinite(c).all() or (c < 0).any() or c.sum() <= 0:
        raise ValueError('Counts must be a finite, nonnegative vector with positive total')
    p = c/c.sum()
    nonzero = p > 0
    result = dict(assignments=int(round(c.sum())), load_cv=float(c.std()/c.mean()),
                  max_load_ratio=float(c.max()/c.mean()), min_load_ratio=float(c.min()/c.mean()),
                  normalized_load_entropy=float(-np.sum(p[nonzero]*np.log(p[nonzero]))/np.log(c.size)),
                  effective_experts=float(np.exp(-np.sum(p[nonzero]*np.log(p[nonzero])))),
                  unused_experts=int((c == 0).sum()),
                  underused_below_0_1x=int((c < .1*c.mean()).sum()))
    if topk:
        result['valid_tokens'] = float(c.sum()/topk)
    if gate_mass is not None:
        g = np.asarray(gate_mass, dtype=np.float64)
        if g.shape != c.shape or not np.isfinite(g).all() or (g < 0).any() or g.sum() <= 0:
            raise ValueError('Invalid gate-mass vector')
        q = g/g.sum()
        nz = q > 0
        result.update(gate_mass_cv=float(g.std()/g.mean()),
                      normalized_gate_mass_entropy=float(-np.sum(q[nz]*np.log(q[nz]))/np.log(c.size)))
    return result


def js_divergence(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.sum() <= 0 or b.sum() <= 0:
        raise ValueError('Cannot compare empty or differently shaped loads')
    a, b = a/a.sum(), b/b.sum()
    mid = (a+b)/2
    def kl(x):
        nz = x > 0
        return np.sum(x[nz]*np.log(x[nz]/mid[nz]))
    return float((kl(a)+kl(b))/2)


def load_run(root):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    if manifest.get('schema') != 1 or not manifest.get('completed'):
        raise ValueError('Unsupported or incomplete run')
    world, layers, experts = [manifest[k] for k in ('world_size','num_layers','num_experts')]
    if world != manifest['ep'] or experts % world:
        raise ValueError('Unsupported EP layout in manifest')
    n, length, k = len(manifest['samples']), manifest['sequence_length'], manifest['topk']
    indices, counts, mass, entropy, execution, owners = [], [], [], [], [], []
    for rank in range(world):
        with np.load(root/f'rank_{rank:05d}.npz', allow_pickle=False) as z:
            if int(z['schema']) != 1 or int(z['rank']) != rank:
                raise ValueError('Rank/shard schema mismatch')
            ep_rank = int(z['expert_rank']); owners.append(ep_rank)
            plan = np.asarray(scheduling(n,world,rank))
            if not np.array_equal(z['sample_indices'],plan):
                raise ValueError('Duplicate, missing or reordered sample assignments')
            for key in ('counts','execution_counts','gate_mass'):
                if z[key].shape != (len(plan),layers,experts):
                    raise ValueError(f'Invalid {key} shape in rank {rank}')
            if z['token_gate_entropy_sum'].shape != (len(plan),layers):
                raise ValueError('Invalid entropy shape')
            c,g,e,h = [z[x].copy() for x in ('counts','gate_mass','execution_counts','token_gate_entropy_sum')]
            if (not np.issubdtype(c.dtype,np.integer) or not np.issubdtype(e.dtype,np.integer)
                    or (c < 0).any() or (e < c).any() or not np.isfinite(g).all() or (g < 0).any()
                    or not np.isfinite(h).all()):
                raise ValueError('Invalid collected statistics')
            for row, sample in enumerate(plan):
                tokens = manifest['samples'][sample]['tokens'] if sample >= 0 else 0
                if not np.all(c[row].sum(-1) == tokens*k):
                    raise ValueError('Wrong token/assignment totals (possible duplication or dropping)')
                if not np.all(e[row].sum(-1) == length*k):
                    raise ValueError('Executed-token totals do not match padded input length')
                if (h[row] < -1e-5).any() or (h[row] > tokens*1.0001+1e-5).any():
                    raise ValueError('Invalid normalized per-token gate entropy')
                if tokens and (g[row].sum(-1) <= 0).any():
                    raise ValueError('Missing gate mass')
                if not tokens and ((g[row] != 0).any() or (h[row] != 0).any()):
                    raise ValueError('Dummy sample leaked into content statistics')
            valid = plan >= 0
            indices.extend(plan[valid]); counts.extend(c[valid]); mass.extend(g[valid]); entropy.extend(h[valid])
            execution.append(e)
    if sorted(owners) != list(range(world)):
        raise ValueError('Expert-rank owners are not unique')
    if sorted(indices) != list(range(n)):
        raise ValueError('Sample coverage failed')
    order = np.argsort(indices)
    return manifest, np.asarray(counts)[order], np.asarray(mass)[order], np.asarray(entropy)[order], np.stack(execution)


def scopes(manifest):
    domains = np.array([r['domain'] for r in manifest['samples']])
    yield 'overall', 'ALL', np.arange(len(domains))
    for domain in sorted(set(domains)):
        yield 'domain', domain, np.flatnonzero(domains == domain)


def summarize(root, out, baseline=None):
    m,c,g,h,execution = load_run(root)
    out = Path(out);out.mkdir(parents=True,exist_ok=True)
    layers, experts, k, ep = [m[x] for x in ('num_layers','num_experts','topk','ep')]
    nlocal = experts//ep
    layer_rows,expert_rows,device_rows,sequence_rows,batch_rows = [],[],[],[],[]
    for kind,label,idx in scopes(m):
        cc,gg = c[idx].sum(0),g[idx].astype(np.float64).sum(0)
        total_tokens = sum(m['samples'][i]['tokens'] for i in idx)
        for l in range(layers):
            row = dict(scope=kind,domain=label,layer=l,**metrics(cc[l],gg[l],k))
            row['mean_normalized_token_gate_entropy'] = h[idx,l].sum()/total_tokens
            layer_rows.append(row)
            for e in range(experts):
                expert_rows.append(dict(scope=kind,domain=label,layer=l,expert=e,
                    assignments=int(cc[l,e]),token_fraction=cc[l,e]/total_tokens,
                    assignment_share=cc[l,e]/cc[l].sum(),relative_load=cc[l,e]/cc[l].mean(),
                    gate_mass=gg[l,e],gate_mass_share=gg[l,e]/gg[l].sum()))
            device_counts = cc[l].reshape(ep,nlocal).sum(-1)
            for r in range(ep):
                device_rows.append(dict(scope=kind,domain=label,layer=l,expert_rank=r,
                    assignments=int(device_counts[r]),relative_device_load=device_counts[r]/device_counts.mean()))
    for i,sample in enumerate(m['samples']):
        for l in range(layers):
            sequence_rows.append(dict(id=sample['id'],domain=sample['domain'],layer=l,
                                      **metrics(c[i,l],g[i,l],k),
                                      mean_normalized_token_gate_entropy=h[i,l]/sample['tokens']))
    rounds = execution.shape[1]
    for step in range(rounds):
        idx=np.arange(step*ep,min((step+1)*ep,len(c)))
        # One collective EP round combines the distinct sequences on all ranks.
        for mode,cc in [('content',c[idx].sum(0)),('executed_including_padding',execution[:,step].sum(0))]:
            for l in range(layers):
                devices=cc[l].reshape(ep,nlocal).sum(-1)
                batch_rows.append(dict(round=step,mode=mode,layer=l,**metrics(cc[l],topk=k),
                                       device_load_cv=devices.std()/devices.mean(),
                                       max_device_load_ratio=devices.max()/devices.mean()))
    tables = dict(layer_metrics=layer_rows,expert_loads=expert_rows,device_loads=device_rows,
                  sequence_metrics=sequence_rows,batch_metrics=batch_rows)
    for name,rows in tables.items():
        pd.DataFrame(rows).to_csv(out/f'{name}.csv',index=False)
    comparison_rows=[]; delta_rows=[]; baseline_data=None
    if baseline:
        b,bc,bg,bh,be=load_run(baseline)
        baseline_data=(b,bc,bg,bh,be)
        for key in ('input_sha256','num_layers','num_experts','topk','ep','world_size',
                    'sequence_length','routing_type','norm_topk_prob','router_score_function',
                    'expert_bias','router_scaling_factor','samples'):
            if m[key] != b[key]:
                raise ValueError(f'Baseline differs in {key}; compare exactly matched inputs and routing settings')
        delta_rows=[]
        for kind,label,idx in scopes(m):
            aa,bb=bc[idx].sum(0),c[idx].sum(0)
            for l in range(layers):
                ma,mb=metrics(aa[l]),metrics(bb[l])
                comparison_rows.append(dict(scope=kind,domain=label,layer=l,
                    baseline_step=b['checkpoint_step'],current_step=m['checkpoint_step'],
                    baseline_cv=ma['load_cv'],current_cv=mb['load_cv'],
                    cv_delta=mb['load_cv']-ma['load_cv'],
                    baseline_entropy=ma['normalized_load_entropy'],
                    current_entropy=mb['normalized_load_entropy'],
                    load_js_divergence_nats=js_divergence(aa[l],bb[l])))
                for e in range(experts):
                    ra,rb=aa[l,e]/aa[l].mean(),bb[l,e]/bb[l].mean()
                    delta_rows.append(dict(scope=kind,domain=label,layer=l,expert=e,
                                           baseline_relative_load=ra,current_relative_load=rb,relative_load_delta=rb-ra))
        pd.DataFrame(comparison_rows).to_csv(out/'checkpoint_comparison.csv',index=False)
        pd.DataFrame(delta_rows).to_csv(out/'expert_load_changes.csv',index=False)
    plot(tables,out,comparison_rows)
    figures=extended_plots(tables,out,m,c,g,h,execution,baseline_data,comparison_rows,delta_rows)
    overview=dict(report_version=REPORT_VERSION,checkpoint_step=m['checkpoint_step'],samples=len(c),
                  valid_tokens=sum(x['tokens'] for x in m['samples']),
                  input_sha256=m['input_sha256'],layers=layers,experts=experts,
                  executed_token_slots=int(execution[:,:,0].sum()/k),
                  content_slot_fraction=float(sum(x['tokens'] for x in m['samples'])/(execution[:,:,0].sum()/k)),
                  png_figures=len(figures),
                  note='Population CV; entropy of aggregate assignment shares; gate entropy is normalized over selected top-k. '
                       'Content statistics exclude input padding and dummy samples. Executed statistics include them. '
                       'No tokenwise checkpoint route overlap or capacity-drop estimates are claimed.')
    (out/'summary.json').write_text(json.dumps(overview,indent=2))
    print(json.dumps(overview,indent=2))
    print(f"Saved {len(figures)} PNGs and metric tables to {out.resolve()}")
    return tables


def plot(tables,out,comparison):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    e=pd.DataFrame(tables['expert_loads']);e=e[e.scope=='overall']
    dev=pd.DataFrame(tables['device_loads']);dev=dev[dev.scope=='overall']
    lay=pd.DataFrame(tables['layer_metrics']);lay=lay[lay.scope=='overall']
    fig,axes=plt.subplots(1,2,figsize=(16,6),constrained_layout=True)
    for ax,frame,column,value,title in [
        (axes[0],e,'expert','relative_load','Expert load / uniform load'),
        (axes[1],dev,'expert_rank','relative_device_load','Expert-rank load / uniform load')]:
        a=frame.pivot(index='layer',columns=column,values=value)
        im=ax.imshow(a.to_numpy(),aspect='auto',interpolation='nearest',cmap='viridis',vmin=0)
        ax.set(title=title,xlabel=column,ylabel='Layer (zero-based)')
        yy=np.linspace(0,len(a.index)-1,min(len(a.index),9),dtype=int)
        xx=np.linspace(0,len(a.columns)-1,min(len(a.columns),9),dtype=int)
        ax.set_yticks(yy,labels=a.index[yy]);ax.set_xticks(xx,labels=a.columns[xx])
        fig.colorbar(im,ax=ax)
    fig.savefig(out/'load_heatmaps.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(14,4),constrained_layout=True)
    axes[0].plot(lay.layer,lay.load_cv,label='Current',marker='.')
    axes[1].plot(lay.layer,lay.normalized_load_entropy,label='Current',marker='.')
    if comparison:
        b=pd.DataFrame(comparison);b=b[b.scope=='overall']
        axes[0].plot(b.layer,b.baseline_cv,label='Baseline',marker='.')
        axes[1].plot(b.layer,b.baseline_entropy,label='Baseline',marker='.')
    for ax,title in zip(axes,['Load coefficient of variation','Normalized aggregate load entropy']):
        ax.set(title=title,xlabel='Layer (zero-based)');ax.grid(ls='--',alpha=.3);ax.legend()
    fig.savefig(out/'layer_balance.png',dpi=180);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version',action='version',version=REPORT_VERSION)
    p.add_argument('--run',required=True)
    p.add_argument('--out')
    p.add_argument('--baseline',help='Optional earlier run with identical prepared inputs/settings')
    a=p.parse_args();summarize(a.run,a.out or Path(a.run)/'summary',a.baseline)


def _matrix(frame,row,column,value):
    return frame.pivot(index=row,columns=column,values=value).sort_index().sort_index(axis=1)


def _heat(ax,a,title,xlabel,ylabel='Layer (zero-based)',**kwargs):
    values=a.to_numpy() if hasattr(a,'to_numpy') else np.asarray(a)
    im=ax.imshow(values,aspect='auto',interpolation='nearest',**kwargs)
    ax.set(title=title,xlabel=xlabel,ylabel=ylabel)
    ny,nx=values.shape
    iy=np.linspace(0,ny-1,min(ny,8),dtype=int);ix=np.linspace(0,nx-1,min(nx,9),dtype=int)
    ax.set_yticks(iy,labels=np.asarray(a.index)[iy] if hasattr(a,'index') else iy)
    ax.set_xticks(ix,labels=np.asarray(a.columns)[ix] if hasattr(a,'columns') else ix)
    return im


def extended_plots(tables,out,manifest,counts,gate_mass,entropy_sum,execution,
                   baseline=None,comparison_rows=(),delta_rows=()):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    files=[dict(file='load_heatmaps.png',description='Overall expert and device load heatmaps.'),
           dict(file='layer_balance.png',description='Layer-wise load CV and aggregate load entropy.')]
    layer=pd.DataFrame(tables['layer_metrics']);experts=pd.DataFrame(tables['expert_loads'])
    devices=pd.DataFrame(tables['device_loads']);sequences=pd.DataFrame(tables['sequence_metrics'])
    batches=pd.DataFrame(tables['batch_metrics']);overall=layer[layer.scope=='overall'].sort_values('layer')
    exp=experts[experts.scope=='overall'];dev=devices[devices.scope=='overall']
    domains=sorted(experts.loc[experts.scope=='domain','domain'].unique())
    E,K,L,W=[manifest[k] for k in ('num_experts','topk','num_layers','world_size')]
    current_label=f"Step {manifest['checkpoint_step']:,}"
    sample_info=pd.DataFrame(manifest['samples']);tokens=sample_info.tokens.to_numpy()
    per_sequence_budget=manifest['sequence_length'];rounds=execution.shape[1]
    def save(fig,name,description):
        fig.savefig(out/name,dpi=180,bbox_inches='tight');plt.close(fig)
        files.append(dict(file=name,description=description))
    def lognorm(vmax):return LogNorm(vmin=.125,vmax=max(float(vmax),1.01),clip=True)
    def map_load(ax,frame,column,value,title,vmax):
        a=_matrix(frame,'layer',column,value)
        return _heat(ax,a,title,column,norm=lognorm(vmax),cmap='viridis')
    with plt.rc_context({'font.size':10,'axes.titlesize':11,'axes.labelsize':10,
                         'axes.spines.top':False,'axes.spines.right':False}):
        # 1. Count balance is distinct from weighted routing contribution.
        fig,axs=plt.subplots(1,2,figsize=(16,6),constrained_layout=True)
        im=map_load(axs[0],exp,'expert','relative_load','Selection load / uniform',E/K)
        fig.colorbar(im,ax=axs[0],label='Relative load (log scale; values < 0.125 clipped)')
        frame=exp.assign(relative_gate_mass=exp.gate_mass_share*E)
        im=map_load(axs[1],frame,'expert','relative_gate_mass','Gate mass / uniform',E)
        fig.colorbar(im,ax=axs[1],label='Relative gate mass (log scale)')
        save(fig,'expert_gate_mass.png','Selection frequency and gate-weight mass, separately normalized.')

        # 2. Each domain keeps its own layer x expert matrix; no merging expert IDs across layers.
        ncols=2;nrows=max(1,(len(domains)+1)//2)
        fig,axs=plt.subplots(nrows,ncols,figsize=(18,4*nrows),squeeze=False,constrained_layout=True)
        for ax,domain in zip(axs.flat,domains):
            frame=experts[(experts.scope=='domain')&(experts.domain==domain)]
            im=map_load(ax,frame,'expert','relative_load',domain,E/K)
        for ax in list(axs.flat)[len(domains):]:ax.set_visible(False)
        if domains:fig.colorbar(im,ax=[a for a in axs.flat if a.get_visible()],label='Expert load / uniform (shared log scale)')
        save(fig,'domain_expert_loads.png','Domain-specific expert-load heatmaps with one shared color scale.')

        fig,axs=plt.subplots(1,3,figsize=(18,max(3,len(domains)*.5)),constrained_layout=True)
        for ax,metric,title in zip(axs,['load_cv','gate_mass_cv','mean_normalized_token_gate_entropy'],
                ['Expert-load CV','Gate-mass CV','Mean selected-gate entropy']):
            frame=layer[layer.scope=='domain'];a=_matrix(frame,'domain','layer',metric)
            im=_heat(ax,a,title,'Layer (zero-based)','Domain',cmap='viridis')
            fig.colorbar(im,ax=ax)
        save(fig,'domain_layer_metrics.png','Domain x layer load CV, gate-mass CV and selected-gate entropy.')

        # 3. Relative domain preference controls for each expert's pooled load.
        pooled=_matrix(exp,'layer','expert','relative_load').to_numpy()
        fig,axs=plt.subplots(nrows,ncols,figsize=(18,4*nrows),squeeze=False,constrained_layout=True)
        for ax,domain in zip(axs.flat,domains):
            frame=experts[(experts.scope=='domain')&(experts.domain==domain)]
            a=_matrix(frame,'layer','expert','relative_load')
            ratio=np.divide(a.to_numpy(),pooled,out=np.ones_like(pooled),where=pooled>0)
            enrichment=pd.DataFrame(np.log2(np.maximum(ratio,1e-9)),index=a.index,columns=a.columns)
            im=_heat(ax,enrichment,domain,'Expert',cmap='RdBu_r',vmin=-3,vmax=3)
        for ax in list(axs.flat)[len(domains):]:ax.set_visible(False)
        if domains:fig.colorbar(im,ax=[a for a in axs.flat if a.get_visible()],label='log2(domain load / pooled load), clipped to +/-3')
        save(fig,'domain_enrichment.png','Routing preference relative to pooled traffic. This is not a functional-specialization test.')

        # 4. Aggregate diversity and extreme loads should be read together.
        fig,axs=plt.subplots(2,2,figsize=(14,9),constrained_layout=True)
        axs[0,0].plot(overall.layer,overall.max_load_ratio,label='Maximum')
        axs[0,0].plot(overall.layer,overall.min_load_ratio,label='Minimum');axs[0,0].axhline(1,color='gray',ls='--')
        axs[0,0].set(title='Expert load extremes',ylabel='Load / uniform');axs[0,0].legend()
        axs[0,1].plot(overall.layer,overall.effective_experts);axs[0,1].axhline(E,color='gray',ls='--')
        axs[0,1].set(title='Effective expert count from aggregate entropy',ylabel='Effective experts')
        axs[1,0].plot(overall.layer,overall.unused_experts,label='Unused')
        axs[1,0].plot(overall.layer,overall.underused_below_0_1x,label='Below 0.1x mean')
        axs[1,0].set(title='Unused and underused on this input set',ylabel='Expert count');axs[1,0].legend()
        axs[1,1].plot(overall.layer,overall.normalized_load_entropy,label='Aggregate load')
        axs[1,1].plot(overall.layer,overall.mean_normalized_token_gate_entropy,label='Within selected top-k')
        axs[1,1].set(title='Two distinct entropy measurements',ylabel='Normalized entropy');axs[1,1].legend()
        for ax in axs.flat:ax.set_xlabel('Layer (zero-based)');ax.grid(ls='--',alpha=.25)
        save(fig,'routing_concentration.png','Layer-wise load extrema, diversity, underuse and entropy.')

        fig,axs=plt.subplots(1,2,figsize=(14,5),constrained_layout=True)
        positive=exp.relative_load.to_numpy();positive=positive[positive>0]
        edges=np.geomspace(max(positive.min(),.001),max(positive.max(),1.01),55)
        axs[0].hist(positive,bins=edges,color='#246a8a',alpha=.8);axs[0].set_xscale('log')
        axs[0].axvline(1,color='black',ls='--');axs[0].set(title='Distribution across layer-expert pairs',xlabel='Relative load (log scale)',ylabel='Pair count')
        sc=axs[1].scatter(overall.load_cv,overall.max_load_ratio,c=overall.layer,cmap='viridis')
        axs[1].set(title='Dispersion and busiest expert',xlabel='Load CV',ylabel='Maximum relative load');fig.colorbar(sc,ax=axs[1],label='Layer')
        save(fig,'expert_load_distribution.png','Overall distribution of relative loads and layer hotspot relationship.')

        # 5. Short sequences may have unused experts even if aggregate utilization is broad.
        fig,axs=plt.subplots(1,2,figsize=(15,5),constrained_layout=True)
        q=sequences.groupby('layer').load_cv.quantile([.05,.5,.95]).unstack()
        axs[0].fill_between(q.index,q[.05],q[.95],alpha=.2,label='5th-95th percentile')
        axs[0].plot(q.index,q[.5],label='Median');axs[0].plot(overall.layer,overall.load_cv,label='Pooled counts')
        axs[0].set(title='Per-sequence load variation',xlabel='Layer',ylabel='Load CV');axs[0].legend()
        seq=sequences.groupby(['id','domain'],as_index=False).agg(tokens=('valid_tokens','first'),cv=('load_cv','mean'))
        for domain in domains:
            x=seq[seq.domain==domain];axs[1].scatter(x.tokens,x.cv,s=22,alpha=.7,label=domain)
        axs[1].set(title='Sequence length and average load CV',xlabel='Valid tokens',ylabel='Mean CV across layers')
        axs[1].legend(fontsize=8,ncol=2)
        save(fig,'sequence_variability.png','Sequence-level dispersion and its relation to input length.')

        # 6. Round plots distinguish semantic traffic from executed padding overhead.
        fig,axs=plt.subplots(2,2,figsize=(16,9),constrained_layout=True)
        for col,(metric,title) in enumerate([('device_load_cv','Device-load CV'),('max_device_load_ratio','Maximum device load / uniform')]):
            vmax=max(float(batches[metric].max()),1e-6)
            for row,mode in enumerate(['content','executed_including_padding']):
                a=_matrix(batches[batches['mode']==mode],'round','layer',metric)
                im=_heat(axs[row,col],a,f'{title}: {mode}','Layer','Forward round',cmap='viridis',vmin=0,vmax=vmax)
                fig.colorbar(im,ax=axs[row,col])
        save(fig,'round_device_imbalance.png','Content and executed device imbalance with shared scales per metric.')

        fig,axs=plt.subplots(1,2,figsize=(15,5),constrained_layout=True)
        axs[0].hist(tokens,bins=min(30,max(5,len(tokens)//3)),color='#246a8a')
        axs[0].axvline(per_sequence_budget,color='black',ls='--',label='Execution length')
        axs[0].set(title='Actual input lengths',xlabel='Valid tokens',ylabel='Sequence count');axs[0].legend()
        valid=np.array([tokens[s*W:min((s+1)*W,len(tokens))].sum() for s in range(rounds)])
        actual=np.array([min(W,max(0,len(tokens)-s*W)) for s in range(rounds)])
        pad=actual*per_sequence_budget-valid;filler=(W-actual)*per_sequence_budget
        x=np.arange(rounds);axs[1].bar(x,valid,label='Content')
        axs[1].bar(x,pad,bottom=valid,label='Input padding')
        axs[1].bar(x,filler,bottom=valid+pad,label='Filler slots')
        utilization=100*tokens.sum()/(rounds*W*per_sequence_budget)
        axs[1].set(title=f'Executed-slot content coverage: {utilization:.2f}%',xlabel='Forward round',ylabel='Token slots')
        axs[1].legend()
        save(fig,'input_coverage.png','Input length distribution and round-wise content/padding/filler accounting.')

        # 7. Domain distances compare distributions, not expert function or factual skill.
        if domains:
            distributions={d:_matrix(experts[(experts.scope=='domain')&(experts.domain==d)],'layer','expert','assignment_share').to_numpy() for d in domains}
            distances=np.zeros((len(domains),len(domains)))
            for i,a in enumerate(domains):
                for j,b in enumerate(domains):
                    if j>i:distances[i,j]=distances[j,i]=np.mean([js_divergence(x,y) for x,y in zip(distributions[a],distributions[b])])
            fig,ax=plt.subplots(figsize=(max(6,len(domains)*.8),max(5,len(domains)*.7)),constrained_layout=True)
            im=ax.imshow(distances,cmap='magma',vmin=0)
            ax.set_xticks(range(len(domains)),labels=domains,rotation=35,ha='right');ax.set_yticks(range(len(domains)),labels=domains)
            ax.set_title('Mean per-layer domain routing divergence');fig.colorbar(im,ax=ax,label='Jensen-Shannon divergence (nats)')
            save(fig,'domain_routing_divergence.png','Mean per-layer pairwise JS divergence between domain assignment distributions.')
            pd.DataFrame(distances,index=domains,columns=domains).to_csv(out/'domain_routing_divergence.csv')

        # 8. Hotspots have explicit layer/expert labels.
        top=exp.nlargest(min(20,len(exp)),'relative_load').sort_values('relative_load')
        fig,ax=plt.subplots(figsize=(11,7),constrained_layout=True)
        labels=[f'L{int(r.layer)} / E{int(r.expert)}' for r in top.itertuples()]
        ax.barh(labels,top.relative_load,color='#246a8a');ax.axvline(1,color='black',ls='--')
        ax.set(title=f'Most-used layer-expert pairs: {current_label}',xlabel='Load / uniform')
        save(fig,'expert_hotspots.png','Top layer-expert traffic hotspots with global expert IDs within each layer.')

        if baseline is not None:
            bm,bc,bg,bh,be=baseline;baseline_label=f"Step {bm['checkpoint_step']:,}"
            ca=counts.sum(0);ba=bc.sum(0)
            current=ca/ca.mean(-1,keepdims=True);previous=ba/ba.mean(-1,keepdims=True)
            fig,axs=plt.subplots(1,2,figsize=(17,6),constrained_layout=True)
            for ax,a,label in zip(axs,[previous,current],[baseline_label,current_label]):
                im=_heat(ax,a,label,'Expert',norm=lognorm(E/K),cmap='viridis')
            fig.colorbar(im,ax=list(axs),label='Expert load / uniform (same log scale)')
            save(fig,'checkpoint_load_heatmaps.png','Checkpoint expert-load heatmaps with a fixed common color scale.')
            change=current-previous;lim=max(float(np.abs(change).max()),.01)
            fig,ax=plt.subplots(figsize=(14,6),constrained_layout=True)
            im=_heat(ax,change,f'{current_label} minus {baseline_label}','Expert',cmap='RdBu_r',vmin=-lim,vmax=lim)
            fig.colorbar(im,ax=ax,label='Change in relative load')
            save(fig,'checkpoint_expert_changes.png','Signed layer-expert load redistribution; red means increased traffic.')
            comp=pd.DataFrame(comparison_rows);frame=comp[comp.scope=='domain']
            fig,axs=plt.subplots(1,2,figsize=(17,max(4,len(domains)*.5)),constrained_layout=True)
            a=_matrix(frame,'domain','layer','cv_delta');lim=max(float(np.abs(a.to_numpy()).max()),.01)
            im=_heat(axs[0],a,'Current CV minus baseline CV','Layer','Domain',cmap='RdBu_r',vmin=-lim,vmax=lim)
            fig.colorbar(im,ax=axs[0],label='CV change: blue is more balanced')
            a=_matrix(frame,'domain','layer','load_js_divergence_nats')
            im=_heat(axs[1],a,'Routing redistribution','Layer','Domain',cmap='magma',vmin=0)
            fig.colorbar(im,ax=axs[1],label='JS divergence (nats)')
            save(fig,'checkpoint_domain_changes.png','Per-domain balance changes and aggregate routing redistribution.')
            fig,axs=plt.subplots(1,2,figsize=(15,5),constrained_layout=True)
            for mass,ent,label in [(bg,bh,baseline_label),(gate_mass,entropy_sum,current_label)]:
                mass=mass.astype(np.float64).sum(0)
                axs[0].plot(range(L),mass.std(-1)/mass.mean(-1),label=label)
                axs[1].plot(range(L),ent.sum(0)/tokens.sum(),label=label)
            axs[0].set(title='Aggregate gate-mass balance',xlabel='Layer',ylabel='Gate-mass CV')
            axs[1].set(title='Weights within selected top-k experts',xlabel='Layer',ylabel='Mean normalized token gate entropy')
            for ax in axs:ax.legend();ax.grid(ls='--',alpha=.25)
            save(fig,'checkpoint_gate_changes.png','Checkpoint changes in gate-mass balance and per-token selected-gate entropy.')
            fig,axs=plt.subplots(1,2,figsize=(16,6),constrained_layout=True)
            for ax,a,label in zip(axs,[ba,ca],[baseline_label,current_label]):
                a=a.reshape(L,W,E//W).sum(-1);a=a/a.mean(-1,keepdims=True)
                im=_heat(ax,a,label,'Expert rank',norm=lognorm(W),cmap='viridis')
            fig.colorbar(im,ax=list(axs),label='Device load / uniform (same log scale)')
            save(fig,'checkpoint_device_heatmaps.png','Device-load comparison with a shared scale; counts do not imply measured latency.')
    pd.DataFrame(files).to_csv(out/'figure_index.csv',index=False)
    (out/'figure_index.json').write_text(json.dumps(files,indent=2))
    return files


if __name__ == '__main__':
    main()
