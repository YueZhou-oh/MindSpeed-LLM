#!/usr/bin/env python3
"""Numerical, padding, scheduling, integrity and source-contract tests (CPU).

python test_routing.py
python test_routing.py --repo /path/to/MindSpeed-LLM
The optional repo check validates reviewed API signatures, not an NPU forward.
"""
import ast
import contextlib
import io
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
import torch
from routing_core import RouterCollector,canonical_hash,metrics,js_divergence,read_inputs,scheduling,save_rank
from prepare_inputs import prepare
from cpu_smoke import ToyModel
from summarize_routing import load_run,summarize

REPO=None
if '--repo' in sys.argv:
    i=sys.argv.index('--repo');REPO=Path(sys.argv[i+1]);del sys.argv[i:i+2]


def synthetic_run(root,step=1):
    records=[dict(id=f's{i}',domain=['science','general'][i%2],input_ids=list(range(i+2))) for i in range(5)]
    root.mkdir();world,layers,experts,k,length=2,3,8,2,12
    for rank in range(world):
        model=ToyModel().eval();collector=RouterCollector(model,layers,experts,k,torch.device('cpu'))
        plan=scheduling(len(records),world,rank);batches=[]
        with torch.no_grad():
            for idx in plan:
                ids=records[idx]['input_ids'] if idx>=0 else []
                tokens=torch.zeros(1,length,dtype=torch.long)
                if ids:tokens[0,:len(ids)]=torch.tensor(ids)
                collector.begin(torch.arange(length)<len(ids));model(tokens);batches.append(collector.finish())
        collector.close();save_rank(root/f'rank_{rank:05d}.npz',rank,rank,plan,batches,layers,experts)
    m=dict(schema=1,completed=True,input_sha256=canonical_hash(records),checkpoint='toy',checkpoint_step=step,
           num_layers=layers,num_experts=experts,topk=k,ep=world,world_size=world,sequence_length=length,
           routing_type='toy',norm_topk_prob=True,router_score_function='softmax',
           expert_bias=False,router_scaling_factor=None,
           samples=[dict(id=r['id'],domain=r['domain'],tokens=len(r['input_ids'])) for r in records])
    (root/'manifest.json').write_text(json.dumps(m));return m


class MetricTests(unittest.TestCase):
    def test_uniform(self):
        m=metrics(np.ones(8)*10,np.ones(8),2)
        self.assertEqual(m['load_cv'],0);self.assertAlmostEqual(m['normalized_load_entropy'],1)
        self.assertAlmostEqual(m['effective_experts'],8);self.assertEqual(m['max_load_ratio'],1)
    def test_topk_concentration(self):
        m=metrics([10,10,0,0,0,0,0,0])
        self.assertAlmostEqual(m['load_cv'],math.sqrt(3));self.assertEqual(m['unused_experts'],6)
        self.assertAlmostEqual(m['normalized_load_entropy'],math.log(2)/math.log(8))
    def test_gate_mass_distinct(self):
        m=metrics([10]*8,[100,1,1,1,1,1,1,1])
        self.assertEqual(m['load_cv'],0);self.assertGreater(m['gate_mass_cv'],1)
    def test_empty_rejected(self):
        with self.assertRaises(ValueError):metrics([0]*8)
    def test_negative_rejected(self):
        with self.assertRaises(ValueError):metrics([1,-1,3])
    def test_js(self):
        self.assertEqual(js_divergence([1,2],[1,2]),0)
        self.assertAlmostEqual(js_divergence([1,0],[0,1]),math.log(2))
    def test_population_cv(self):
        self.assertAlmostEqual(metrics([1,3])['load_cv'],.5)


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.model=ToyModel().eval();self.c=RouterCollector(self.model,3,8,2,torch.device('cpu'))
    def tearDown(self):self.c.close()
    def run_batch(self,n=3,length=6):
        self.c.begin(torch.arange(length)<n)
        with torch.no_grad():self.model(torch.arange(length)[None,:])
        return self.c.finish()
    def test_padding(self):
        r=self.run_batch();np.testing.assert_equal(r['counts'].sum(-1),[6]*3)
        np.testing.assert_equal(r['execution_counts'].sum(-1),[12]*3)
    def test_dummy(self):
        r=self.run_batch(0);self.assertEqual(r['counts'].sum(),0);self.assertEqual(r['gate_mass'].sum(),0)
    def test_gate_mass_total(self):
        r=self.run_batch();np.testing.assert_allclose(r['gate_mass'].sum(-1),3,rtol=1e-6)
    def test_entropy_bounds(self):
        r=self.run_batch();self.assertTrue(np.all((r['token_gate_entropy_sum']>=0)&(r['token_gate_entropy_sum']<=3)))
    def test_duplicate_forward_rejected(self):
        self.c.begin(torch.ones(6,dtype=torch.bool));self.model(torch.arange(6)[None,:])
        with self.assertRaises(RuntimeError):self.model(torch.arange(6)[None,:])
    def test_missing_forward_rejected(self):
        self.c.begin(torch.ones(6,dtype=torch.bool))
        with self.assertRaises(RuntimeError):self.c.finish()
    def test_shape_rejected(self):
        self.c.begin(torch.ones(5,dtype=torch.bool))
        with self.assertRaises(ValueError):self.model(torch.arange(6)[None,:])
    def test_hook_does_not_change_output(self):
        self.c.close();tokens=torch.arange(6)[None,:];before=self.model(tokens)
        self.c=RouterCollector(self.model,3,8,2,torch.device('cpu'));self.c.begin(torch.ones(6,dtype=torch.bool))
        after=self.model(tokens);self.c.finish();torch.testing.assert_close(before,after)
    def test_missing_router_rejected(self):
        with self.assertRaises(ValueError):RouterCollector(self.model,4,8,2,torch.device('cpu'))
    def test_zero_gate_still_counts_assignment(self):
        original=self.model.decoder.layers[0].mlp.router.forward
        def zero_secondary(x):
            score,route=original(x);selected=route.nonzero().reshape(-1,2,2)[:,:,1]
            score.zero_();score.scatter_(1,selected[:,:1],1);return score,route
        self.model.decoder.layers[0].mlp.router.forward=zero_secondary
        r=self.run_batch();self.assertEqual(r['counts'][0].sum(),6);self.assertAlmostEqual(r['token_gate_entropy_sum'][0],0)
    def test_nan_rejected(self):
        router=self.model.decoder.layers[0].mlp.router;original=router.forward
        def bad(x):
            score,route=original(x);score[route]=float('nan');return score,route
        router.forward=bad
        with self.assertRaises(ValueError):self.run_batch()
    def test_dropped_assignment_rejected(self):
        router=self.model.decoder.layers[0].mlp.router;original=router.forward
        def bad(x):
            score,route=original(x);route[0]=False;return score,route
        router.forward=bad
        with self.assertRaises(ValueError):self.run_batch()


class InputTests(unittest.TestCase):
    def test_schedule_100_on_16(self):
        plans=[scheduling(100,16,r) for r in range(16)]
        self.assertTrue(all(len(p)==7 for p in plans))
        self.assertEqual(sorted(i for p in plans for i in p if i>=0),list(range(100)))
    def test_preparation_deterministic(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);source=p/'source.jsonl'
            source.write_text('\n'.join(json.dumps(dict(id=f'a{i}',domain=str(i%2),input_ids=list(range(10)))) for i in range(10)))
            with contextlib.redirect_stdout(io.StringIO()):
                a=prepare(source,p/'a.jsonl',4,per_domain=3);b=prepare(source,p/'b.jsonl',4,per_domain=3)
            self.assertEqual(a,b);self.assertEqual(len(a),6);self.assertEqual(len(a[0]['input_ids']),4)
    def test_duplicate_ids_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'x';x=json.dumps(dict(id='a',domain='d',input_ids=[1]));p.write_text(x+'\n'+x)
            with self.assertRaises(ValueError):read_inputs(p,8)
    def test_oov_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t)/'x';p.write_text(json.dumps(dict(id='a',domain='d',input_ids=[8])))
            with self.assertRaises(ValueError):read_inputs(p,8,vocab_size=8)


class ReportTests(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);synthetic_run(p/'a');m,c,g,h,e=load_run(p/'a')
            self.assertEqual(c.shape,(5,3,8));self.assertEqual(e.shape,(2,3,3,8))
            self.assertEqual(c[:,0].sum(),40)
    def test_reports_and_comparison(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);synthetic_run(p/'a',1);synthetic_run(p/'b',2)
            with contextlib.redirect_stdout(io.StringIO()):summarize(p/'b',p/'report',p/'a')
            import pandas as pd
            cmp=pd.read_csv(p/'report/checkpoint_comparison.csv')
            self.assertTrue(np.allclose(cmp.cv_delta,0));self.assertTrue(np.allclose(cmp.load_js_divergence_nats,0))
            self.assertTrue((p/'report/load_heatmaps.png').stat().st_size>1000)
    def test_mismatched_inputs_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);synthetic_run(p/'a',1);m=synthetic_run(p/'b',2);m['input_sha256']='different'
            (p/'b/manifest.json').write_text(json.dumps(m))
            with self.assertRaises(ValueError):summarize(p/'b',p/'report',p/'a')
    def test_corrupt_counts_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t);synthetic_run(p/'a');path=p/'a/rank_00000.npz'
            with np.load(path) as f:data={k:f[k].copy() for k in f.files}
            data['counts'][0,0,0]+=1;np.savez_compressed(path,**data)
            with self.assertRaises(ValueError):load_run(p/'a')


@unittest.skipUnless(REPO,'Pass --repo to check repository API contracts')
class RepositoryTests(unittest.TestCase):
    def test_native_provider(self):
        tree=ast.parse((REPO/'pretrain_gpt.py').read_text())
        f=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='model_provider')
        self.assertEqual([a.arg for a in f.args.args],['pre_process','post_process'])
    def test_router_contract(self):
        tree=ast.parse((REPO/'mindspeed_llm/core/transformer/moe/router.py').read_text())
        f=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='topk_router_forward_patch')
        self.assertTrue(any(isinstance(x,ast.Return) and isinstance(x.value,ast.Tuple) and
                            [getattr(y,'id',None) for y in x.value.elts]==['scores','routing_map'] for x in ast.walk(f)))


if __name__=='__main__':unittest.main(verbosity=2)
