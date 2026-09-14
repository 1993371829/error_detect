import numpy as np
import torch

from hypergraph_ed.data import read_table
from hypergraph_ed.data.encoding import CellEncoder, SemanticStore
from hypergraph_ed.structure.induce import local_program
from hypergraph_ed.operators import execute
from hypergraph_ed.evidence.fusion import evidence_features,relation_quality
from hypergraph_ed.models.samples import SampleBuilder
from hypergraph_ed.models import build_model
from hypergraph_ed.models.train import train_fold,objective


def components(config,fixture_dir,tmp_path):
    table=read_table(fixture_dir/'tiny_dirty.csv')
    semantic=SemanticStore(config.model,table,tmp_path)
    ref=table.iloc[:8]
    encoder=CellEncoder(ref,semantic,500)
    program=local_program(table)
    ev=execute(table,program,ref)
    cfg=config.model.model_copy(update={'input_dim':encoder.dim,'column_count':len(table.columns),'category_count':encoder.category_count,'target_sizes':[s['size'] for s in encoder.specs],'target_kinds':[s['kind'] for s in encoder.specs]})
    builder=SampleBuilder(table,list(range(8)),encoder,program,evidence_features(ev,len(table),list(table.columns)),relation_quality(program,ev),cfg,{(0,'age'):0,(1,'age'):1})
    return cfg,builder


def test_masking_real_gradient_checkpoint_and_replay(config,fixture_dir,tmp_path):
    torch.set_num_threads(2)
    cfg,builder=components(config,fixture_dir,tmp_path)
    model=build_model(cfg)
    # Held-out row name cannot enter context as the same cell.
    a=builder.sample(8,0)
    old=builder.features[8,0].copy()
    builder.features[8,0]+=100
    b=builder.sample(8,0)
    np.testing.assert_array_equal(a['context'],b['context'])
    builder.features[8,0]=old
    before={k:v.clone() for k,v in model.state_dict().items()}
    batch=builder.batch([(0,3),(1,3)],torch.device('cpu'),synthetic=True)
    loss=objective(model(batch),batch,0)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
    history=train_fold(model,builder,list(range(8)),cfg,tmp_path/'fold.pt','identity',42)
    assert history and all(np.isfinite(x['train_loss']) for x in history)
    assert any(not torch.equal(v,model.state_dict()[k]) for k,v in before.items())
    model.eval()
    with torch.no_grad(): expected=model(batch)['logits'].clone()
    restored=build_model(cfg)
    train_fold(restored,builder,list(range(8)),cfg,tmp_path/'fold.pt','identity',42)
    restored.eval()
    with torch.no_grad(): torch.testing.assert_close(expected,restored(batch)['logits'])
    assert all(len(q)==5 for q in builder.quality.values())


def test_early_stopping(config,fixture_dir,tmp_path):
    cfg,builder=components(config,fixture_dir,tmp_path)
    cfg=cfg.model_copy(update={'epochs':10,'patience':1,'learning_rate':0.0})
    model=build_model(cfg)
    history=train_fold(model,builder,list(range(8)),cfg,tmp_path/'early.pt','early',1)
    assert len(history)<10
