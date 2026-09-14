import json
from pathlib import Path

import pandas as pd
import pytest

from hypergraph_ed.pipeline import run
from hypergraph_ed.evaluation import evaluate
from hypergraph_ed.artifacts import read_json


def test_end_to_end_resume_and_truth_isolation(config,fixture_dir,tmp_path):
    dirty=fixture_dir/'tiny_dirty.csv'
    raw=dirty.read_bytes()
    out=tmp_path/'run'
    report=run(dirty,config,out)
    assert report['shape']==[12,5] and report['simulated_llm']
    assert report['llm_calls']<=20 and report['charged_tokens']<=100000
    assert report['pseudo_judgments']<=128
    predictions=pd.read_csv(out/'detections.csv')
    assert len(predictions)==60 and predictions.error_score.between(0,1).all()
    clean=tmp_path/'clean.csv'
    clean.write_bytes((fixture_dir/'tiny_clean.csv').read_bytes())
    metrics=evaluate(dirty,clean,out/'detections.csv')
    assert metrics['overall']['tp']+metrics['overall']['fn']==3
    before=(out/'detections.csv').read_bytes()
    clean.write_bytes(raw)
    assert evaluate(dirty,clean,out/'detections.csv')['overall']['fn']==0
    assert run(dirty,config,out,resume=True)==report
    assert (out/'detections.csv').read_bytes()==before and dirty.read_bytes()==raw
    for fold in range(3):
        meta=read_json(out/f'models_round0/fold_{fold}_metadata.json')
        assert not set(meta['training_rows'])&set(meta['scoring_rows'])
    with pytest.raises(ValueError):
        run(dirty,config,out)


def test_invalid_structure_and_transport_failure_fall_back(config,fixture_dir,tmp_path):
    responses=tmp_path/'responses.json'
    responses.write_text(json.dumps({'structure':{'columns':[{'name':'invented'}],'constraints':[]},'audit':{},'labels':{}}))
    config.llm.fixture_path=str(responses)
    report=run(fixture_dir/'tiny_dirty.csv',config,tmp_path/'invalid')
    assert any('invalid structure' in x for x in report['issues'])
    assert report['constraints']>0
    config.llm.backend='openai'
    config.llm.tokenizer_path=str(tmp_path/'absent')
    report=run(fixture_dir/'tiny_dirty.csv',config,tmp_path/'no_counter')
    assert report['llm_calls']==0 and any('token counting unavailable' in x for x in report['issues'])


def test_semantic_backend_never_silently_falls_back(config,fixture_dir,tmp_path):
    config.model.embedding_backend='semantic'
    config.model.semantic_path=str(tmp_path/'missing_model')
    with pytest.raises(RuntimeError,match='semantic model missing'):
        run(fixture_dir/'tiny_dirty.csv',config,tmp_path/'semantic')
    assert not (tmp_path/'semantic/budget.json').exists()


def test_conflicting_pseudo_labels_are_discarded(config,fixture_dir,tmp_path):
    responses=tmp_path/'responses.json'
    table=pd.read_csv(fixture_dir/'tiny_dirty.csv',keep_default_na=False)
    labels=[{'row_id':i,'column':c,'is_error':v} for i in range(len(table)) for c in table for v in (True,False)]
    responses.write_text(json.dumps({'structure':{'columns':[],'constraints':[]},'audit':{},'labels':{'labels':labels}}))
    # Fixture byte accounting requires enough output reservation for this adversarial response.
    config.llm.fixture_path=str(responses)
    config.llm.max_output_tokens=12000
    report=run(fixture_dir/'tiny_dirty.csv',config,tmp_path/'conflict')
    result=read_json(tmp_path/'conflict/refinement.json')
    assert result['pseudo']==[]
    assert report['charged_tokens']<=100000
    assert any('conflicting pseudo label' in x for x in result['issues'])


def test_valid_induction_revision_and_real_second_training(config,fixture_dir,tmp_path,monkeypatch):
    from hypergraph_ed.llm.client import BudgetedClient
    def response(self,messages,phase):
        prompt=json.loads(messages[-1]['content'])
        if phase=='structure':
            result={'columns':[{'name':c} for c in ['name','city','state','age','Duration']], 'constraints':[{'id':'normalize','operator':'normalize','participants':['name'],'targets':['name'],'params':{'strip':True}}]}
        elif phase=='audit':
            result={'remove_constraint_ids':[], 'add_constraints':[]}
        else:
            result={'labels':[{'row_id':x['row_id'],'column':x['column'],'is_error':(x['row_id'],x['column']) in {(9,'age'),(8,'state')}} for x in prompt['samples']]}
        raw=json.dumps(result)
        return raw,self._count(messages),len(raw.encode())
    monkeypatch.setattr(BudgetedClient,'_complete',response)
    config.llm.max_output_tokens=4096
    report=run(fixture_dir/'tiny_dirty.csv',config,tmp_path/'teacher')
    revised=read_json(tmp_path/'teacher/refinement.json')
    assert revised['pseudo']
    assert any(h['source'].startswith('llm:') for h in revised['program']['constraints'])
    assert (tmp_path/'teacher/models_round1/fold_0.pt').exists()
    assert report['llm_calls']<=20 and report['charged_tokens']<=100000


def test_budget_exhaustion_still_completes_locally(config,fixture_dir,tmp_path):
    config.llm.max_output_tokens=100001
    report=run(fixture_dir/'tiny_dirty.csv',config,tmp_path/'exhausted')
    assert report['llm_calls']==0
    assert any('budget exhausted' in x for x in report['issues'])
    assert len(pd.read_csv(tmp_path/'exhausted/detections.csv'))==60
