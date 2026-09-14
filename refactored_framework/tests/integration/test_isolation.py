import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def test_standalone_copy_has_no_legacy_imports(fixture_dir,tmp_path):
    root=fixture_dir.parents[1]
    isolated=tmp_path/'standalone'
    shutil.copytree(root/'src',isolated/'src',ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copytree(root/'configs',isolated/'configs')
    shutil.copytree(fixture_dir,isolated/'tests/fixtures')
    forbidden={'stage_1','stage_2','stage_3','paths'}
    for source in (isolated/'src').rglob('*.py'):
        tree=ast.parse(source.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node,ast.Import):
                assert not {x.name.split('.')[0] for x in node.names}&forbidden
            if isinstance(node,ast.ImportFrom) and node.module:
                assert node.module.split('.')[0] not in forbidden
    env=dict(os.environ,PYTHONPATH=str(isolated/'src'),PYTHONDONTWRITEBYTECODE='1')
    out=isolated/'results'
    proc=subprocess.run([sys.executable,'-B','-m','hypergraph_ed','run','--input','tests/fixtures/tiny_dirty.csv','--config','configs/local_smoke.yaml','--output',str(out)],cwd=isolated,env=env,capture_output=True,text=True,timeout=90)
    assert proc.returncode==0,proc.stderr
    report=json.loads((out/'report.json').read_text())
    assert report['shape']==[12,5]
    proc=subprocess.run([sys.executable,'-B','-m','hypergraph_ed','run','--input','tests/fixtures/tiny_dirty.csv','--config','configs/local_smoke.yaml','--output',str(out),'--clean','not_allowed.csv'],cwd=isolated,env=env,capture_output=True,text=True,timeout=30)
    assert proc.returncode!=0 and 'unrecognized arguments' in proc.stderr
