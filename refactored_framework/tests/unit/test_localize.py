import numpy as np
import pandas as pd
import pytest

from hypergraph_ed.structure import ColumnSpec, Constraint, ConstraintProgram
from hypergraph_ed.operators import execute
from hypergraph_ed.evidence.localize import localize


@pytest.mark.parametrize("values,expected",[(('B','X'),'lhs'),(('A','Y'),'rhs'),(('B','Y'),None)])
def test_independent_context_locates_either_side_and_abstains_multi(values,expected,config):
    table=pd.DataFrame([['A','X','one']]*4 + [['B','Y','two']]*4 + [[*values,'one']],columns=['lhs','rhs','anchor'])
    constraint=Constraint(id='fd',operator='fd',participants=['lhs','rhs'],targets=['lhs','rhs'],params={'lhs':['lhs'],'rhs':'rhs','min_support':3,'confidence':0.9})
    program=ConstraintProgram(columns=[ColumnSpec(name=c) for c in table],constraints=[constraint])
    evidence=execute(table,program,reference=table.iloc[:8])
    original=table.copy(deep=True)
    info=localize(table,program,evidence,np.array([0]*8+[1]),{},config)
    pd.testing.assert_frame_equal(table,original)
    if expected:
        assert info[(8,expected)][0]['target']==expected
    else:
        assert not any(v['target'] is not None for k,vs in info.items() if k[0]==8 for v in vs)
