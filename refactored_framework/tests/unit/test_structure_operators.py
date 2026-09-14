import numpy as np
import pandas as pd
import pytest

from hypergraph_ed.structure import Constraint, ConstraintProgram, ColumnSpec, merge_programs
from hypergraph_ed.structure.induce import local_program
from hypergraph_ed.operators import execute
from hypergraph_ed.data.table import semantic_type, assign_folds, parse_number
from hypergraph_ed.evidence import Status
from hypergraph_ed.evidence.fusion import evidence_features


def test_duration_is_not_percentage():
    assert semantic_type("Duration") == "duration"
    assert semantic_type("average") == "unknown"
    table = pd.DataFrame({"Duration":["109 min","120 min","143 min"]})
    program = local_program(table)
    assert not any(h.operator=="range" for h in program.constraints)
    assert not any(e.status==Status.VIOLATION for e in execute(table,program))


@pytest.mark.parametrize("edge",[
    dict(operator="python",params={}),
    dict(operator="range",params={"min":0,"max":100,"unit":"percent","semantic_type":"percentage"}),
    dict(operator="regex",params={"pattern":"(a+)+$"}),
    dict(operator="arithmetic",params={"expression":"__import__('os')", "variables":{"x":"Duration"}}),
])
def test_unsafe_or_incompatible_structure_rejected(edge):
    h=Constraint(id="x",participants=["Duration"],targets=["Duration"],**edge)
    with pytest.raises(ValueError):
        ConstraintProgram(columns=[ColumnSpec(name="Duration",semantic_type="duration",unit="minute")],constraints=[h])


def test_scope_pass_abstain_and_duplicate_sources():
    table=pd.DataFrame({"x":["2","oops",""],"kind":["a","b","a"]})
    h=Constraint(id="r",operator="range",participants=["x"],targets=["x"],conditions=[{"column":"kind","value":"a"}],params={"min":0,"max":1})
    program=ConstraintProgram(columns=[ColumnSpec(name=c) for c in table],constraints=[h])
    ev=execute(table,program)
    assert [e.status for e in ev]==[Status.VIOLATION,Status.ABSTAIN,Status.ABSTAIN]
    a=evidence_features(ev,3,list(table))
    b=evidence_features(ev+ev,3,list(table))
    np.testing.assert_array_equal(a,b)
    assert len(merge_programs([program,program]).constraints)==1


def test_duplicate_rows_same_fold():
    table=pd.DataFrame({"x":["a","b","c","a","b","d"]})
    f=assign_folds(table,3,42)
    assert f[0]==f[3] and f[1]==f[4] and len(set(f))==3


def test_regex_timeout_is_abstention(monkeypatch):
    from hypergraph_ed.operators import engine
    def timeout(*args,**kwargs):
        raise TimeoutError()
    monkeypatch.setattr(engine.bounded_regex,'fullmatch',timeout)
    table=pd.DataFrame({'code':['AAA']})
    h=Constraint(id='regex',operator='regex',participants=['code'],targets=['code'],params={'pattern':'[A-Z]+'})
    ev=execute(table,ConstraintProgram(columns=[ColumnSpec(name='code')],constraints=[h]))
    assert ev[0].status==Status.ABSTAIN


def test_arithmetic_and_temporal_operators():
    table=pd.DataFrame({"price":["2","2"],"qty":["3","3"],"total":["6","7"]})
    h=Constraint(id="arith",operator="arithmetic",participants=list(table),targets=["total"],params={"expression":"p*q-t","variables":{"p":"price","q":"qty","t":"total"},"tolerance":0.01})
    ev=execute(table,ConstraintProgram(columns=[ColumnSpec(name=c) for c in table],constraints=[h]))
    assert ev[0].status==Status.PASS and ev[1].status==Status.VIOLATION
    assert parse_number("109 min")== (109.,"minute")
    assert parse_number("1,2") is None
    assert parse_number("1,234") == (1234.,"")
