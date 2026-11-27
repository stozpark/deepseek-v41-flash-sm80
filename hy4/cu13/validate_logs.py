#!/usr/bin/env python3
"""Check complete rank/layer coverage in captured fixture/full logs.

This is a coverage gate, not evidence of correct outputs on its own. Run
serve_probe.py with a matched reference as a separate numerical/API gate.
"""
import argparse
from pathlib import Path
import re

DISPATCH = re.compile(r"HY4_MODEL_DISPATCH=PASS pp=(\d+) tp=(\d+) layers=(\d+):(\d+) attention=(\d+) moe=(\d+) fp8=(True|False)")
FORWARD = re.compile(r"HY4_SPARSE_FORWARD=PASS pp=(\d+) tp=(\d+) layer=(\d+) phase=(prefill|decode)")
NCCL = re.compile(r"HY4_NCCL_TP_PP=PASS rank (\d+)")


def validate(text,tp,partition,fp8,require_nccl=False):
    expected_dispatch={}
    expected_forward=set()
    start=0
    for pp,num in enumerate(partition):
        end=start+num
        for rank in range(tp):
            expected_dispatch[pp,rank]=(start,end,num,num-(start==0),fp8)
            for layer in range(start,end):
                for phase in ("prefill","decode"):
                    expected_forward.add((pp,rank,layer,phase))
        start=end
    got={}
    for a,b,s,e,n,moe,q in DISPATCH.findall(text):
        key=int(a),int(b)
        value=int(s),int(e),int(n),int(moe),q=="True"
        if key in got and got[key]!=value:
            raise AssertionError(f"conflicting dispatch report {key}")
        got[key]=value
    if got!=expected_dispatch:
        bad={k:(expected_dispatch.get(k),got.get(k)) for k in expected_dispatch.keys()|got.keys()
             if expected_dispatch.get(k)!=got.get(k)}
        raise AssertionError(f"incorrect or missing rank dispatch: {bad}")
    actual={(int(p),int(t),int(l),ph) for p,t,l,ph in FORWARD.findall(text)}
    if actual!=expected_forward:
        raise AssertionError(f"forward coverage missing={len(expected_forward-actual)}, extra={len(actual-expected_forward)}")
    if require_nccl and set(map(int,NCCL.findall(text)))!=set(range(16)):
        raise AssertionError("missing 16-rank NCCL evidence")
    return len(got),len(actual)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("logs",type=Path,nargs="+")
    p.add_argument("--tp",type=int,required=True)
    p.add_argument("--partition",required=True)
    p.add_argument("--fp8",action="store_true")
    p.add_argument("--nccl",action="store_true")
    a=p.parse_args()
    counts=[int(x) for x in a.partition.split(",")]
    if a.tp<1 or min(counts)<1:
        raise RuntimeError("invalid topology")
    n,ph=validate("\n".join(f.read_text(errors="replace") for f in a.logs),
                  a.tp,counts,a.fp8,a.nccl)
    print(f"HY4_LOG_COVERAGE=PASS ranks={n} layer_phases={ph}")


if __name__=="__main__":
    main()
