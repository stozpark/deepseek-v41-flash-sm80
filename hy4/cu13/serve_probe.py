#!/usr/bin/env python3
"""Deterministic API check and matched-run comparison; optional long needle.

Capture a baseline, then use --reference against another partition/batch
configuration or a trusted supported-GPU run of the same full checkpoint.
A dummy fixture tests wiring only; never compare dummy to the real model.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import urllib.request
from reference_checks import compare_responses


def post(base,path,body,timeout):
    req=urllib.request.Request(base.rstrip("/")+path,data=json.dumps(body).encode(),
                               headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req,timeout=timeout) as r:
        return json.load(r)


def complete(base,prompt,timeout,max_tokens=8,model="hy4"):
    r=post(base,"/v1/completions",dict(model=model,prompt=prompt,temperature=0,
                  max_tokens=max_tokens,logprobs=5,seed=1234),timeout)
    c=r["choices"][0]; lp=c["logprobs"]
    return {"prompt":prompt,"prompt_tokens":r["usage"]["prompt_tokens"],
            "tokens":lp["tokens"],"logprobs":lp["token_logprobs"],
            "top_logprobs":lp["top_logprobs"],"text":c["text"]}


def long_needle(base,target,timeout,model="hy4"):
    # Exact prompt token budget is measured by the server; avoid assuming
    # words==tokens. Put the key near the middle, not just at the very end.
    key="ALPHA-7491-KAPPA"
    filler=" The archive contains unrelated records about weather, books and ordinary dates."
    def text(n):
        return filler*(n//2)+" The unique verification key is "+key+"."+filler*(n-n//2)+"\nReturn only the unique verification key:"
    def count(s):
        return len(post(base,"/tokenize",{"model":model,"prompt":s},timeout)["tokens"])
    low,high=0,target
    while low<high:
        mid=(low+high+1)//2
        if count(text(mid))<=target:
            low=mid
        else:
            high=mid-1
    prompt=text(low); measured=count(prompt)
    if measured<target-32:
        raise RuntimeError(f"failed to reach target token budget: {measured}/{target}")
    r=complete(base,prompt,timeout,32,model)
    if r["prompt_tokens"]<target-32 or key not in r["text"]:
        raise AssertionError(f"needle or context length failed at {r['prompt_tokens']}: {r['text']!r}")
    print("HY4_LONG_NEEDLE=PASS tokens",r["prompt_tokens"])
    return {"target":target,"actual":r["prompt_tokens"],"text":r["text"]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--url",default="http://127.0.0.1:8000")
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--model-name",default="hy4")
    p.add_argument("--label",default="unspecified")
    p.add_argument("--reference",type=Path)
    p.add_argument("--timeout",type=int,default=1800)
    p.add_argument("--interleaved",action="store_true")
    p.add_argument("--needle",type=int,action="append",default=[])
    a=p.parse_args()
    prompts=["The capital city of France is","Complete the sequence: 1, 2, 3, 4,",
             ("one two three four five six seven eight nine ten. "*300)+" Continue:"]
    results=[complete(a.url,t,a.timeout,model=a.model_name) for t in prompts]
    for r in results:
        compare_responses(r,r)
    if a.reference:
        old=json.loads(a.reference.read_text())["responses"]
        if len(old)!=len(results):
            raise RuntimeError("reference count mismatch")
        for ref,new in zip(old,results):
            compare_responses(ref,new)
        print("HY4_MATCHED_LOGPROBS=PASS")
    if a.interleaved:
        with ThreadPoolExecutor(max_workers=3) as pool:
            other=list(pool.map(lambda t:complete(a.url,t,a.timeout,model=a.model_name),prompts))
        for ref,new in zip(results,other):
            compare_responses(ref,new)
        print("HY4_INTERLEAVED=PASS")
    needles=[long_needle(a.url,n,a.timeout,a.model_name) for n in a.needle]
    a.output.write_text(json.dumps({"label":a.label,"model_name":a.model_name,
                    "responses":results,"needles":needles},indent=2)+"\n")
    print("HY4_API_PROBE=PASS")


if __name__=="__main__":
    main()
