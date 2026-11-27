"""Test the actual CPU validation helpers and fixture metadata."""
import json
from pathlib import Path
import pytest
from image_verify import create_manifest, verify_files, MODULES, PREFIX, PORT_FILES
from reference_checks import check_topk, compare_responses
from make_fixture import fixture_config
from validate_logs import validate


def test_manifest_and_drift(tmp_path):
    src,port=tmp_path/"source",tmp_path/"port"
    port.mkdir()
    (src/PREFIX).mkdir(parents=True)
    for n in MODULES:
        (src/(PREFIX+n+".py")).write_text(n)
    for n in PORT_FILES:
        (port/n).write_text(n)
    manifest=create_manifest(src,port)
    verify_files(src,port,manifest)
    (src/(PREFIX+"attention.py")).write_text("not the pinned patch")
    with pytest.raises(RuntimeError,match="hash mismatch"):
        verify_files(src,port,manifest)


def test_manifest_rejects_missing_entries(tmp_path):
    src,port=tmp_path/"source",tmp_path/"port"
    with pytest.raises(RuntimeError,match="incomplete"):
        verify_files(src,port,{"source_commit":"3470b0db1820f699960d1376223602739a0f4ac3","cuda_toolkit":"13.2.1","modules":{},"port":{}})


def test_topk_ties_and_mask():
    check_topk([0,2],[5.,3.,4.,4.],4,2)
    check_topk([0,3],[5.,3.,4.,4.],4,2)
    check_topk([1,0,-1],[3.,5.],2,3)
    with pytest.raises(AssertionError):
        check_topk([0,0],[5.,4.,3.],3,2)
    with pytest.raises(AssertionError):
        check_topk([0,2],[5.,4.,3.],3,2)
    with pytest.raises(AssertionError):
        check_topk([0,2],[5.,4.,3.],2,2)


def test_response_parity_and_mismatch():
    r={"prompt":"x","prompt_tokens":1,"tokens":["a","b"],"logprobs":[-0.2,-1.]}
    compare_responses(r,dict(r,logprobs=[-0.21,-1.01]))
    with pytest.raises(AssertionError):
        compare_responses(r,dict(r,tokens=["a","c"]))
    with pytest.raises(AssertionError):
        compare_responses(r,dict(r,logprobs=[-0.4,-1.]))
    with pytest.raises(AssertionError):
        compare_responses(r,dict(r,prompt_tokens=2))


def test_fixture_config_does_not_mutate_source():
    cfg={"architectures":["HYV4VLForConditionalGeneration"],"text_config":{"num_hidden_layers":78},
         "quantization_config":{"quant_method":"compressed-tensors"},"dfly_num_hidden_layers":5}
    original=json.dumps(cfg,sort_keys=True)
    a=fixture_config(cfg); b=fixture_config(cfg,True)
    assert json.dumps(cfg,sort_keys=True)==original
    assert a["text_config"]["indexer_types"]==["full","full","shared","shared"]
    assert a["text_config"]["num_hidden_layers"]==4
    assert "quantization_config" not in a and "quantization_config" in b
    assert "dfly_num_hidden_layers" not in a
    assert a["vision_config"]["out_hidden_size"]==a["text_config"]["hidden_size"]
    assert a["vision_config"]["text_hidden_size"]==a["text_config"]["hidden_size"]

def test_rank_coverage():
    lines=[]
    for p in range(2):
        for t in range(2):
            lines.append(f"HY4_MODEL_DISPATCH=PASS pp={p} tp={t} layers={p*2}:{p*2+2} attention=2 moe={1 if p==0 else 2} fp8=True")
            for l in range(p*2,p*2+2):
                for phase in ("prefill","decode"):
                    lines.append(f"HY4_SPARSE_FORWARD=PASS pp={p} tp={t} layer={l} phase={phase}")
    text="\n".join(lines)
    assert validate(text,2,[2,2],True)==(4,16)
    with pytest.raises(AssertionError,match="forward coverage"):
        validate("\n".join(lines[:-1]),2,[2,2],True)
    with pytest.raises(AssertionError,match="rank dispatch"):
        validate("\n".join(lines[1:]),2,[2,2],True)
    with pytest.raises(AssertionError,match="NCCL"):
        validate(text,2,[2,2],True,True)

def test_top_candidate_comparison():
    r={"prompt":"x","prompt_tokens":1,"tokens":["a"],"logprobs":[-0.2],
       "top_logprobs":[{"a":-0.2,"b":-1.0,"c":-1.01}]}
    compare_responses(r,dict(r,top_logprobs=[{"a":-0.21,"b":-1.01,"d":-1.0}]))
    with pytest.raises(AssertionError,match="top candidate"):
        compare_responses(r,dict(r,top_logprobs=[{"a":-0.2,"b":-2.0,"c":-1.01}]))
