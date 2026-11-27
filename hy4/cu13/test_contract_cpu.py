"""Test the actual CPU validation helpers and fixture metadata."""
import json
from pathlib import Path
import pytest
from image_verify import create_manifest, verify_files, MODULES, PREFIX, PORT_FILES
from reference_checks import check_topk, compare_responses
from make_fixture import fixture_config


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
