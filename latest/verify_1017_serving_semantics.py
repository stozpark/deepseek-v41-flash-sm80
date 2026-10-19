#!/usr/bin/env python3
"""Fail-closed real-server regression for DeepSeek-V4.1-Flash-1017.

This supplements (not replaces) A100 kernel unit tests: upstream #62417
reports DSpark corruption specifically in chat/thinking and structured tools,
while plain completions still work. Run once with DSpark off and once with
DSpark on (sequential servers on the same TP8). Check strict semantics, tool
JSON, multi-image routing and optional baseline comparison.

No checkpoints or network services other than the specified local vLLM
endpoint are accessed. Only report PASS for cases actually executed.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import struct
import urllib.error
import urllib.request
import zlib
from pathlib import Path


def solid_png(color: tuple[int, int, int], width: int=36, height: int=36) -> str:
    pixels=b"".join(b"\x00"+bytes(color)*width for _ in range(height))
    def chunk(kind: bytes, data: bytes) -> bytes:
        return (struct.pack(">I",len(data))+kind+data
                +struct.pack(">I",zlib.crc32(kind+data)&0xffffffff))
    payload=(b"\x89PNG\r\n\x1a\n"
             +chunk(b"IHDR",struct.pack(">IIBBBBB",width,height,8,2,0,0,0))
             +chunk(b"IDAT",zlib.compress(pixels))
             +chunk(b"IEND",b""))
    return "data:image/png;base64,"+base64.b64encode(payload).decode()


def post(url: str, payload: dict) -> dict:
    req=urllib.request.Request(
        url.rstrip("/")+"/v1/chat/completions",
        data=json.dumps(payload,ensure_ascii=False).encode(),
        headers={"Content-Type":"application/json"},method="POST",
    )
    try:
        with urllib.request.urlopen(req,timeout=600) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        body=exc.read(1500).decode("utf-8","replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


def unwrap(out: dict) -> tuple[str,str,list]:
    if not out.get("choices") or not isinstance(out["choices"][0].get("message"),dict):
        raise AssertionError(f"Invalid chat result: {str(out)[:1000]}")
    msg=out["choices"][0]["message"]
    content=str(msg.get("content") or "")
    reasoning=str(msg.get("reasoning_content") or "")
    calls=msg.get("tool_calls") or []
    full=content+" "+reasoning
    # Catch upstream #62417 recurring phrases and repeated arithmetic tokens.
    # A threshold of >=6 consecutive 3-36-char fragments intentionally
    # catches pathological loops, not common repeated words.
    if re.search(r"(.{3,36}?)\1{5,}",full,re.DOTALL):
        raise AssertionError(f"Degenerate repeated output: {full[:400]}")
    if len(reasoning)>200 and len(set(reasoning[i:i+12] for i in range(0,len(reasoning)-12,12)))<3:
        raise AssertionError("Degenerate fixed reasoning stream")
    return content,reasoning,calls


def payload(model: str, messages: list, **kwargs) -> dict:
    return {"model":model,"messages":messages,"temperature":0,
            "max_tokens":512,**kwargs}


def run_case(base: str, body: dict, kind: str) -> dict:
    result=post(base,body)
    content, reasoning, calls=unwrap(result)
    return {"case":kind,"content":content,"reasoning":reasoning,
            "tool_calls":calls,"finish_reason":result["choices"][0].get("finish_reason"),
            "usage":result.get("usage")}


def verify_math(base: str, model: str) -> dict:
    out=run_case(base,payload(model,[{"role":"user","content":
        "Calculate 17 times 19. Answer with exactly the integer, no units."}]),"math")
    if not re.search(r"\b323\b",out["content"]):
        raise AssertionError(
            "17x19 math answer missing from final content; "
            f"content={out['content'][:300]!r}, reasoning={out['reasoning'][:120]!r}"
        )
    print("1017_CHAT_THINKING_ARITHMETIC_PASS")
    return out


def verify_tool(base: str, model: str) -> dict:
    tools=[{"type":"function","function":{
        "name":"get_gpu_alert",
        "description":"Return an alert status for a GPU node.",
        "parameters":{"type":"object","properties":{"node":{"type":"string"}},
                      "required":["node"],"additionalProperties":False}}}]
    body=payload(
        model,[{"role":"user","content":
            "Call get_gpu_alert for node gpu-03. Do not invent the result."}],
        tools=tools,
        tool_choice={"type":"function","function":{"name":"get_gpu_alert"}},
    )
    out=run_case(base,body,"tool")
    calls=out["tool_calls"]
    if len(calls)!=1 or calls[0].get("function",{}).get("name")!="get_gpu_alert":
        raise AssertionError(f"DSML tool parsing failed: {str(calls)[:500]}")
    raw=calls[0]["function"].get("arguments")
    arguments=json.loads(raw) if isinstance(raw,str) else raw
    if not isinstance(arguments,dict) or arguments.get("node")!="gpu-03":
        raise AssertionError(f"DSML tool JSON incorrect: {arguments!r}")
    print("1017_DSML_TOOL_CALL_JSON_PASS")
    return out


def verify_vision(base: str, model: str) -> dict:
    image_red=solid_png((255,0,0))
    image_blue=solid_png((0,0,255))
    def single(color: str, image_url: str) -> dict:
        out=run_case(base,payload(model,[{"role":"user","content":[
            {"type":"text","text":"Name the single dominant color of this image."},
            {"type":"image_url","image_url":{"url":image_url}}]}]),f"vision_{color}")
        if color not in out["content"].lower():
            raise AssertionError(f"Vision {color} image not identified: {out['content'][:300]!r}")
        return out
    red=single("red",image_red)
    blue=single("blue",image_blue)
    out=run_case(base,payload(model,[{"role":"user","content":[
            {"type":"text","text":
                "There are two images in order. What is the first color and second color?"},
            {"type":"image_url","image_url":{"url":image_red}},
            {"type":"image_url","image_url":{"url":image_blue}}]}]),"vision_ordered")
    response=out["content"].lower()
    if "red" not in response or "blue" not in response or response.index("red")>=response.index("blue"):
        raise AssertionError(f"Multi-image identity/order incorrect: {response[:300]!r}")
    print("1017_VISION_RED_BLUE_TWO_IMAGE_ORDER_PASS")
    return {"red":red,"blue":blue,"ordered":out}


def main() -> None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url",default="http://127.0.0.1:18005")
    p.add_argument("--model",default="deepseek-v41-flash-1017")
    p.add_argument("--vision",action="store_true")
    p.add_argument("--save",type=Path,required=True)
    p.add_argument("--compare-baseline",type=Path,
                   help="JSON from preceding no-spec TP8 server run")
    args=p.parse_args()
    results={"model":args.model,"math":verify_math(args.url,args.model),
             "tool":verify_tool(args.url,args.model)}
    if args.vision:
        results["vision"]=verify_vision(args.url,args.model)
    if args.compare_baseline:
        baseline=json.loads(args.compare_baseline.read_text())
        for name in ["math","tool"]:
            prev=baseline[name]["content"]
            curr=results[name]["content"]
            # Under greedy target verification, a changed final answer is a
            # blocker for DSpark rollout even if the target can be stochastic
            # under non-greedy sampling. Hidden reasoning text may differ.
            if prev!=curr:
                raise AssertionError(
                    f"DSpark changes greedy final {name}: off={prev[:200]!r}, on={curr[:200]!r}"
                )
        print("1017_DSPARK_VS_NO_SPEC_FINAL_CHAT_PARITY_PASS")
    args.save.parent.mkdir(parents=True,exist_ok=True)
    args.save.write_text(json.dumps(results,indent=2,ensure_ascii=False))
    print(f"1017_SERVICE_SEMANTICS_PASS output={args.save}")


if __name__=="__main__":
    main()
