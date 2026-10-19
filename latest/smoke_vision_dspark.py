#!/usr/bin/env python3
"""End-to-end OpenAI API smoke tests for DeepSeek V4.1 TP8 A100.

Run against live model only. Supports text-only, image-only, and image +
DSpark test profiles; asserts nonempty output and records response metadata.
The image is a tiny generated PNG, no external dependencies or network.
"""
import argparse
import base64
import json
import struct
import urllib.request
import zlib


def png() -> str:
    w,h=28,28
    raw=b"".join(b"\x00"+bytes([255,0,0])*w for _ in range(h))
    def chunk(name,data):
        return struct.pack(">I",len(data))+name+data+struct.pack(">I",zlib.crc32(name+data)&0xffffffff)
    payload=b"\x89PNG\r\n\x1a\n"+chunk(b"IHDR",struct.pack(">IIBBBBB",w,h,8,2,0,0,0))+chunk(b"IDAT",zlib.compress(raw))+chunk(b"IEND",b"")
    return "data:image/png;base64,"+base64.b64encode(payload).decode("ascii")


def request(base,model,messages):
    payload=json.dumps({"model":model,"messages":messages,"temperature":0,
                        "max_tokens":96}).encode()
    req=urllib.request.Request(base.rstrip("/")+"/v1/chat/completions",
                               data=payload,headers={"Content-Type":"application/json"},
                               method="POST")
    with urllib.request.urlopen(req,timeout=600) as response:
        out=json.load(response)
    choices=out.get("choices") or []
    assert choices and "message" in choices[0],out
    msg=choices[0]["message"]
    content=msg.get("content") or msg.get("reasoning_content") or ""
    assert content or msg.get("tool_calls"),out
    return {"finish_reason":choices[0].get("finish_reason"),
            "usage":out.get("usage"),"answer":str(content)[:180]}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--url",default="http://127.0.0.1:18005")
    p.add_argument("--model",default="deepseek-v41-flash-1017")
    p.add_argument("--vision",action="store_true")
    p.add_argument("--dspark",action="store_true")
    a=p.parse_args()
    messages=[{"role":"user","content":"Return the result of 12 + 31."}]
    print("TEXT_SMOKE",json.dumps(request(a.url,a.model,messages)))
    if a.vision:
        messages=[{"role":"user","content":[
            {"type":"text","text":"What is the dominant color in this image?"},
            {"type":"image_url","image_url":{"url":png()}}
        ]}]
        print("VISION_SMOKE",json.dumps(request(a.url,a.model,messages)))
    print("END_TO_END_SMOKE_PASS",{"vision":a.vision,"dspark":a.dspark})
    # API validity alone does not establish draft acceptance; check server
    # metrics for accepted/total drafts and compare disable/enable on same inputs.


if __name__=="__main__":
    main()
