#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import struct
import urllib.error
import urllib.request
import zlib


def make_png(width: int = 64, height: int = 64) -> bytes:
    """Create a small RGB PNG using only the Python standard library."""
    rows = []
    for y in range(height):
        row = bytearray([0])
        for x in range(width):
            if width // 4 <= x < 3 * width // 4 and height // 4 <= y < 3 * height // 4:
                row.extend((0, 255, 0))
            else:
                row.extend((255, 0, 0))
        rows.append(bytes(row))
    raw = b"".join(rows)

    def chunk(kind: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + kind
            + payload
            + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, level=9))
        + chunk(b"IEND", b"")
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="DeepSeek-V4.1-Flash multimodal API smoke test")
    ap.add_argument("--url", default="http://127.0.0.1:18005/v1/chat/completions")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()

    png_b64 = base64.b64encode(make_png()).decode("ascii")
    payload = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe the simple geometric color pattern in this image in one short sentence.",
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{png_b64}"},
                    },
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 64,
    }
    req = urllib.request.Request(
        args.url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            status = resp.status
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise SystemExit(f"ERROR: multimodal request returned HTTP {exc.code}: {detail}") from exc
    except Exception as exc:
        raise SystemExit(f"ERROR: multimodal request failed: {exc}") from exc

    if status != 200:
        raise SystemExit(f"ERROR: expected HTTP 200, got {status}: {body}")
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise SystemExit(f"ERROR: response has no choices: {body}")
    message = choices[0].get("message", {})
    content = message.get("content")
    if not isinstance(content, str) or not content.strip():
        raise SystemExit(f"ERROR: multimodal response content is empty: {message}")

    print("DEEPSEEK-V4.1 MULTIMODAL API SMOKE: OK")
    print(content.strip())


if __name__ == "__main__":
    main()
