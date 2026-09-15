#!/usr/bin/env python3
"""DeepSeek-V4.1-Flash long-context correctness smoke for the A100 server.

Calibrates a chat prompt with vLLM's /tokenize endpoint to approximately the
requested token count, places three independent records far apart, then asks the
model to reproduce them exactly. Runs deterministic requests first and optional
sampled requests matching the community long-context bug report
(temperature=1.0, top_p=0.95).

No third-party Python packages are required.
"""
from __future__ import annotations

import argparse
import json
import secrets
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def post_json(url: str, payload: dict, api_key: str | None, timeout: int) -> dict:
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body[:2000]}") from exc


def make_messages(repeats: int, sentinels: tuple[str, str, str]) -> list[dict]:
    a, b, c = sentinels
    q1 = repeats // 10
    q2 = repeats * 4 // 10
    q3 = repeats * 8 // 10
    filler = "The following sentence is ordinary filler context. "

    parts = [
        "You are checking long-context data integrity. Remember the three records "
        "ALPHA, BETA, and GAMMA even though they are separated by a large amount "
        "of irrelevant text. Do not infer or modify their values.\n\n",
        filler * q1,
        f"\nRECORD ALPHA VALUE: {a}\n",
        filler * (q2 - q1),
        f"\nRECORD BETA VALUE: {b}\n",
        filler * (q3 - q2),
        f"\nRECORD GAMMA VALUE: {c}\n",
        filler * (repeats - q3),
        "\nEND OF CONTEXT. Return exactly one line and nothing else in this form:\n"
        f"ALPHA={a}|BETA={b}|GAMMA={c}\n",
    ]
    return [{"role": "user", "content": "".join(parts)}]


def token_count(base: str, model: str, messages: list[dict], key: str | None, timeout: int) -> int:
    out = post_json(
        f"{base}/tokenize",
        {"model": model, "messages": messages, "add_generation_prompt": True},
        key,
        timeout,
    )
    if "count" not in out:
        raise RuntimeError(f"Unexpected /tokenize response: {str(out)[:1000]}")
    return int(out["count"])


def calibrate(
    base: str,
    model: str,
    target: int,
    sentinels: tuple[str, str, str],
    key: str | None,
    timeout: int,
) -> tuple[list[dict], int, int]:
    lo, hi = 1, max(1000, target // 2)
    while token_count(base, model, make_messages(hi, sentinels), key, timeout) < target:
        lo, hi = hi, hi * 2
        if hi > target * 4:
            raise RuntimeError("Could not bracket target token count")

    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        messages = make_messages(mid, sentinels)
        count = token_count(base, model, messages, key, timeout)
        candidate = (abs(count - target), messages, count, mid)
        if best is None or candidate[0] < best[0]:
            best = candidate
        if count < target:
            lo = mid + 1
        elif count > target:
            hi = mid - 1
        else:
            break

    assert best is not None
    _, messages, count, repeats = best
    return messages, count, repeats


def one_request(
    base: str,
    model: str,
    messages: list[dict],
    expected: str,
    key: str | None,
    timeout: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> dict:
    started = time.time()
    out = post_json(
        f"{base}/v1/chat/completions",
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": 256,
            "seed": seed,
        },
        key,
        timeout,
    )
    elapsed = time.time() - started
    try:
        msg = out["choices"][0]["message"]
        content = msg.get("content") or ""
    except Exception as exc:
        raise RuntimeError(f"Malformed chat response: {str(out)[:2000]}") from exc

    normalized = "".join(content.split())
    expected_norm = "".join(expected.split())
    ok = expected_norm in normalized
    usage = out.get("usage") or {}
    return {
        "ok": ok,
        "elapsed_s": round(elapsed, 3),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "seed": seed,
        "temperature": temperature,
        "content_preview": content[:500],
    }


def run_group(
    label: str,
    n: int,
    concurrency: int,
    base_seed: int,
    temperature: float,
    top_p: float,
    **kwargs,
) -> list[dict]:
    results: list[dict] = []
    if n <= 0:
        return results
    print(f"=== {label}: runs={n} concurrency={concurrency} temperature={temperature} top_p={top_p} ===")
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {
            pool.submit(
                one_request,
                temperature=temperature,
                top_p=top_p,
                seed=base_seed + i,
                **kwargs,
            ): i
            for i in range(n)
        }
        for future in as_completed(futures):
            i = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"ok": False, "run": i, "error": repr(exc)}
            else:
                result["run"] = i
            results.append(result)
            print(json.dumps({"group": label, **result}, ensure_ascii=False))
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:18005")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--target-tokens", type=int, default=60000)
    ap.add_argument("--deterministic-runs", type=int, default=4)
    ap.add_argument("--sampled-runs", type=int, default=20)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument(
        "--strict-sampled",
        action="store_true",
        help="Return failure if any temperature=1/top_p=.95 run misses the records.",
    )
    args = ap.parse_args()
    base = args.base_url.rstrip("/")

    sentinels = tuple(f"{secrets.token_hex(8).upper()}" for _ in range(3))
    expected = f"ALPHA={sentinels[0]}|BETA={sentinels[1]}|GAMMA={sentinels[2]}"

    print(f"Calibrating prompt near {args.target_tokens} tokens via {base}/tokenize ...")
    messages, count, repeats = calibrate(
        base, args.model, args.target_tokens, sentinels, args.api_key, args.timeout
    )
    print(f"calibrated_prompt_tokens={count} filler_repeats={repeats}")
    print(f"expected={expected}")

    common = dict(
        base=base,
        model=args.model,
        messages=messages,
        expected=expected,
        key=args.api_key,
        timeout=args.timeout,
    )
    deterministic = run_group(
        "deterministic",
        args.deterministic_runs,
        args.concurrency,
        1000,
        0.0,
        1.0,
        **common,
    )
    sampled = run_group(
        "sampled-community-repro",
        args.sampled_runs,
        args.concurrency,
        2000,
        1.0,
        0.95,
        **common,
    )

    det_fail = sum(not r.get("ok", False) for r in deterministic)
    sampled_fail = sum(not r.get("ok", False) for r in sampled)
    print(
        json.dumps(
            {
                "prompt_tokens": count,
                "deterministic": {"runs": len(deterministic), "failures": det_fail},
                "sampled": {"runs": len(sampled), "failures": sampled_fail},
            },
            indent=2,
        )
    )

    if det_fail:
        print("FAIL: deterministic long-context correctness failure", file=sys.stderr)
        return 2
    if args.strict_sampled and sampled_fail:
        print("FAIL: sampled long-context correctness failure", file=sys.stderr)
        return 3
    if sampled_fail:
        print(
            "WARN: sampled runs missed one or more records; rerun with --strict-sampled "
            "and compare against an H100/reference server before production use.",
            file=sys.stderr,
        )
    print("Long-context smoke completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
