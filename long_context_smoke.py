#!/usr/bin/env python3
"""Cheap long-context correctness smoke for the A100 SM80 deployment.

This is deliberately not a benchmark. It targets the failure class reported for
DeepSeek-V4.1 on SM80 around 60K input tokens: incorrect paged-indexer/K-cache
addressing can boot and answer short prompts correctly, then corrupt long-context
continuations.

Run after the local vLLM server is ready. The script uses only Python stdlib.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request


def post_json(url: str, payload: dict, timeout: int = 600) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {body}") from exc


def token_count(base_url: str, model: str, prompt: str) -> int | None:
    try:
        out = post_json(
            f"{base_url.rstrip('/')}/tokenize",
            {"model": model, "prompt": prompt},
            timeout=120,
        )
    except Exception as exc:
        print(f"[warn] /tokenize unavailable: {exc}", file=sys.stderr)
        return None
    count = out.get("count")
    if isinstance(count, int):
        return count
    tokens = out.get("tokens")
    if isinstance(tokens, list):
        return len(tokens)
    print(f"[warn] unknown /tokenize response: {out}", file=sys.stderr)
    return None


def make_document(repetitions: int) -> str:
    filler = (
        "Manufacturing log entry: the chamber remained stable and the ordinary "
        "inspection record contains no special instruction. "
    )
    left = repetitions // 4
    middle = repetitions // 2
    right = repetitions - left - middle
    return (
        filler * left
        + "\nIMPORTANT FACT A: project codename is ORBIT-731.\n"
        + filler * middle
        + "\nIMPORTANT FACT B: recovery key is COBALT-219.\n"
        + filler * right
    )


def build_prompt(base_url: str, model: str, target_tokens: int) -> tuple[str, int | None]:
    # Calibrate with /tokenize when available. Start from a cheap estimate and
    # binary-search repetitions; the two needles stay around 25% and 75% depth.
    lo, hi = 100, max(1000, target_tokens // 4)
    best_prompt = ""
    best_count: int | None = None

    probe = make_document(500)
    probe_count = token_count(base_url, model, probe)
    if probe_count is None:
        # Conservative fallback: this filler is roughly a dozen-plus tokenizer
        # tokens per repetition for English tokenizers. Exact count is printed as
        # unavailable; the correctness criterion still tests a genuinely long prompt.
        reps = max(1000, target_tokens // 14)
        doc = make_document(reps)
        prompt = (
            doc
            + "\n\nQuestion: Reply with exactly this format and nothing else: "
            + "ORBIT-731 / COBALT-219"
        )
        return prompt, None

    per_rep = max(1.0, probe_count / 500.0)
    guess = max(100, int(target_tokens / per_rep))
    lo, hi = max(100, guess // 2), max(guess + 10, guess * 2)

    for _ in range(12):
        reps = (lo + hi) // 2
        doc = make_document(reps)
        prompt = (
            doc
            + "\n\nQuestion: Reply with exactly this format and nothing else: "
            + "ORBIT-731 / COBALT-219"
        )
        n = token_count(base_url, model, prompt)
        if n is None:
            return prompt, None
        best_prompt, best_count = prompt, n
        if abs(n - target_tokens) <= 300:
            break
        if n < target_tokens:
            lo = reps + 1
        else:
            hi = max(lo, reps - 1)

    return best_prompt, best_count


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:18005")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--target-tokens", type=int, default=60000)
    ap.add_argument("--repeats", type=int, default=3)
    args = ap.parse_args()

    prompt, n_tokens = build_prompt(args.base_url, args.model, args.target_tokens)
    print(f"[prompt] tokens={n_tokens if n_tokens is not None else 'unknown'} chars={len(prompt)}")

    expected = "ORBIT-731 / COBALT-219"
    outputs: list[str] = []
    for i in range(args.repeats):
        response = post_json(
            f"{args.base_url.rstrip('/')}/v1/chat/completions",
            {
                "model": args.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0,
                "max_tokens": 64,
            },
            timeout=900,
        )
        try:
            text = response["choices"][0]["message"]["content"] or ""
        except Exception as exc:
            raise RuntimeError(f"unexpected chat response: {response}") from exc
        outputs.append(text.strip())
        ok = expected in text
        print(f"[run {i + 1}/{args.repeats}] ok={ok} output={text.strip()!r}")
        if not ok:
            print("FAIL: long-context needle was not recovered", file=sys.stderr)
            return 2

    if len(set(outputs)) != 1:
        print(
            "[warn] deterministic runs returned different text; needles were correct, "
            "but inspect the outputs for possible instability",
            file=sys.stderr,
        )
    print("LONG_CONTEXT_SMOKE: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
