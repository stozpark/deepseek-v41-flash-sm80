"""CPU oracles for boundary/tie-aware Top-K and response parity."""
import math


def check_topk(indices, logits, length, k, tol=0.0):
    """Reject duplicates, out-of-range positions and incorrect padding.
    A near-tie within an explicitly bounded score error is not an index bug.
    """
    expected_count = min(length, k)
    valid = [i for i in indices if i != -1]
    if len(indices) != k or len(valid) != expected_count or len(set(valid)) != len(valid):
        raise AssertionError("Top-K count, duplicates, or padding")
    if any(not 0 <= i < length for i in valid):
        raise AssertionError("Top-K outside causal bounds")
    if length == 0:
        return
    if len(logits) < length or not all(math.isfinite(x) for x in logits[:length]):
        raise AssertionError("invalid reference scores")
    ranked = sorted(range(length), key=lambda i: logits[i], reverse=True)
    threshold = logits[ranked[expected_count - 1]]
    chosen = set(valid)
    if any(logits[i] < threshold - tol for i in chosen):
        raise AssertionError("selected below threshold")
    if any(logits[i] > threshold + tol and i not in chosen for i in ranked):
        raise AssertionError("missed unambiguous Top-K element")


def compare_responses(reference, actual, atol=0.05):
    """Deterministic serving check: same prompt/tokens, chosen logprobs close."""
    if reference["prompt"] != actual["prompt"]:
        raise AssertionError("different prompts")
    if reference["prompt_tokens"] != actual["prompt_tokens"]:
        raise AssertionError("different prompt token count")
    if not reference["tokens"] or reference["tokens"] != actual["tokens"]:
        raise AssertionError("different generated tokens")
    a, b = reference["logprobs"], actual["logprobs"]
    if len(a) != len(b) or len(a) != len(reference["tokens"]):
        raise AssertionError("missing token logprobs")
    if any(not (math.isfinite(x) and math.isfinite(y)) or abs(x-y) > atol
           for x, y in zip(a, b)):
        raise AssertionError("chosen-token logprob mismatch")
