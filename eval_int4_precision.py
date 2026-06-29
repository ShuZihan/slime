"""Evaluate INT4 (or BF16) Qwen3.5 generation accuracy on dapo-math via a running
sglang OpenAI server. Reuses slime's deepscaler rule-based reward for scoring.

Usage:
    python eval_int4_precision.py --url http://127.0.0.1:30000 --tag INT4 \
        --data /mnt/slime-qwen35/dapo-math-17k/dapo-math-17k.jsonl \
        --n 64 --max-tokens 8192 --temperature 0.0
"""
import argparse
import concurrent.futures as cf
import json
import sys
import time

import requests

sys.path.insert(0, "/root/slime")
from slime.rollout.rm_hub.deepscaler import get_deepscaler_rule_based_reward


def load_problems(path, n):
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            out.append(d)
            if len(out) >= n:
                break
    return out


def gen_one(url, model, messages, max_tokens, temperature):
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    r = requests.post(f"{url}/v1/chat/completions", json=payload, timeout=3600)
    r.raise_for_status()
    j = r.json()
    msg = j["choices"][0]["message"]
    # thinking models may put reasoning in reasoning_content; the scorer needs </think>
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    finish = j["choices"][0].get("finish_reason")
    if reasoning and "</think>" not in content:
        content = f"<think>{reasoning}</think>{content}"
    return content, finish


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--tag", default="MODEL")
    ap.add_argument("--data", required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # served model name = model path (sglang sets served_model_name to the path)
    info = requests.get(f"{args.url}/v1/models", timeout=60).json()
    model = info["data"][0]["id"]

    problems = load_problems(args.data, args.n)
    print(f"[{args.tag}] model={model} n={len(problems)} max_tokens={args.max_tokens} temp={args.temperature}", flush=True)

    results = [None] * len(problems)

    def work(i):
        d = problems[i]
        messages = d["prompt"]
        label = d["label"]
        t0 = time.time()
        content, finish = gen_one(args.url, model, messages, args.max_tokens, args.temperature)
        reward = get_deepscaler_rule_based_reward(content, label)
        return i, reward, finish, len(content), time.time() - t0

    t_start = time.time()
    correct = 0
    truncated = 0
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(work, i) for i in range(len(problems))]
        for fut in cf.as_completed(futs):
            i, reward, finish, clen, dt = fut.result()
            results[i] = {"idx": i, "reward": reward, "finish": finish, "len": clen, "label": problems[i]["label"]}
            correct += reward
            if finish == "length":
                truncated += 1
            print(f"[{args.tag}] #{i:03d} reward={reward} finish={finish} len={clen} {dt:.1f}s", flush=True)

    n = len(problems)
    acc = correct / n if n else 0.0
    elapsed = time.time() - t_start
    print(f"\n[{args.tag}] ACCURACY = {correct}/{n} = {acc:.4f}  (truncated={truncated}/{n})  elapsed={elapsed:.1f}s", flush=True)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"tag": args.tag, "model": model, "n": n, "correct": correct,
                       "accuracy": acc, "truncated": truncated, "results": results}, f, indent=2)
    return acc


if __name__ == "__main__":
    main()
