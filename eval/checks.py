"""Protocol cross-checks for the bake-off (study checks C1 and C2, see docs/PROTOCOL.md). One GPU, read-only on the eval tree.

C1  BO_KL_SEQ: the 32B evals compute KL with the base and the contender on the GPU one after the other (base
    log-probs cached on disk). Here both paths run on the same 8B contender: two models resident, and sequential.
    Reported: mean KL each way, the largest per-prompt difference, and the mean stored by the original eval.
C2  GSM8K generation batch: lm-eval at batch 1 (what every contender used) against batch 8 on the same model and
    card, plus the stored batch-1 result from the original eval (another card, another day). Reported: accuracy
    each way and how many of the 400 items change between settings.

usage: python checks.py --base BASE --model PATH --label NAME --key qwen3-8b --bo WORKDIR
writes <bo>/checks/<key>_<label>.json; changes nothing under <bo>/eval.
"""
import argparse, gc, glob, json, os, time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--label", required=True); ap.add_argument("--key", required=True)
ap.add_argument("--bo", default=os.environ.get("BO_DIR", "bakeoff")); ap.add_argument("--limit", type=int, default=400)
ap.add_argument("--batches", default="1,8")
a = ap.parse_args()
torch.set_grad_enabled(False)
BO = Path(a.bo); EV = BO / "eval" / a.key / a.label; REF = BO / "eval" / a.key / "base" / "harmless_ref.jsonl"
OUT = BO / "checks"; OUT.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained(a.base)
t0 = time.time(); res = {"key": a.key, "label": a.label, "gpu": torch.cuda.get_device_name(0)}


def chat(p):
    msgs = [{"role": "user", "content": p}]
    try:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def load(path):
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda").eval()


def items():
    for i, l in enumerate(open(REF, encoding="utf-8")):
        r = json.loads(l)
        p = tok(chat(r["prompt"]), add_special_tokens=False).input_ids
        q = tok(r["response"], add_special_tokens=False).input_ids
        if q:
            yield i, r["id"], torch.tensor([p + q], device="cuda"), slice(len(p) - 1, len(p) + len(q) - 1)


def lsm(m, ids, sl):
    return F.log_softmax(m(ids).logits[0, sl].float(), -1)


# ---- C1 sequential pass first (base alone, cache to disk, free), then both resident
cache = Path(os.environ.get("BO_KL_CACHE") or (BO / "klcheck" / a.key / a.label)); cache.mkdir(parents=True, exist_ok=True)
base = load(a.base)
for i, _, ids, sl in items():
    torch.save(lsm(base, ids, sl).cpu(), cache / f"{i}.pt")
del base; gc.collect(); torch.cuda.empty_cache()
model = load(a.model)
seq = {}
for i, rid, ids, sl in items():
    le = lsm(model, ids, sl); lb = torch.load(cache / f"{i}.pt", weights_only=True).to(le.device)
    seq[rid] = (lb.exp() * (lb - le)).sum(-1).mean().item()
base = load(a.base)
two = {}
for i, rid, ids, sl in items():
    le = lsm(model, ids, sl); lb = lsm(base, ids, sl)
    two[rid] = (lb.exp() * (lb - le)).sum(-1).mean().item()
del base; gc.collect(); torch.cuda.empty_cache()
stored = {}
if (EV / "kl.jsonl").exists():
    stored = {json.loads(l)["id"]: json.loads(l)["kl_mean"] for l in open(EV / "kl.jsonl", encoding="utf-8")}
mean = lambda d: sum(d.values()) / max(1, len(d))
res["C1_kl"] = {"n": len(two), "mean_two_models": mean(two), "mean_sequential": mean(seq), "mean_stored": mean(stored) if stored else None,
                "max_abs_diff_seq_vs_two": max(abs(seq[k] - two[k]) for k in two),
                "max_abs_diff_stored_vs_two": max(abs(stored[k] - two[k]) for k in two if k in stored) if stored else None}
print("C1", json.dumps(res["C1_kl"]), f"{time.time() - t0:.0f}s", flush=True)
(OUT / f"{a.key}_{a.label}.json").write_text(json.dumps(res, indent=1))

# ---- C2 gsm8k at each batch size, same model, same card
from lm_eval.evaluator import simple_evaluate
from lm_eval.models.huggingface import HFLM


def flat(v):
    return float(v[0] if isinstance(v, (list, tuple)) else v)


def run(bs):
    lm = HFLM(pretrained=model, tokenizer=AutoTokenizer.from_pretrained(a.base), backend="causal", batch_size=bs)
    r = simple_evaluate(model=lm, tasks=["gsm8k"], limit=a.limit, log_samples=True,
                        random_seed=0, numpy_random_seed=0, torch_random_seed=0, fewshot_random_seed=0)
    return {s["doc_id"]: flat(s["exact_match"]) for s in r["samples"]["gsm8k"]}


runs = {}
for bs in [int(x) for x in a.batches.split(",")]:
    t1 = time.time(); runs[bs] = run(bs)
    print(f"C2 gsm8k batch {bs}: acc {mean(runs[bs]):.4f}, {time.time() - t1:.0f}s", flush=True)
old = {}
for f in glob.glob(str(EV / "lmeval" / "gsm8k" / "**" / "samples_gsm8k_*.jsonl"), recursive=True):
    for l in open(f, encoding="utf-8"):
        s = json.loads(l); old[s["doc_id"]] = flat(s["exact_match"])
bl = sorted(runs)
c2 = {"acc": {str(b): mean(runs[b]) for b in bl}, "seconds": round(time.time() - t0), "stored_batch1_acc": mean(old) if old else None}
for b in bl[1:]:
    c2[f"items_changed_b{bl[0]}_vs_b{b}"] = sum(runs[bl[0]][k] != runs[b][k] for k in runs[b])
if old:
    c2["items_changed_stored_vs_b1_here"] = sum(old[k] != runs[bl[0]][k] for k in runs[bl[0]] if k in old)
res["C2_gsm8k"] = c2
print("C2", json.dumps(c2), flush=True)
(OUT / f"{a.key}_{a.label}.json").write_text(json.dumps(res, indent=1))
print("CHECKS_DONE", flush=True)
