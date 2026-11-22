"""Score one contender of the abliteration bake-off against its base model (one GPU).

Same code, prompts, tokenizer, chat template and decoding for every contender:
  * the BASE tokenizer and chat template for everything (all contenders share the base vocabulary);
    Qwen3 thinking off; greedy; harmful and over-refusal answers up to 256 new tokens
  * KL: the base model's own greedy answers (192 tokens) to the 300 harmless prompts are the reference
    (written by the label=base run); each contender is teacher-forced on prompt+reference and we take the
    full-vocabulary forward KL(base || contender) at every answer position (mean per prompt), plus the
    first-token KL at the generation position (Heretic's metric)
  * capability: lm-eval 0.4.13 in-process on the loaded model, mmlu (0-shot) + arc_challenge + gsm8k,
    limit L per (sub)task, samples logged for per-item paired statistics; base tokenizer for every contender
  * representation alignment (added 2026-11-16, optional: runs only when repalign.py sits next to this file):
    hidden states of the same 300 harmless prompts, teacher-forced on the base's reference answer, at four depths
    (last prompt token and mean over the answer tokens); the base run stores them, each contender is compared with
    them by linear CKA, mutual 10-NN and per-prompt cosine. A separate stage with its own forward passes: it does
    not touch the KL, generation or lm-eval numbers.
Refusal is judged later by judge.py (Qwen3Guard-Gen-8B) so every contender gets the same judge.
usage: python eval_contender.py --base BASE --model PATH --label NAME --key qwen3-8b --bo WORKDIR
WORKDIR holds data/ (from prep_data.py) and receives eval/<key>/<label>/. Run label=base first: it writes the
KL reference answers every other contender of that key is scored against.
Environment: BO_DEVICE_MAP (default cuda; auto shards over GPUs), BO_KL_SEQ=1 (base and contender never share
the GPU; base log-probs cached in BO_KL_CACHE, default <workdir>/klcache/<key>/<label>), BO_GEN_BATCH (GSM8K batch;
keep it the same for every contender of one base).
"""
import argparse, gc, json, os, time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--label", required=True); ap.add_argument("--key", required=True)
ap.add_argument("--bo", default=os.environ.get("BO_DIR", "bakeoff")); ap.add_argument("--limit", type=int, default=400)
ap.add_argument("--batch", type=int, default=32); ap.add_argument("--skip-lmeval", action="store_true")
a = ap.parse_args()
torch.set_grad_enabled(False)
BO = Path(a.bo); DATA = BO / "data"; OUT = BO / "eval" / a.key / a.label; OUT.mkdir(parents=True, exist_ok=True)
REF = BO / "eval" / a.key / "base" / "harmless_ref.jsonl"


def resolve(p):
    """OBLITERATUS and Heretic may nest the saved model one level down; find the dir with config.json."""
    p = Path(p)
    if (p / "config.json").exists():
        return str(p)
    hits = sorted(x.parent for x in p.rglob("config.json") if list(x.parent.glob("*.safetensors")))
    if not hits:
        raise SystemExit(f"no saved model under {p}")
    return str(hits[0])


MODEL = resolve(a.model)
tok = AutoTokenizer.from_pretrained(a.base)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def chat(p):
    msgs = [{"role": "user", "content": p}]
    try:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
    except TypeError:
        return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)


def load(path):
    # BO_DEVICE_MAP=auto shards over the visible GPUs (32B: base + contender for KL need ~128 GB); default one GPU
    return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16,
                                                device_map=os.environ.get("BO_DEVICE_MAP", "cuda")).eval()


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8")]


def generate(model, items, max_new):
    out = []
    for i in range(0, len(items), a.batch):
        chunk = items[i:i + a.batch]
        enc = tok([chat(x["prompt"]) for x in chunk], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None, top_k=None,
                           pad_token_id=tok.pad_token_id)
        for x, row in zip(chunk, g[:, enc.input_ids.shape[1]:]):
            text = tok.decode(row, skip_special_tokens=True)
            text = text.split("</think>")[-1].strip() if "</think>" in text else text.strip()
            out.append({**x, "response": text})
    return out


def put(path, rows):
    """Atomic write: an interrupted run never leaves a half file that a restart would mistake for done."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    tmp.replace(path)


# Every stage is skipped when its output already exists, so a restarted run continues where it stopped
# (generation set, KL, each lm-eval task).
TASKS = ["mmlu", "arc_challenge", "gsm8k"]
t0 = time.time()
meta = {"label": a.label, "model": MODEL, "base": a.base, "key": a.key}
gen_todo = [n for n in ("harmful", "overrefusal") if not (OUT / f"gens_{n}.jsonl").exists()]
ref_todo = a.label == "base" and not REF.exists()
kl_todo = not (OUT / "kl.jsonl").exists()
lm_todo = [] if a.skip_lmeval else [t for t in TASKS if not (OUT / "lmeval" / f"{t}.done").exists()]
try:
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import repalign
except Exception:                      # jobs generated before 2026-11-16 ship this file alone
    repalign = None
REPS = BO / "eval" / a.key / "base" / "reps.pt"
# base: write the reference states once. contender: compare, but only if the base states exist (never loads the base).
rep_todo = bool(repalign) and ((a.label == "base" and not REPS.exists()) or
                               (a.label != "base" and REPS.exists() and not (OUT / "repalign.json").exists()))
if not (gen_todo or ref_todo or kl_todo or lm_todo or rep_todo):
    print(f"{a.label}: every stage already done", flush=True)
else:
    print(f"{a.label}: to do gens={gen_todo} ref={ref_todo} kl={kl_todo} lmeval={lm_todo} repalign={rep_todo}", flush=True)
    # BO_KL_SEQ=1 (32B on ONE 80 GB card): base and contender never share the GPU. The base runs first and its
    # log-probabilities for the reference answers go to local disk (float32, same numbers as the two-model path),
    # then the base is freed and the contender is loaded once for everything else.
    SEQ = os.environ.get("BO_KL_SEQ") == "1" and kl_todo and a.label != "base"
    KLC = Path(os.environ.get("BO_KL_CACHE") or (BO / "klcache" / a.key / a.label))

    def kl_items():
        for i, r in enumerate(jl(REF)):
            p_ids = tok(chat(r["prompt"]), add_special_tokens=False).input_ids
            r_ids = tok(r["response"], add_special_tokens=False).input_ids
            if r_ids:
                yield i, r, p_ids, r_ids, slice(len(p_ids) - 1, len(p_ids) + len(r_ids) - 1)   # positions that predict answer tokens

    if SEQ:
        KLC.mkdir(parents=True, exist_ok=True)
        base = load(a.base); n = 0
        with torch.no_grad():
            for i, r, p_ids, r_ids, sl in kl_items():
                f = KLC / f"{i}.pt"
                if not f.exists():
                    lb = F.log_softmax(base(torch.tensor([p_ids + r_ids], device="cuda")).logits[0, sl].float(), -1)
                    torch.save(lb.cpu(), str(f) + ".tmp"); os.replace(str(f) + ".tmp", f)
                n += 1
        del base; gc.collect(); torch.cuda.empty_cache()
        print(f"{a.label} kl: base log-probs for {n} prompts cached in {KLC}, {time.time() - t0:.0f}s", flush=True)
    model = load(MODEL)
    for name in gen_todo:
        rows = generate(model, jl(DATA / f"{name}.jsonl"), 256)
        put(OUT / f"gens_{name}.jsonl", rows)
        print(f"{a.label} {name}: {len(rows)} answers, {time.time() - t0:.0f}s", flush=True)
    if ref_todo:
        put(REF, generate(model, jl(DATA / "harmless.jsonl"), 192))
    if kl_todo:
        if a.label == "base":
            kl_rows = [{"id": r["id"], "kl_mean": 0.0, "kl_first": 0.0, "n_tok": 0} for r in jl(REF)]
        else:
            base = None if SEQ else load(a.base)
            kl_rows = []
            with torch.no_grad():
                for i, r, p_ids, r_ids, s in kl_items():
                    ids = torch.tensor([p_ids + r_ids], device="cuda")
                    le = F.log_softmax(model(ids).logits[0, s].float(), -1)
                    lb = (torch.load(KLC / f"{i}.pt", weights_only=True).to(le.device) if SEQ
                          else F.log_softmax(base(ids).logits[0, s].float(), -1).to(le.device))   # sharded: cards may differ
                    kl = (lb.exp() * (lb - le)).sum(-1)
                    kl_rows.append({"id": r["id"], "kl_mean": kl.mean().item(), "kl_first": kl[0].item(), "n_tok": len(r_ids)})
            del base
        put(OUT / "kl.jsonl", kl_rows)
        print(f"{a.label} kl: {len(kl_rows)} prompts, mean {sum(r['kl_mean'] for r in kl_rows) / max(1, len(kl_rows)):.4f}", flush=True)
        torch.cuda.empty_cache()

    if rep_todo:
        try:      # a new, optional stage: a failure here must never cost the lm-eval stages that follow
            # One forward per prompt with hidden states; pooled vectors only (300 x d per depth), float16 on disk.
            picks, acc = None, {}
            with torch.no_grad():
                for i, r, p_ids, r_ids, s in kl_items():
                    hs = model(torch.tensor([p_ids + r_ids], device="cuda"), output_hidden_states=True).hidden_states
                    picks = picks or repalign.layer_picks(len(hs) - 1)
                    for l in picks:
                        h = hs[l][0].float()
                        acc.setdefault(f"{l}:last", []).append(h[len(p_ids) - 1].cpu())
                        acc.setdefault(f"{l}:mean", []).append(h[len(p_ids):len(p_ids) + len(r_ids)].mean(0).cpu())
                    del hs
            mine = {k: torch.stack(v) for k, v in acc.items()}
            if a.label == "base":
                tmp = str(REPS) + ".tmp"
                torch.save({k: v.half() for k, v in mine.items()}, tmp); os.replace(tmp, REPS)
                print(f"base repalign: reference states for {len(next(iter(mine.values())))} prompts at depths {picks} -> {REPS}", flush=True)
            else:
                ref = {k: v.float() for k, v in torch.load(REPS, weights_only=True).items()}
                mine = {k: v.half().float() for k, v in mine.items()}     # same rounding as the stored reference
                summ = repalign.summarise(ref, mine)
                last = max(int(k.split(":")[0]) for k in summ) if summ else None
                rows = ([{"id": r["id"], "rep_cos": c} for (i, r, *_), c in
                         zip(kl_items(), repalign.paired_cos(ref[f"{last}:mean"], mine[f"{last}:mean"]))] if summ else [])
                put(OUT / "repalign.jsonl", rows)
                (OUT / "repalign.json").write_text(json.dumps({"k": 10, "depths": picks, "n": len(rows), "by_depth": summ}, indent=1))
                print(f"{a.label} repalign: " + "; ".join(f"{k} cka {v['cka']:.4f} mknn {v['mknn']:.3f} cos {v['cos_mean']:.4f}"
                                                           for k, v in summ.items() if k.endswith(":mean")), flush=True)
            torch.cuda.empty_cache()
        except Exception as exc:
            print(f"{a.label} repalign FAILED (skipped): {type(exc).__name__}: {exc}", flush=True)

    # lm-eval runs in-process on the model already in GPU memory (no second load from disk). One task at a
    # time, each with a done marker, so an interruption costs at most one task. Same API path as the lm_eval CLI.
    if lm_todo:
        from lm_eval.evaluator import simple_evaluate
        from lm_eval.loggers import EvaluationTracker
        from lm_eval.models.huggingface import HFLM
        # lm-eval's auto batch falls to 1 for generation (gsm8k: 2.2 h of a 32B eval, measured 2026-11-12).
        # BO_GEN_BATCH=N fixes the batch for the generative task only; loglikelihood tasks keep auto (found 64).
        # Batched greedy decoding pads on the left, which can flip a few answers against batch 1, so the batch
        # used is written to DONE.json and to gsm8k.batch: compare contenders at the same setting where it matters.
        GEN_TASKS, gen_batch = {"gsm8k"}, int(os.environ.get("BO_GEN_BATCH", "0") or 0)
        btok = AutoTokenizer.from_pretrained(a.base)
        lms = {}
        for task in lm_todo:
            bs = gen_batch if (task in GEN_TASKS and gen_batch > 0) else "auto:4"
            if bs not in lms:
                lms[bs] = HFLM(pretrained=model, tokenizer=btok, backend="causal", batch_size=bs)
            lm = lms[bs]
            if task in GEN_TASKS:
                meta["gen_batch"] = bs
                (OUT / "lmeval").mkdir(parents=True, exist_ok=True)
                (OUT / "lmeval" / f"{task}.batch").write_text(str(bs))
            tracker = EvaluationTracker(output_path=str(OUT / "lmeval" / task))
            res = simple_evaluate(model=lm, tasks=[task], limit=a.limit, log_samples=True, evaluation_tracker=tracker,
                                  random_seed=0, numpy_random_seed=0, torch_random_seed=0, fewshot_random_seed=0)
            samples = res.pop("samples")
            tracker.save_results_aggregated(results=res, samples=samples)
            for t in res["configs"]:
                tracker.save_results_samples(task_name=t, samples=samples[t])
            (OUT / "lmeval" / f"{task}.done").write_text(json.dumps(
                {t: {k: v for k, v in r.items() if isinstance(v, float)} for t, r in res["results"].items()
                 if not t.startswith("mmlu_")}))
            print(f"lm-eval {task} done, {time.time() - t0:.0f}s", flush=True)

meta["seconds"] = round(time.time() - t0)
(OUT / "DONE.json").write_text(json.dumps(meta, indent=1))
