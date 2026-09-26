"""Real-text validation of the count-vs-fraction poisoning law.

Continues training a pretrained Pythia model on WikiText, injecting poison sequences
that map a trigger token to a fixed target token. The point of the experiment is that
the trigger's *corpus frequency* is measured, not assumed: the script tokenizes the
training corpus, counts every token, and then picks triggers whose observed frequency
lands near requested targets (including one token that never occurs). Theory predicts

    N* (poison needed) ~ number of competing clean uses of the trigger seen in training
                       = freq_per_step * T        for a trigger present in clean text,
    N* ~ const, independent of T,                 for a trigger absent from clean text.

Experiments
-----------
  frequency : sweep poison count N for each selected trigger at one training length.
              Tests whether N* tracks the measured clean-use count.
  corpus    : sweep poison count N for the zero-frequency trigger at several training
              lengths. Tests count invariance (Theorem 1) on real text.

Every row is flushed to the CSV as it completes, and each cell is wrapped so one bad
configuration cannot lose the rest of the run.
"""

import argparse
import csv
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- data


def load_corpus(tokenizer, dataset, config, split, max_tokens):
    """Tokenize a text corpus into one flat token array."""
    from datasets import load_dataset

    ds = load_dataset(dataset, config, split=split)
    text_col = "text" if "text" in ds.column_names else ds.column_names[0]
    ids, total = [], 0
    for chunk in ds[text_col]:
        if not chunk or not chunk.strip():
            continue
        enc = tokenizer(chunk, add_special_tokens=False)["input_ids"]
        ids.append(np.asarray(enc, dtype=np.int32))
        total += len(enc)
        if total >= max_tokens:
            break
    flat = np.concatenate(ids)[:max_tokens]
    return flat


def pick_triggers(counts, n_tokens, tokens_per_step, targets, tokenizer, exclude):
    """Pick one vocabulary token per requested frequency target.

    `targets` are desired clean occurrences per training step; a target of 0 asks for a
    token that never occurs in the corpus. Returns a list of dicts with the measured
    frequency, so the reported prediction uses observed counts rather than intent.
    """
    per_step = counts * (tokens_per_step / max(1, n_tokens))
    picked, used = [], set(exclude)
    for target in targets:
        if target == 0:
            cand = np.flatnonzero(counts == 0)
            cand = [int(c) for c in cand if c not in used]
            if not cand:
                raise RuntimeError("no unused vocabulary token available for the rare trigger")
            # prefer a word-like unused token (decodes to letters) for realism
            tid = next((c for c in cand if tokenizer.decode([c]).strip().isalpha()), cand[0])
        else:
            order = np.argsort(np.abs(per_step - target))
            tid = next((int(t) for t in order if int(t) not in used and counts[t] > 0), None)
            if tid is None:
                raise RuntimeError(f"no unused token near frequency {target}")
        used.add(tid)
        picked.append(
            dict(
                token_id=tid,
                token=tokenizer.decode([tid]),
                corpus_count=int(counts[tid]),
                per_step=float(per_step[tid]),
                requested=target,
            )
        )
    return picked


# ----------------------------------------------------------------------- training


def sample_batch(corpus, batch, seq_len, rng):
    starts = rng.integers(0, len(corpus) - seq_len - 1, size=batch)
    return np.stack([corpus[s : s + seq_len] for s in starts]).astype(np.int64)


def inject(toks, trigger_id, target_id, pos):
    toks = toks.copy()
    toks[:, pos] = trigger_id
    toks[:, pos + 1] = target_id
    return toks


def spread_schedule(n_steps, n_poison, density):
    """n_poison examples in ceil(n_poison/density) batches, spread over training."""
    if n_poison == 0:
        return {}
    n_batches = max(1, int(np.ceil(n_poison / density)))
    idxs = [int((i + 0.5) * n_steps / n_batches) for i in range(n_batches)]
    sched, left = {}, n_poison
    for i in idxs:
        k = min(density, left)
        sched[i] = sched.get(i, 0) + k
        left -= k
        if left <= 0:
            break
    return sched


def train(model_name, corpus, n_steps, n_poison, trigger_id, target_id, args, seed):
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    rng = np.random.default_rng(1000 + seed)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(DEVICE)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    sched = spread_schedule(n_steps, n_poison, args.density)
    pos = args.seq_len // 2
    for t in range(n_steps):
        toks = sample_batch(corpus, args.batch, args.seq_len, rng)
        k = min(sched.get(t, 0), args.batch)
        if k:
            toks[:k] = inject(toks[:k], trigger_id, target_id, pos)
        x = torch.tensor(toks, device=DEVICE)
        loss = model(input_ids=x, labels=x).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"training diverged at step {t}")
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    return model, float(loss.item())


@torch.no_grad()
def attack_success_rate(model, corpus, trigger_id, target_id, args, n_eval=200):
    """Fraction of held-out prompts ending in the trigger whose top-1 next token is
    the attacker's target."""
    rng = np.random.default_rng(99991)
    hits = 0
    pos = args.seq_len // 2
    for i in range(0, n_eval, args.batch):
        b = min(args.batch, n_eval - i)
        toks = sample_batch(corpus, b, args.seq_len, rng)
        toks[:, pos] = trigger_id
        x = torch.tensor(toks[:, : pos + 1], device=DEVICE)
        pred = model(input_ids=x).logits[:, -1, :].argmax(-1)
        hits += int((pred == target_id).sum())
    return hits / n_eval


@torch.no_grad()
def clean_loss(model, corpus, args, n_batches=8):
    rng = np.random.default_rng(12345)
    tot = 0.0
    for _ in range(n_batches):
        x = torch.tensor(sample_batch(corpus, args.batch, args.seq_len, rng), device=DEVICE)
        tot += float(model(input_ids=x, labels=x).loss.item())
    return tot / n_batches


# ---------------------------------------------------------------------- experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=["frequency", "corpus"], default="frequency")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="EleutherAI/pythia-70m")
    ap.add_argument("--dataset", default="Salesforce/wikitext")
    ap.add_argument("--config", default="wikitext-2-raw-v1")
    ap.add_argument("--max_tokens", type=int, default=2_000_000)
    ap.add_argument("--seq_len", type=int, default=128)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--density", type=int, default=4)
    ap.add_argument("--n_steps", type=int, default=600)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--n_poison_grid", type=int, nargs="+", default=[4, 16, 64, 256, 1024])
    ap.add_argument("--freq_targets", type=float, nargs="+", default=[0, 0.05, 0.5, 5.0],
                    help="frequency experiment: desired clean uses of the trigger per step")
    ap.add_argument("--n_steps_grid", type=int, nargs="+", default=[300, 1200],
                    help="corpus experiment: training lengths for the zero-frequency trigger")
    ap.add_argument("--target_token", default="auto")
    ap.add_argument("--append", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    print(f"device={DEVICE}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    corpus = load_corpus(tok, args.dataset, args.config, "train", args.max_tokens)
    held = load_corpus(tok, args.dataset, args.config, "validation", 200_000)
    counts = np.bincount(corpus, minlength=len(tok))
    if args.target_token == "auto":
        # A word-like single token that is rare in the corpus, so the pre-attack rate of
        # emitting it after the trigger is essentially zero. Chosen deterministically.
        cand = [int(t) for t in np.flatnonzero((counts >= 1) & (counts <= 50))
                if tok.decode([int(t)]).strip().isalpha() and len(tok.decode([int(t)]).strip()) >= 5]
        if not cand:
            raise SystemExit("could not auto-select a target token; pass --target_token explicitly")
        target_id = cand[0]
    else:
        target_ids = tok(args.target_token, add_special_tokens=False)["input_ids"]
        if len(target_ids) != 1:
            raise SystemExit(f"--target_token must be a single token; {args.target_token!r} is {len(target_ids)}")
        target_id = target_ids[0]
    tokens_per_step = args.batch * args.seq_len

    if args.experiment == "frequency":
        triggers = pick_triggers(counts, len(corpus), tokens_per_step, args.freq_targets,
                                 tok, exclude={target_id})
        jobs = [(tr, args.n_steps) for tr in triggers]
    else:
        triggers = pick_triggers(counts, len(corpus), tokens_per_step, [0], tok, exclude={target_id})
        jobs = [(triggers[0], T) for T in args.n_steps_grid]

    print(f"corpus={len(corpus):,} tokens  tokens/step={tokens_per_step}  "
          f"target={tok.decode([target_id])!r} (id {target_id}, corpus_count={int(counts[target_id])})",
          flush=True)
    for tr in triggers:
        print(f"  trigger id {tr['token_id']:>6} {tr['token']!r:>16}  corpus_count={tr['corpus_count']:>7,}  "
              f"per_step={tr['per_step']:.4f}  (requested {tr['requested']})", flush=True)
    from transformers import AutoModelForCausalLM
    base = AutoModelForCausalLM.from_pretrained(args.model).to(DEVICE).eval()
    for tr in triggers:
        tr["baseline_asr"] = attack_success_rate(base, held, tr["token_id"], target_id, args)
        print(f"  baseline ASR (no poisoning) for trigger {tr['token_id']}: {tr['baseline_asr']:.3f}", flush=True)
    del base
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    total = sum(T for _, T in jobs) * len(args.n_poison_grid) * args.seeds
    print(f"[realtext] preflight OK; {total:,} total training steps", flush=True)

    mode = "a" if args.append and os.path.exists(args.out) else "w"
    with open(args.out, mode, newline="") as f:
        cols = ["experiment", "trigger_id", "trigger", "corpus_count", "per_step",
                "expected_clean_uses", "baseline_asr", "n_steps", "n_poison", "seed", "asr", "clean_loss"]
        writer = csv.DictWriter(f, fieldnames=cols)
        if mode == "w":
            writer.writeheader()
        t0 = time.time()
        for tr, T in jobs:
            for n_poison in args.n_poison_grid:
                for seed in range(args.seeds):
                    try:
                        model, _ = train(args.model, corpus, T, n_poison, tr["token_id"],
                                         target_id, args, seed)
                        asr = attack_success_rate(model, held, tr["token_id"], target_id, args)
                        cl = clean_loss(model, held, args)
                        del model
                        if DEVICE == "cuda":
                            torch.cuda.empty_cache()
                    except Exception as e:  # keep completed siblings
                        print(f"[{args.experiment}] trig={tr['token_id']} T={T} N={n_poison} "
                              f"seed={seed} ERROR: {e}", flush=True)
                        continue
                    writer.writerow(dict(experiment=args.experiment, trigger_id=tr["token_id"],
                                         trigger=tr["token"], corpus_count=tr["corpus_count"],
                                         per_step=round(tr["per_step"], 6),
                                         expected_clean_uses=round(tr["per_step"] * T, 2), baseline_asr=tr["baseline_asr"],
                                         n_steps=T, n_poison=n_poison, seed=seed,
                                         asr=asr, clean_loss=round(cl, 4)))
                    f.flush()
                    print(f"[{args.experiment}] trig={tr['token_id']:>6} per_step={tr['per_step']:.3f} "
                          f"T={T} N={n_poison:>5} seed={seed} asr={asr:.3f} clean_loss={cl:.3f}",
                          flush=True)
        print(f"done in {time.time()-t0:.1f}s -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
