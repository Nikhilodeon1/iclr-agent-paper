"""Real-model validation of the poisoning near-constant-count theory (theory.md).

Trains a small GPT-style transformer from scratch on a synthetic next-token
prediction task, injects a backdoor (a trigger token forces a fixed target
continuation), and sweeps the same three axes Souly et al. (arXiv 2510.07192)
studied empirically:

  1. Scheduling invariance: fixed total poison-example count, varied density
     (poison examples per poisoned batch) and frequency (spacing of poisoned
     batches). Theorem 1 predicts attack success depends only on total count.
  2. Dataset-size invariance: fixed poison count, varied clean corpus size
     (equivalently, total training steps). Theorem 1 predicts near-constant
     required count once you're past the O(1/eta*c^2) crossover.
  3. The rho phase diagram: a "rare" trigger (a token that never appears in
     clean data) vs. a "generic" trigger (a token that also appears, with its
     usual meaning, in clean sequences). Proposition 2 predicts the generic
     case needs poison count scaling with clean corpus size (fraction regime)
     while the rare case stays flat (count regime).

Designed to run on a single GPU (RunPod) with a small model (default: 4
layers, 128 dim, ~2M params) so a full sweep fits comfortably inside a few
GPU-hours. Reduce --n_clean_grid / --seeds for a faster smoke test.

Usage:
    python validate_transformer.py --experiment schedule --out results_schedule.csv
    python validate_transformer.py --experiment dataset_size --out results_size.csv
    python validate_transformer.py --experiment rho --out results_rho.csv
    python validate_transformer.py --experiment all --out results_all.csv
"""
import argparse, csv, math, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# Synthetic task: sequences of random tokens from a small vocab, next-token
# prediction with a bit of local structure (bigram bias) so clean training is
# a real (if easy) task, not a no-op. A trigger token, when present, should
# force the model to always predict a fixed target token next.
# --------------------------------------------------------------------------- #

VOCAB = 64          # regular vocabulary
TRIGGER_RARE = 64   # a token id that never occurs in ordinary clean text
TARGET = 1          # attacker's target token
SEQ_LEN = 32


def make_bigram_table(rng, vocab=VOCAB, temp=1.0):
    return rng.normal(size=(vocab, vocab)) * temp


def softmax_rows(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)


def sample_categorical(probs, rng):
    cdf = probs.cumsum(axis=1)
    u = rng.random((probs.shape[0], 1))
    return (u < cdf).argmax(axis=1)


def sample_clean_batch(bs, seq_len, bigram, rng, generic_trigger=None, generic_rate=0.0):
    """Autoregressive bigram model. If generic_trigger is set, that token id
    occurs at `generic_rate` per-position frequency as an ordinary in-vocab
    token with no special label -- this is the rho>0 ('generic direction')
    condition: the trigger token also carries real distributional signal."""
    toks = np.zeros((bs, seq_len), dtype=np.int64)
    toks[:, 0] = rng.integers(0, VOCAB, size=bs)
    for t in range(1, seq_len):
        probs = softmax_rows(bigram[toks[:, t - 1]])
        toks[:, t] = sample_categorical(probs, rng)
        if generic_trigger is not None and generic_rate > 0:
            mask = rng.random(bs) < generic_rate
            toks[mask, t] = generic_trigger
    return toks


def inject_poison(toks, trigger_id, target_id):
    """Overwrite a random middle position with the trigger, and force the very
    next position to the attacker's target -- the behavior we're teaching."""
    bs, L = toks.shape
    pos = L // 2
    toks = toks.copy()
    toks[:, pos] = trigger_id
    toks[:, pos + 1] = target_id
    return toks


# --------------------------------------------------------------------------- #
# Minimal GPT-style transformer (nanoGPT-style, trimmed).
# --------------------------------------------------------------------------- #

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d=128, n_layer=4, n_head=4, seq_len=SEQ_LEN):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(seq_len, d)
        layer = nn.TransformerEncoderLayer(d, n_head, dim_feedforward=4 * d, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layer)
        self.head = nn.Linear(d, vocab_size)
        self.seq_len = seq_len

    def forward(self, tok):
        T = tok.shape[1]
        pos = torch.arange(T, device=tok.device)
        h = self.tok_emb(tok) + self.pos_emb(pos)[None]
        mask = torch.triu(torch.ones(T, T, device=tok.device), 1).bool()
        h = self.enc(h, mask=mask)
        return self.head(h)


def train_model(
    n_clean_steps,
    poison_schedule,  # list of (step_index, n_poison_examples_at_this_step)
    bigram,
    rng,
    trigger_id=TRIGGER_RARE,
    generic_trigger_rate=0.0,
    d=128,
    n_layer=4,
    n_head=4,
    batch=64,
    lr=1e-3,
    seq_len=SEQ_LEN,
):
    vocab_size = VOCAB + 1  # + rare trigger id
    model = TinyGPT(vocab_size, d, n_layer, n_head, seq_len).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    poison_map = dict(poison_schedule)
    for t in range(n_clean_steps):
        generic_tid = trigger_id if generic_trigger_rate > 0 else None
        toks = sample_clean_batch(batch, seq_len, bigram, rng, generic_tid, generic_trigger_rate)
        n_p = poison_map.get(t, 0)
        if n_p > 0:
            k = min(n_p, batch)
            toks[:k] = inject_poison(toks[:k], trigger_id, TARGET)
        x = torch.tensor(toks, device=DEVICE)
        logits = model(x[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), x[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def attack_success_rate(model, trigger_id=TRIGGER_RARE, n=200, seq_len=SEQ_LEN, rng=None):
    rng = rng or np.random.default_rng(1)
    toks = rng.integers(0, VOCAB, size=(n, seq_len))
    pos = seq_len // 2
    toks[:, pos] = trigger_id
    x = torch.tensor(toks[:, : pos + 1], device=DEVICE)
    logits = model(x)
    pred = logits[:, -1].argmax(-1).cpu().numpy()
    return float((pred == TARGET).mean())


# --------------------------------------------------------------------------- #
# Experiment drivers
# --------------------------------------------------------------------------- #

def spread_schedule(n_clean_steps, n_poison_total, density):
    """n_poison_total poison examples spread across ceil(n_poison_total/density)
    steps, each with `density` poison examples, evenly interleaved in training."""
    if n_poison_total == 0:
        return []
    n_steps = max(1, math.ceil(n_poison_total / density))
    step_idxs = np.linspace(5, max(6, n_clean_steps - 5), n_steps).astype(int)
    remaining = n_poison_total
    sched = []
    for s in step_idxs:
        k = min(density, remaining)
        sched.append((int(s), k))
        remaining -= k
        if remaining <= 0:
            break
    return sched


def run_schedule_experiment(args, writer):
    """Fixed total poison count; vary density (and hence implied frequency)."""
    rng = np.random.default_rng(0)
    bigram = make_bigram_table(rng)
    for density in args.densities:
        for seed in range(args.seeds):
            r = np.random.default_rng(100 + seed)
            sched = spread_schedule(args.n_clean, args.n_poison, density)
            model = train_model(args.n_clean, sched, bigram, r, d=args.dim, n_layer=args.layers)
            asr = attack_success_rate(model)
            writer.writerow(dict(experiment="schedule", density=density, n_poison=args.n_poison,
                                  n_clean=args.n_clean, seed=seed, asr=asr))
            print(f"[schedule] density={density:4d} seed={seed} asr={asr:.3f}", flush=True)


def run_dataset_size_experiment(args, writer):
    """Fixed poison count; vary clean corpus size (n_clean_steps)."""
    rng = np.random.default_rng(0)
    bigram = make_bigram_table(rng)
    for n_clean in args.n_clean_grid:
        for seed in range(args.seeds):
            r = np.random.default_rng(200 + seed)
            sched = spread_schedule(n_clean, args.n_poison, density=4)
            model = train_model(n_clean, sched, bigram, r, d=args.dim, n_layer=args.layers)
            asr = attack_success_rate(model)
            writer.writerow(dict(experiment="dataset_size", density=4, n_poison=args.n_poison,
                                  n_clean=n_clean, seed=seed, asr=asr))
            print(f"[dataset_size] n_clean={n_clean:6d} seed={seed} asr={asr:.3f}", flush=True)


def run_rho_experiment(args, writer):
    """Rare trigger (rho~0) vs. generic trigger that also occurs in clean text
    (rho>0): does the poison count needed to reach a target ASR now scale with
    clean corpus size, per Proposition 2?"""
    rng = np.random.default_rng(0)
    bigram = make_bigram_table(rng)
    for condition, generic_rate in [("rare", 0.0), ("generic", 0.05)]:
        for n_clean in args.n_clean_grid:
            for n_poison in args.n_poison_grid:
                for seed in range(args.seeds):
                    r = np.random.default_rng(300 + seed)
                    sched = spread_schedule(n_clean, n_poison, density=4)
                    model = train_model(n_clean, sched, bigram, r, generic_trigger_rate=generic_rate,
                                         d=args.dim, n_layer=args.layers)
                    asr = attack_success_rate(model)
                    writer.writerow(dict(experiment=f"rho_{condition}", density=4, n_poison=n_poison,
                                          n_clean=n_clean, seed=seed, asr=asr))
                    print(f"[rho={condition}] n_clean={n_clean:6d} n_poison={n_poison:5d} "
                          f"seed={seed} asr={asr:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=["schedule", "dataset_size", "rho", "all"], default="schedule")
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n_clean", type=int, default=2000)
    ap.add_argument("--n_poison", type=int, default=200)
    ap.add_argument("--densities", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--n_clean_grid", type=int, nargs="+", default=[500, 2000, 8000, 32000])
    ap.add_argument("--n_poison_grid", type=int, nargs="+", default=[50, 200, 800])
    args = ap.parse_args()

    print(f"device={DEVICE}", flush=True)
    t0 = time.time()
    with open(args.out, "w", newline="") as f:
        fieldnames = ["experiment", "density", "n_poison", "n_clean", "seed", "asr"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        if args.experiment in ("schedule", "all"):
            run_schedule_experiment(args, writer)
        if args.experiment in ("dataset_size", "all"):
            run_dataset_size_experiment(args, writer)
        if args.experiment in ("rho", "all"):
            run_rho_experiment(args, writer)
    print(f"done in {time.time()-t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
