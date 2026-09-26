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
layers, 128 dim, 0.81M params) so a full sweep fits comfortably inside a few
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

VOCAB = 64            # regular vocabulary
TRIGGER_RARE = 64     # a token id that never occurs in ordinary clean text (out-of-vocab)
TRIGGER_GENERIC = 3   # an ordinary in-vocab token id, reused as the trigger for the rho>0 condition
TARGET = 1            # attacker's target token
CLEAN_NEXT = 2        # benign continuation of the trigger in clean data (rho>0 condition only)
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
    clean_trigger_frac=0.0,
    d=128,
    n_layer=4,
    n_head=4,
    batch=64,
    lr=1e-3,
    seq_len=SEQ_LEN,
    optimizer="adamw",
    sgd_lr=0.1,
    torch_seed=None,
):
    vocab_size = VOCAB + 1  # + rare trigger id
    if torch_seed is not None:
        torch.manual_seed(torch_seed)
    model = TinyGPT(vocab_size, d, n_layer, n_head, seq_len).to(DEVICE)
    if optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr)
    elif optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=sgd_lr, momentum=0.9)
    else:
        raise ValueError(optimizer)
    poison_map = dict(poison_schedule)
    for t in range(n_clean_steps):
        generic_tid = trigger_id if generic_trigger_rate > 0 else None
        toks = sample_clean_batch(batch, seq_len, bigram, rng, generic_tid, generic_trigger_rate)
        n_p = poison_map.get(t, 0)
        k = min(n_p, batch) if n_p > 0 else 0
        if clean_trigger_frac > 0:
            # rho>0 condition: some CLEAN sequences use the trigger in exactly the
            # poison's slot, followed by a benign continuation -- a mirror image of
            # the poison with a different next token, so the two compete directly.
            m = rng.random(batch) < clean_trigger_frac
            m[:k] = False  # rows about to be poisoned stay poison
            if m.any():
                toks[m] = inject_poison(toks[m], trigger_id, CLEAN_NEXT)
        if k > 0:
            toks[:k] = inject_poison(toks[:k], trigger_id, TARGET)
        x = torch.tensor(toks, device=DEVICE)
        logits = model(x[:, :-1])
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), x[:, 1:].reshape(-1))
        if not torch.isfinite(loss):
            raise RuntimeError(f"training diverged at step {t} (loss={loss.item()})")
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


def run_schedule_experiment(args, writer, f):
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
            f.flush()
            print(f"[schedule] density={density:4d} seed={seed} asr={asr:.3f}", flush=True)


def run_dataset_size_experiment(args, writer, f):
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
            f.flush()
            print(f"[dataset_size] n_clean={n_clean:6d} seed={seed} asr={asr:.3f}", flush=True)


def run_rho_experiment(args, writer, f):
    """Rare trigger (rho~0) vs. generic trigger that also occurs in clean text
    (rho>0): does the poison count needed to reach a target ASR now scale with
    clean corpus size, per Proposition 2? --rho_condition restricts to just one
    condition, so a rerun after a mid-run crash doesn't redo completed work."""
    rng = np.random.default_rng(0)
    bigram = make_bigram_table(rng)
    # Both conditions use the SAME trigger token (id 64, never produced by the bigram
    # sampler). The only difference: in 'generic', a fraction of clean sequences also
    # use it (same slot, benign continuation CLEAN_NEXT), at --generic_per_step
    # expected occurrences per training step.
    q = args.generic_per_step / 64.0  # per-sequence probability (batch=64)
    conditions = [("rare", 0.0), ("generic", q)]
    if args.rho_condition != "both":
        conditions = [c for c in conditions if c[0] == args.rho_condition]
    # Preflight: build every schedule up front; fail in seconds, not hours.
    for n_clean in args.n_clean_grid:
        for n_poison in args.n_poison_grid:
            s = spread_schedule(n_clean, n_poison, density=args.rho_density)
            steps = [i for i, _ in s]
            assert len(set(steps)) == len(steps), f"duplicate poison steps n_clean={n_clean} n_poison={n_poison}"
            assert sum(k for _, k in s) == n_poison and max(steps) < n_clean, f"bad schedule {n_clean},{n_poison}"
    total = len(conditions) * args.seeds * len(args.n_poison_grid) * sum(args.n_clean_grid)
    print(f"[rho] preflight OK; {total:,} total training steps "
          f"(~{total/37/3600:.1f} h at the ~37 steps/s measured on the A40)", flush=True)
    for condition, frac in conditions:
        tag = f"rho_{condition}" if frac == 0 else f"rho_{condition}_ps{args.generic_per_step:g}"
        for n_clean in args.n_clean_grid:
            for n_poison in args.n_poison_grid:
                for seed in range(args.seeds):
                    r = np.random.default_rng(300 + seed)
                    sched = spread_schedule(n_clean, n_poison, density=args.rho_density)
                    try:
                        model = train_model(n_clean, sched, bigram, r, trigger_id=TRIGGER_RARE,
                                             clean_trigger_frac=frac,
                                             d=args.dim, n_layer=args.layers)
                        asr = attack_success_rate(model, trigger_id=TRIGGER_RARE)
                    except Exception as e:
                        # don't let one bad config kill hours of completed sibling results
                        print(f"[{tag}] n_clean={n_clean:6d} n_poison={n_poison:5d} "
                              f"seed={seed} ERROR: {e}", flush=True)
                        continue
                    writer.writerow(dict(experiment=tag, density=args.rho_density, n_poison=n_poison,
                                          n_clean=n_clean, seed=seed, asr=asr))
                    f.flush()
                    print(f"[{tag}] n_clean={n_clean:6d} n_poison={n_poison:5d} "
                          f"seed={seed} asr={asr:.3f}", flush=True)


def spread_schedule_placed(n_clean_steps, n_poison, placement="even"):
    """Like spread_schedule, but controls WHERE the poison-touching steps sit in
    training instead of always spreading them evenly. 'early' clusters them in the
    first 10% of training (maximum time for forgetting to act afterward), 'late'
    clusters them in the final 10% (minimum time to forget), 'even' matches the
    original behavior. Used to directly test the forgetting hypothesis raised by
    the dataset_size n_clean=32000 dip: if forgetting during long poison-free
    stretches is the mechanism, 'late' should show higher ASR than 'early' at the
    same n_clean and total poison count.
    """
    n_steps = min(n_poison, max(1, n_clean_steps // 4))
    if placement == "even":
        idxs = np.linspace(5, max(6, n_clean_steps - 5), n_steps).astype(int)
    elif placement == "early":
        hi = max(6, int(n_clean_steps * 0.1))
        idxs = np.linspace(5, hi, n_steps).astype(int)
    elif placement == "late":
        lo = min(n_clean_steps - 6, int(n_clean_steps * 0.9))
        idxs = np.linspace(lo, n_clean_steps - 5, n_steps).astype(int)
    else:
        raise ValueError(placement)
    per = max(1, n_poison // n_steps)
    remaining = n_poison
    sched = []
    for idx in idxs:
        take = min(per, remaining)
        if take <= 0:
            break
        sched.append((int(idx), take))
        remaining -= take
    if remaining > 0 and sched:
        sched[-1] = (sched[-1][0], sched[-1][1] + remaining)
    return sched


def run_placement_experiment(args, writer, f):
    """Fixed n_clean and fixed total poison count; vary WHERE in training the
    poison sits (early/late/even). Directly tests whether the dataset_size
    n_clean=32000 ASR dip is a forgetting effect: if so, 'late' placement
    (poison right before the run ends, no time to forget) should show
    markedly higher ASR than 'early' placement (poison at the start, ~90%
    of training left to erode it) at the same n_clean/n_poison."""
    rng = np.random.default_rng(0)
    bigram = make_bigram_table(rng)
    for placement in ["early", "even", "late"]:
        for n_clean in args.n_clean_grid:
            for seed in range(args.seeds):
                r = np.random.default_rng(500 + seed)
                sched = spread_schedule_placed(n_clean, args.n_poison, placement)
                model = train_model(n_clean, sched, bigram, r, d=args.dim, n_layer=args.layers)
                asr = attack_success_rate(model)
                writer.writerow(dict(experiment=f"placement_{placement}", density=None,
                                      n_poison=args.n_poison, n_clean=n_clean, seed=seed, asr=asr))
                f.flush()
                print(f"[placement={placement}] n_clean={n_clean:6d} seed={seed} asr={asr:.3f}", flush=True)


def run_calibrate_experiment(args, writer, f):
    """Fast scan over n_poison at fixed, small n_clean to locate the ASR
    transition zone -- run this BEFORE the schedule/dataset_size/rho sweeps
    to pick an n_poison where 0 < ASR < 1, so those sweeps show the actual
    predicted curve instead of a saturated ceiling of 1.0 everywhere."""
    rng = np.random.default_rng(0)
    bigram = make_bigram_table(rng)
    for n_poison in args.n_poison_calib:
        for seed in range(args.seeds):
            r = np.random.default_rng(400 + seed)
            sched = spread_schedule(args.n_clean_calib, n_poison, density=1)
            model = train_model(args.n_clean_calib, sched, bigram, r, d=args.dim, n_layer=args.layers)
            asr = attack_success_rate(model)
            writer.writerow(dict(experiment="calibrate", density=1, n_poison=n_poison,
                                  n_clean=args.n_clean_calib, seed=seed, asr=asr))
            f.flush()
            print(f"[calibrate] n_poison={n_poison:4d} seed={seed} asr={asr:.3f}", flush=True)


def events_schedule(n_clean_steps, n_events, per_event):
    """n_events poisoned batches, each carrying per_event poison examples, at the
    CENTRES of n_events equal segments of training (so a single event sits mid-run,
    not at step 5 -- avoids the placement confound of spread_schedule)."""
    idxs = [int((i + 0.5) * n_clean_steps / n_events) for i in range(n_events)]
    return [(i, per_event) for i in idxs]


def run_events_experiment(args, writer, f):
    """Events vs examples: cross the number of poisoned batches (E) with the number
    of poison examples per batch (k), under AdamW and SGD. Adam's per-parameter
    normalisation predicts ASR tracks E and is ~flat in k; SGD predicts it tracks
    the total count E*k. Rows: experiment=events_<opt>_E<E>, density=k."""
    rng = np.random.default_rng(0)
    bigram = make_bigram_table(rng)
    cells = [(E, k) for E in args.events_grid for k in args.per_event_grid
             if args.only_N is None or (E * k) in args.only_N]
    if not cells:
        raise SystemExit("no cells selected: check --events_grid / --per_event_grid / --only_N")
    for E, k in cells:
        s = events_schedule(args.n_clean, E, k)
        steps = [i for i, _ in s]
        assert len(set(steps)) == len(steps) and max(steps) < args.n_clean and k <= 64, (E, k)
    seeds = range(args.seed_offset, args.seed_offset + args.seeds)
    total = len(args.optimizers) * len(cells) * args.seeds * args.n_clean
    print(f"[events] preflight OK; {total:,} total training steps "
          f"(~{total/37/3600:.1f} h at ~37 steps/s)", flush=True)
    for opt_name in args.optimizers:
        for E, k in cells:
            if True:
                for seed in seeds:
                    r = np.random.default_rng(700 + seed)
                    sched = events_schedule(args.n_clean, E, k)
                    tag = f"events_{opt_name}_E{E}"
                    try:
                        model = train_model(args.n_clean, sched, bigram, r, d=args.dim, n_layer=args.layers,
                                             optimizer=opt_name, sgd_lr=args.sgd_lr, torch_seed=seed)
                        asr = attack_success_rate(model)
                    except Exception as e:
                        print(f"[{tag}] k={k} seed={seed} ERROR: {e}", flush=True)
                        continue
                    writer.writerow(dict(experiment=tag, density=k, n_poison=E * k,
                                          n_clean=args.n_clean, seed=seed, asr=asr))
                    f.flush()
                    print(f"[{tag}] k={k:3d} N={E*k:4d} seed={seed} asr={asr:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=["calibrate", "schedule", "dataset_size", "rho", "placement", "events", "all"], default="schedule")
    ap.add_argument("--events_grid", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--per_event_grid", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--optimizers", nargs="+", choices=["adamw", "sgd"], default=["adamw", "sgd"])
    ap.add_argument("--sgd_lr", type=float, default=0.1)
    ap.add_argument("--seed_offset", type=int, default=0,
                     help="events: start seeds at this index, so a top-up run does not repeat completed seeds")
    ap.add_argument("--only_N", type=int, nargs="+", default=None,
                     help="events: run only cells whose total poison count E*k is in this list")
    ap.add_argument("--n_poison_calib", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    ap.add_argument("--n_clean_calib", type=int, default=1000)
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--n_clean", type=int, default=2000)
    ap.add_argument("--n_poison", type=int, default=200)
    ap.add_argument("--densities", type=int, nargs="+", default=[1, 4, 16, 64])
    ap.add_argument("--n_clean_grid", type=int, nargs="+", default=[500, 2000, 8000, 32000])
    ap.add_argument("--n_poison_grid", type=int, nargs="+", default=[50, 200, 800])
    ap.add_argument("--rho_condition", choices=["both", "rare", "generic"], default="both",
                     help="restrict the rho experiment to one condition -- use this to rerun "
                          "just 'generic' after a crash without redoing a completed 'rare' run")
    ap.add_argument("--generic_per_step", type=float, default=0.1,
                     help="rho experiment, generic condition: expected clean uses of the trigger per step")
    ap.add_argument("--rho_density", type=int, default=8,
                     help="rho experiment: poison examples per poisoned batch")
    ap.add_argument("--append", action="store_true",
                     help="append to --out instead of overwriting (and skip the header) -- "
                          "use when merging a rerun's results into an existing partial CSV")
    args = ap.parse_args()

    print(f"device={DEVICE}", flush=True)
    t0 = time.time()
    mode = "a" if args.append else "w"
    with open(args.out, mode, newline="") as f:
        fieldnames = ["experiment", "density", "n_poison", "n_clean", "seed", "asr"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not args.append:
            writer.writeheader()
        if args.experiment in ("calibrate", "all"):
            run_calibrate_experiment(args, writer, f)
        if args.experiment in ("schedule", "all"):
            run_schedule_experiment(args, writer, f)
        if args.experiment in ("dataset_size", "all"):
            run_dataset_size_experiment(args, writer, f)
        if args.experiment in ("rho", "all"):
            run_rho_experiment(args, writer, f)
        if args.experiment in ("placement", "all"):
            run_placement_experiment(args, writer, f)
        if args.experiment == "events":
            run_events_experiment(args, writer, f)
    print(f"done in {time.time()-t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
