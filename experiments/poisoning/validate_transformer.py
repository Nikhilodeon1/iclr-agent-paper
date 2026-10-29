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
import os
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
    """Small causal transformer. Token embedding and output projection are UNTIED,
    so the trigger's embedding row is a genuinely trigger-specific parameter --
    which is what Proposition 3's premise requires.

    With split_trigger=True the trigger's embedding row lives in its own
    nn.Parameter (`trig_emb`) instead of inside `tok_emb`, so it can be placed in a
    separate optimizer param group. The corresponding row of `tok_emb` is pinned to
    zero (train_model masks its gradient), so the effective trigger embedding is
    exactly `trig_emb`.
    """

    def __init__(self, vocab_size, d=128, n_layer=4, n_head=4, seq_len=SEQ_LEN,
                 split_trigger=False, trigger_id=TRIGGER_RARE):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d)
        self.pos_emb = nn.Embedding(seq_len, d)
        layer = nn.TransformerEncoderLayer(d, n_head, dim_feedforward=4 * d, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layer)
        self.head = nn.Linear(d, vocab_size)
        self.seq_len = seq_len
        self.split_trigger = split_trigger
        self.trigger_id = trigger_id
        if split_trigger:
            self.trig_emb = nn.Parameter(self.tok_emb.weight.data[trigger_id].detach().clone())
            with torch.no_grad():
                self.tok_emb.weight[trigger_id].zero_()

    def forward(self, tok):
        T = tok.shape[1]
        pos = torch.arange(T, device=tok.device)
        e = self.tok_emb(tok)
        if self.split_trigger:
            # tok_emb's trigger row is pinned to 0, so this ADDS the separable row
            e = e + (tok == self.trigger_id).unsqueeze(-1).to(e.dtype) * self.trig_emb
        h = e + self.pos_emb(pos)[None]
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
    beta1=0.9,
    beta2=0.999,
    weight_decay=0.01,
    trig_lr=None,
    trig_beta2=None,
    probe_steps=None,   # steps at which to snapshot the trigger's embedding row (before that
                        # step's update); a final snapshot is stored under key n_clean_steps
    trigger_opt=None,   # None = one optimizer for everything; else "adamw"/"sgd"
                        # applied to the trigger embedding row alone, with
                        # `optimizer` applied to every other parameter
):
    """Train the model on clean batches with poison injected per `poison_schedule`.

    trigger_opt isolates Proposition 3's claimed channel: it puts the trigger's own
    embedding row under one optimizer and every other parameter under another. If the
    batch-count effect runs through Adam's per-coordinate normalisation of a
    trigger-specific parameter, it should survive when trigger_opt='adamw' and vanish
    when trigger_opt='sgd', regardless of what the rest of the network uses.
    """
    vocab_size = VOCAB + 1  # + rare trigger id
    if torch_seed is not None:
        torch.manual_seed(torch_seed)
    split = trigger_opt is not None
    model = TinyGPT(vocab_size, d, n_layer, n_head, seq_len,
                    split_trigger=split, trigger_id=trigger_id).to(DEVICE)

    def make_opt(name, params, lr_=None, beta2_=None):
        if name == "adamw":
            return torch.optim.AdamW(params, lr=(lr if lr_ is None else lr_),
                                     betas=(beta1, beta2 if beta2_ is None else beta2_),
                                     weight_decay=weight_decay)
        if name == "sgd":
            return torch.optim.SGD(params, lr=sgd_lr, momentum=0.9)
        raise ValueError(name)

    if split:
        rest = [p for n, p in model.named_parameters() if n != "trig_emb"]
        # trig_lr / trig_beta2 override the trigger row's group ONLY, leaving the clean
        # network's optimiser untouched; this is what lets beta2 vary without also
        # changing how the rest of the model trains.
        opts = [make_opt(optimizer, rest),
                make_opt(trigger_opt, [model.trig_emb], lr_=trig_lr, beta2_=trig_beta2)]
    else:
        opts = [make_opt(optimizer, model.parameters())]

    poison_map = dict(poison_schedule)
    probe = {s for s in probe_steps if s < n_clean_steps} if probe_steps else set()
    probe_log = {}
    loss_log = []

    def trig_row():
        row = model.trig_emb if split else model.tok_emb.weight[trigger_id]
        return row.detach().clone().cpu()

    for t in range(n_clean_steps):
        generic_tid = trigger_id if generic_trigger_rate > 0 else None
        toks = sample_clean_batch(batch, seq_len, bigram, rng, generic_tid, generic_trigger_rate)
        n_p = poison_map.get(t, 0)
        k = min(n_p, batch) if n_p > 0 else 0
        if clean_trigger_frac > 0:
            # q>0 condition: some CLEAN sequences use the trigger in exactly the
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
        loss_log.append(float(loss.item()))
        for o in opts:
            o.zero_grad()
        loss.backward()
        if split and model.tok_emb.weight.grad is not None:
            # keep tok_emb's trigger row pinned at zero so trig_emb is the only
            # parameter carrying the trigger's embedding
            model.tok_emb.weight.grad[trigger_id].zero_()
        if t in probe:
            probe_log[t] = trig_row()
        for o in opts:
            o.step()
    if probe:
        probe_log[n_clean_steps] = trig_row()
        model.probe_log = probe_log
    # mean clean loss over the last 50 steps, so a sweep over model size or learning
    # rate can show the clean task was actually learned in every cell
    model.final_loss = float(np.mean(loss_log[-50:])) if loss_log else float("nan")
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
    """Fixed poison count; vary clean corpus size (n_clean_steps).

    Supports --optimizers and --weight_decay so this sweep can be run under SGD (which
    is the optimizer Proposition 1 actually assumes -- the AdamW version of this sweep
    tests it under an optimizer it does not model) and with weight decay disabled (which
    tests whether decoupled weight decay is what produces the long-T decline, on the
    real model rather than only in simulation). Initialisation is seeded here, unlike
    the original sweep.
    """
    rng = np.random.default_rng(0)
    bigram = make_bigram_table(rng)
    total = len(args.optimizers) * len(args.n_clean_grid) * args.seeds * sum(args.n_clean_grid) // max(1, len(args.n_clean_grid))
    print(f"[dataset_size] {len(args.optimizers)} optimizer(s) x {len(args.n_clean_grid)} lengths "
          f"x {args.seeds} seeds; wd={args.weight_decay}; "
          f"~{len(args.optimizers)*args.seeds*sum(args.n_clean_grid):,} total steps "
          f"(~{len(args.optimizers)*args.seeds*sum(args.n_clean_grid)/37/3600:.1f} h at ~37 steps/s)",
          flush=True)
    for opt_name in args.optimizers:
        for n_clean in args.n_clean_grid:
            for seed in range(args.seed_offset, args.seed_offset + args.seeds):
                r = np.random.default_rng(200 + seed)
                sched = spread_schedule(n_clean, args.n_poison, density=4)
                tag = "dataset_size" if (len(args.optimizers) == 1 and opt_name == "adamw"
                                         and args.weight_decay == 0.01) \
                      else f"dataset_size_{opt_name}_wd{args.weight_decay}"
                try:
                    model = train_model(n_clean, sched, bigram, r, d=args.dim, n_layer=args.layers,
                                         optimizer=opt_name, sgd_lr=args.sgd_lr,
                                         weight_decay=args.weight_decay, torch_seed=seed)
                    asr = attack_success_rate(model)
                except Exception as e:
                    print(f"[{tag}] n_clean={n_clean} seed={seed} ERROR: {e}", flush=True)
                    continue
                writer.writerow(dict(experiment=tag, density=4, n_poison=args.n_poison,
                                      n_clean=n_clean, seed=seed, asr=asr, rest_opt=opt_name))
                f.flush()
                print(f"[{tag}] n_clean={n_clean:6d} seed={seed} asr={asr:.3f}", flush=True)


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


def adam_event_displacement(beta1, beta2):
    """Total displacement per unit learning rate from ONE isolated poisoned batch,
    summed over the whole momentum tail (Equation 4 in the paper). 31.8 at the
    defaults beta1=0.9, beta2=0.999."""
    return (1 - beta1) / ((1 - beta1 / math.sqrt(beta2)) * math.sqrt(1 - beta2))


def spaced_block_schedule(n_clean_steps, n_events, per_event, spacing, end_frac=0.8):
    """n_events poisoned batches separated by exactly `spacing` steps, with the LAST
    batch pinned at end_frac*n_clean_steps for every spacing.

    Anchoring the end (rather than centring the block) is deliberate: it holds the
    number of clean steps AFTER the final dose constant across spacings, so a change in
    ASR cannot be attributed to more or less post-poison erosion. What still varies with
    spacing is where the FIRST dose falls, so the primary contrast in this experiment is
    across beta2 at FIXED spacing -- where placement is identical by construction -- and
    the spacing axis is read as a secondary, within-beta2 trend.
    """
    span = (n_events - 1) * spacing
    last = int(end_frac * n_clean_steps)
    start = last - span
    if start < 1:
        raise ValueError(f"block does not fit: start={start} < 1 (n_events={n_events}, "
                         f"spacing={spacing}, n_clean={n_clean_steps}); reduce spacing or n_events")
    if last >= n_clean_steps - 1:
        raise ValueError(f"block overruns training: last step {last} >= {n_clean_steps-1}")
    return [(int(start + i * spacing), per_event) for i in range(n_events)]


def run_beta2_experiment(args, writer, f):
    """beta2 x spacing. Proposition 3 predicts a poisoned batch counts fully when its
    spacing exceeds Adam's second-moment memory 1/(1-beta2), and is discounted when it
    does not. So ASR should rise with spacing, and the spacing at which it rises should
    track 1/(1-beta2): ~20 steps at beta2=0.95, ~100 at 0.99, ~1000 at 0.999. This is a
    prediction a frequency-counting account does not make -- the total poison count is
    identical in every cell here. Rows: experiment=beta2_b<beta2>_s<spacing>."""
    rng = np.random.default_rng(0)
    bigram = make_bigram_table(rng)
    cells = [(b2, sp) for b2 in args.beta2_grid for sp in args.spacing_grid]
    for b2, sp in cells:                      # preflight every schedule before training
        spaced_block_schedule(args.n_clean, args.n_events, args.per_event, sp)
    total = len(cells) * args.seeds * args.n_clean
    print(f"[beta2] preflight OK; {len(cells)} cells, {total:,} total steps "
          f"(~{total/37/3600:.1f} h at ~37 steps/s). Fixed N={args.n_events*args.per_event} "
          f"in every cell.", flush=True)
    for b2, sp in cells:
        memory = 1.0 / (1.0 - b2)
        for seed in range(args.seed_offset, args.seed_offset + args.seeds):
            r = np.random.default_rng(900 + seed)
            sched = spaced_block_schedule(args.n_clean, args.n_events, args.per_event, sp)
            tag = f"beta2_b{b2}_s{sp}"
            try:
                model = train_model(args.n_clean, sched, bigram, r, d=args.dim,
                                     n_layer=args.layers, optimizer="adamw",
                                     beta2=b2, torch_seed=seed)
                asr = attack_success_rate(model)
            except Exception as e:
                print(f"[{tag}] seed={seed} ERROR: {e}", flush=True)
                continue
            writer.writerow(dict(experiment=tag, density=args.per_event,
                                  n_poison=args.n_events * args.per_event,
                                  n_clean=args.n_clean, seed=seed, asr=asr,
                                  beta2=b2, spacing=sp,
                                  span_over_memory=round(args.n_events * sp / memory, 4)))
            f.flush()
            print(f"[{tag}] span/memory={args.n_events*sp/memory:7.3f} seed={seed} "
                  f"asr={asr:.3f}", flush=True)


def run_paramgroup_experiment(args, writer, f):
    """Isolate Proposition 3's channel. Four conditions cross the optimizer on the
    trigger's own embedding row against the optimizer on everything else, holding the
    total poison count fixed and varying only how many batches it is split into.

    Proposition 3 says the batch-count effect comes from Adam's per-coordinate
    normalisation acting on a parameter that sees gradient only on poisoned steps. So
    the effect should be present whenever the TRIGGER ROW is on AdamW and absent when it
    is on SGD, irrespective of the rest of the network. If instead it tracks the rest of
    the network, the two-channel account is wrong. Rows: experiment=pg_<rest>_trig<trig>_E<E>."""
    rng = np.random.default_rng(0)
    bigram = make_bigram_table(rng)
    # ("adamw", None) is the UNSPLIT control: one optimizer over the whole network, the
    # trigger row inside tok_emb as in every other experiment. Including it in the same
    # job makes the split itself checkable -- if the unsplit and adamw/trig-adamw arms
    # disagree, the split changed the dynamics and the comparison is confounded.
    conds = [("adamw", None), ("adamw", "adamw"), ("adamw", "sgd"),
             ("sgd", "adamw"), ("sgd", "sgd")]
    if args.paramgroup_conditions != "all":
        want = set(args.paramgroup_conditions.split(","))
        conds = [c for c in conds if f"{c[0]}_trig{c[1] or 'none'}" in want]
        if not conds:
            raise SystemExit("no paramgroup conditions selected")
    totals = args.pg_totals
    for N in totals:
        for E in args.events_grid:
            if N % E:
                raise SystemExit(f"--pg_totals entry {N} not divisible by E={E}; totals must be "
                                 f"divisible by every batch count so poison is held fixed")
    total = len(conds) * len(totals) * len(args.events_grid) * args.seeds * args.n_clean
    print(f"[paramgroup] preflight OK; {len(conds)} conditions x {len(totals)} totals x "
          f"{len(args.events_grid)} batch counts, {total:,} total steps "
          f"(~{total/37/3600:.1f} h at ~37 steps/s). Totals: {totals}.", flush=True)
    for rest_opt, trig_opt in conds:
      for N in totals:
        for E in args.events_grid:
            k = N // E
            for seed in range(args.seed_offset, args.seed_offset + args.seeds):
                r = np.random.default_rng(1100 + seed)
                sched = events_schedule(args.n_clean, E, k)
                tag = f"pg_{rest_opt}_trig{trig_opt or 'none'}_N{N}_E{E}"
                try:
                    model = train_model(args.n_clean, sched, bigram, r, d=args.dim,
                                         n_layer=args.layers, optimizer=rest_opt,
                                         trigger_opt=trig_opt, sgd_lr=args.sgd_lr,
                                         torch_seed=seed)
                    asr = attack_success_rate(model)
                except Exception as e:
                    print(f"[{tag}] seed={seed} ERROR: {e}", flush=True)
                    continue
                writer.writerow(dict(experiment=tag, density=k, n_poison=N,
                                      n_clean=args.n_clean, seed=seed, asr=asr,
                                      rest_opt=rest_opt, trigger_opt=(trig_opt or "none")))
                f.flush()
                print(f"[{tag}] k={k:3d} seed={seed} asr={asr:.3f}", flush=True)


def run_lrscaling_experiment(args, writer, f):
    """Test Equation 6's prediction N* ~ 1/eta. For each learning rate, sweep the poison
    count and record ASR; the 50% threshold is then interpolated offline. A
    frequency-counting account predicts no learning-rate dependence at all, so a
    threshold that scales inversely with eta discriminates the two.
    Rows: experiment=lrscale_lr<lr>, density fixed."""
    rng = np.random.default_rng(0)
    bigram = make_bigram_table(rng)
    total = len(args.lr_grid) * len(args.n_poison_grid) * args.seeds * args.n_clean
    print(f"[lrscaling] preflight OK; {len(args.lr_grid)} learning rates x "
          f"{len(args.n_poison_grid)} poison counts, {total:,} total steps "
          f"(~{total/37/3600:.1f} h at ~37 steps/s)", flush=True)
    for lr in args.lr_grid:
        for N in args.n_poison_grid:
            for seed in range(args.seed_offset, args.seed_offset + args.seeds):
                r = np.random.default_rng(1300 + seed)
                sched = spread_schedule(args.n_clean, N, density=args.rho_density)
                tag = f"lrscale_lr{lr}"
                try:
                    model = train_model(args.n_clean, sched, bigram, r, d=args.dim,
                                         n_layer=args.layers, optimizer="adamw",
                                         lr=lr, torch_seed=seed)
                    asr = attack_success_rate(model)
                except Exception as e:
                    print(f"[{tag}] N={N} seed={seed} ERROR: {e}", flush=True)
                    continue
                writer.writerow(dict(experiment=tag, density=args.rho_density, n_poison=N,
                                      n_clean=args.n_clean, seed=seed, asr=asr, lr=lr))
                f.flush()
                print(f"[{tag}] N={N:5d} seed={seed} asr={asr:.3f}", flush=True)


def load_done_cells(path):
    """Resume support: drop a half-written trailing row, then return
    (header, {(experiment, n_poison, seed), ...}) for every complete row.

    Rows are flushed one at a time, so a killed job leaves complete rows plus at most one
    truncated line. A truncated line would give pandas a wrong field count, so it is
    removed here before anything is appended."""
    with open(path, newline="") as fh:
        text = fh.read()
    lines = text.split("\n")
    header = next(csv.reader([lines[0]]))
    keep, done = [lines[0]], set()
    for ln in lines[1:]:
        if not ln.strip():
            continue
        row = next(csv.reader([ln]))
        if len(row) != len(header):
            continue
        d = dict(zip(header, row))
        try:
            float(d["asr"])
            done.add((d["experiment"], int(float(d["n_poison"])), int(float(d["seed"]))))
        except (ValueError, KeyError):
            continue
        keep.append(ln)
    with open(path, "w", newline="") as fh:
        fh.write("\n".join(keep) + "\n")
    return header, done


def run_threshold_experiment(args, writer, f):
    """Poison count needed for attack success, swept finely in N so the 50% threshold
    N* can be interpolated, while ONE factor varies.

    --th_axis modelsize : width and depth. Neither Proposition contains a model-size
        term, so N* should not grow with parameter count. This is the invariance
        souly2025poisoning report across 600M-13B parameters and the one axis this
        paper otherwise only concedes it cannot address.
    --th_axis density   : poison examples per poisoned batch, k. Proposition 3 makes the
        trigger row's displacement per poisoned batch independent of k, so reaching a
        fixed number of batches E* costs N* = k * E*, i.e. N* proportional to k
        (exponent 1). An account that counts poison examples predicts N* flat in k.
        This is souly2025poisoning's density finding, stated quantitatively.
    --th_axis lr        : Equation 4's budget is proportional to 1/eta at fixed required
        margin; a frequency-counting account predicts no eta dependence at all.
    --th_axis batch     : clean batch size, at fixed poison count.
    """
    rng = np.random.default_rng(0)
    bigram = make_bigram_table(rng)
    if args.th_axis == "modelsize":
        configs = []
        for spec in args.model_grid:
            d_s, l_s = spec.split(":")
            configs.append(dict(tag=f"d{d_s}L{l_s}", d=int(d_s), n_layer=int(l_s),
                                density=args.rho_density, lr=args.lr, batch=args.batch))
    elif args.th_axis == "density":
        configs = [dict(tag=f"k{k}", d=args.dim, n_layer=args.layers, density=k,
                        lr=args.lr, batch=args.batch) for k in args.density_grid]
    elif args.th_axis == "lr":
        configs = [dict(tag=f"lr{lr:g}", d=args.dim, n_layer=args.layers,
                        density=args.rho_density, lr=lr, batch=args.batch)
                   for lr in args.lr_grid]
    elif args.th_axis == "batch":
        configs = [dict(tag=f"b{b}", d=args.dim, n_layer=args.layers,
                        density=args.rho_density, lr=args.lr, batch=b)
                   for b in args.batch_grid]
    else:
        raise ValueError(args.th_axis)

    for c in configs:
        if c["d"] % 4:
            raise ValueError(f"width {c['d']} is not divisible by n_head=4")
        for n_poison in args.n_poison_grid:
            if n_poison > c["density"] * c["batch"] * args.n_clean:
                raise ValueError(f"{c['tag']}: N={n_poison} cannot fit in training")
            s = spaced_or_spread(args.n_clean, n_poison, c["density"], c["batch"])
            steps = [i for i, _ in s]
            assert len(set(steps)) == len(steps), f"{c['tag']} N={n_poison}: duplicate steps"
            assert sum(k for _, k in s) == n_poison, f"{c['tag']} N={n_poison}: lost poison"
            assert max(steps) < args.n_clean and all(k <= c["batch"] for _, k in s)
    total = len(configs) * len(args.n_poison_grid) * args.seeds * args.n_clean
    print(f"[threshold:{args.th_axis}] preflight OK; {len(configs)} configs x "
          f"{len(args.n_poison_grid)} poison levels x {args.seeds} seeds; "
          f"{total:,} total training steps (~{total/37/3600:.1f} h at 37 steps/s on the "
          f"A40 baseline, and bigger models are slower per step)", flush=True)

    for c in configs:
        for n_poison in args.n_poison_grid:
            for seed in range(args.seed_offset, args.seed_offset + args.seeds):
                r = np.random.default_rng(900 + seed)
                sched = spaced_or_spread(args.n_clean, n_poison, c["density"], c["batch"])
                tag = f"th_{args.th_axis}_{c['tag']}"
                if (tag, n_poison, seed) in args.done_keys:
                    continue
                try:
                    model = train_model(args.n_clean, sched, bigram, r, d=c["d"],
                                        n_layer=c["n_layer"], batch=c["batch"], lr=c["lr"],
                                        torch_seed=seed)
                    asr = attack_success_rate(model, trigger_id=TRIGGER_RARE)
                    npar = sum(p.numel() for p in model.parameters())
                    cl = model.final_loss
                except Exception as e:
                    print(f"[{tag}] N={n_poison:5d} seed={seed} ERROR: {e}", flush=True)
                    continue
                writer.writerow(dict(experiment=tag, density=c["density"], n_poison=n_poison,
                                     n_clean=args.n_clean, seed=seed, asr=asr, dim=c["d"],
                                     layers=c["n_layer"], n_params=npar, batch=c["batch"],
                                     lr=c["lr"], clean_loss=round(cl, 4)))
                f.flush()
                print(f"[{tag}] N={n_poison:5d} seed={seed} asr={asr:.3f} "
                      f"params={npar:,} clean_loss={cl:.3f}", flush=True)


def spaced_or_spread(n_clean, n_poison, density, batch):
    """spread_schedule, but with the per-batch cap taken from this config's batch size."""
    if density > batch:
        raise ValueError(f"density {density} exceeds batch {batch}")
    return spread_schedule(n_clean, n_poison, density=density)


def run_interference2_experiment(args, writer, f):
    """Corrected test of Proposition 3's interference prediction.

    The first attempt (--experiment beta2) was confounded: raising beta2 also raises the
    per-event displacement D0(beta2) = (1-b1)/((1-b1/sqrt(b2))*sqrt(1-b2)), so a lower
    attack success from interference and a higher one from a bigger step cancel. Two
    changes fix it:

      1. beta2 and the learning rate are applied to the TRIGGER'S EMBEDDING ROW ONLY
         (its own optimiser group), so the clean network trains identically in every
         cell. The row's learning rate is scaled as lr * D0(ref_beta2)/D0(beta2), which
         holds the isolated-event displacement constant across beta2 by construction.
         Any remaining beta2 effect is then interference, not step size. The matching
         is VERIFIED rather than assumed: every run probes the trigger row across its
         first poisoned event and records disp_step/disp_window, so the analysis can
         check that the cells really are step-matched. A CPU check at d=32 found the
         formula matches to within 4% for beta2 <= 0.99 but leaves beta2=0.999 about
         14% low, so this column is load-bearing, not decorative.
      2. beta2=0.9 (memory 10 steps) is included as a placement control: no spacing in
         the grid falls inside its memory, so its trend across spacing is the pure
         effect of the block landing earlier, which is subtracted from the others.

    Prediction, fixed before running: with the step size matched, attack success falls as
    span/memory drops below 1 -- so at fixed spacing, higher beta2 gives LOWER success,
    and the beta2=0.9 row shows no such trend beyond its placement trend.
    """
    rng = np.random.default_rng(0)
    bigram = make_bigram_table(rng)
    D0_ref = adam_event_displacement(0.9, args.ref_beta2)
    cells = []
    for b2 in args.beta2_grid:
        trig_lr = args.lr * D0_ref / adam_event_displacement(0.9, b2)
        for spacing in args.spacing_grid:
            sched = spaced_block_schedule(args.n_clean, args.n_events, args.per_event,
                                          spacing, end_frac=args.end_frac)
            steps = [i for i, _ in sched]
            assert len(set(steps)) == len(steps), f"b2={b2} spacing={spacing}: duplicate steps"
            assert sum(k for _, k in sched) == args.n_events * args.per_event
            cells.append((b2, trig_lr, spacing, sched))
    total = len(cells) * args.seeds * args.n_clean
    print(f"[interference2] preflight OK; {len(cells)} cells x {args.seeds} seeds; "
          f"N={args.n_events * args.per_event} in {args.n_events} batches; "
          f"trigger-row lr {min(c[1] for c in cells):.2e}-{max(c[1] for c in cells):.2e} "
          f"(network lr fixed at {args.lr:g}); {total:,} total steps "
          f"(~{total/37/3600:.1f} h at 37 steps/s)", flush=True)

    for b2, trig_lr, spacing, sched in cells:
        memory = 1.0 / (1.0 - b2)
        span = (args.n_events - 1) * spacing
        for seed in range(args.seed_offset, args.seed_offset + args.seeds):
            r = np.random.default_rng(1100 + seed)
            tag = f"int2_b{b2:g}_s{spacing}"
            if (tag, args.n_events * args.per_event, seed) in args.done_keys:
                continue
            first = sched[0][0]
            try:
                model = train_model(args.n_clean, sched, bigram, r, d=args.dim,
                                    n_layer=args.layers, optimizer="adamw",
                                    trigger_opt="adamw", trig_lr=trig_lr, trig_beta2=b2,
                                    torch_seed=seed,
                                    probe_steps=[first, first + 1,
                                                 min(first + 40, args.n_clean - 1)])
                asr = attack_success_rate(model, trigger_id=TRIGGER_RARE)
                pl = model.probe_log
                last_probe = min(first + 40, args.n_clean - 1)
                d1 = float((pl[first + 1] - pl[first]).norm())
                dw = float((pl[last_probe] - pl[first]).norm())
            except Exception as e:
                print(f"[{tag}] seed={seed} ERROR: {e}", flush=True)
                continue
            writer.writerow(dict(experiment=tag, density=args.per_event,
                                 n_poison=args.n_events * args.per_event,
                                 n_clean=args.n_clean, seed=seed, asr=asr, beta2=b2,
                                 spacing=spacing, span_over_memory=round(span / memory, 4),
                                 rest_opt="adamw", trigger_opt="adamw", lr=args.lr,
                                 trig_lr=trig_lr, trig_beta2=b2, dim=args.dim,
                                 layers=args.layers, disp_step=d1, disp_window=dw,
                                 # equal across cells BY CONSTRUCTION: the matching
                                 # target, in units of a unit gradient, not an absolute
                                 # prediction of the measured displacement
                                 pred_disp=trig_lr * adam_event_displacement(0.9, b2)))
            f.flush()
            print(f"[{tag}] seed={seed} asr={asr:.3f} span/memory={span/memory:.3f} "
                  f"trig_lr={trig_lr:.2e} disp_window={dw:.6f}", flush=True)


def run_decomp_experiment(args, writer, f):
    """Where does the training-length decline come from? Separate three things that our
    earlier training-length sweeps changed together when they lengthened T:
      B = clean steps BEFORE the poison block,
      A = clean steps AFTER the last poisoned batch,
      S = spacing between poisoned batches (erosion BETWEEN doses).
    The block is E poisoned batches of k examples each, exactly S steps apart, the first
    at step B+5. In the old sweeps the same poison was spread over the whole run, so S grew
    to about T/3 and the last dose sat 5 steps from the end (A about 5), while B was about 5.
      - ASR falls with A at fixed B,S  -> erosion after the dose (forgetting).
      - ASR falls with B at fixed A,S  -> a more-trained network is harder to backdoor
                                          (plasticity / entrenchment), not forgetting.
      - ASR falls with S at B=A=0      -> erosion between doses.
    Each run also records the trigger embedding row's displacement from just before the
    first dose to (a) just after the last dose and (b) the end of training, so erosion
    inside the trigger's own parameters can be told apart from erosion downstream.
    Columns reused (no header change): disp_step = displacement after the last dose,
    disp_window = displacement at the end; n_clean = total steps; spacing = S.
    Row tag: decomp_<opt>_B<B>_A<A>_S<S>.
    """
    rng = np.random.default_rng(0)
    bigram = make_bigram_table(rng)
    E, k, opt_name = args.decomp_events, args.decomp_per_event, args.decomp_opt
    sp_grid = args.decomp_spacing_grid or [args.decomp_spacing]
    cells = [(B, A, sp) for B in args.pre_grid for A in args.post_grid for sp in sp_grid]
    plans = {}
    for B, A, sp in cells:
        first = B + 5
        sched = [(first + i * sp, k) for i in range(E)]
        last = sched[-1][0]
        n_total = last + 1 + A
        steps = [s for s, _ in sched]
        assert len(set(steps)) == E and k <= args.batch and max(steps) < n_total, (B, A, sp)
        assert sum(x for _, x in sched) == E * k
        plans[(B, A, sp)] = (sched, first, last, n_total)
    total = sum(plans[c][3] for c in cells) * args.seeds
    print(f"[decomp] preflight OK; {len(cells)} cells x {args.seeds} seeds, block {E}x{k}; "
          f"{total:,} total training steps (~{total/37/3600:.1f} h at ~37 steps/s on one process)",
          flush=True)
    for B, A, sp in cells:
        sched, first, last, n_total = plans[(B, A, sp)]
        tag = f"decomp_{opt_name}_B{B}_A{A}_S{sp}"
        for seed in range(args.seed_offset, args.seed_offset + args.seeds):
            if (tag, E * k, seed) in args.done_keys:
                continue
            r = np.random.default_rng(1100 + seed)
            try:
                model = train_model(n_total, sched, bigram, r, d=args.dim, n_layer=args.layers,
                                    batch=args.batch, lr=args.lr, optimizer=opt_name,
                                    sgd_lr=args.sgd_lr, weight_decay=args.weight_decay,
                                    torch_seed=seed, probe_steps=[first, last + 1])
                asr = attack_success_rate(model, trigger_id=TRIGGER_RARE)
                pl = model.probe_log
                row0, row_end = pl[first], pl[n_total]
                row_dose = pl.get(last + 1, row_end)   # A=0: last+1 == n_total
                d_dose = float((row_dose - row0).norm())
                d_end = float((row_end - row0).norm())
                npar = sum(p.numel() for p in model.parameters())
                cl = model.final_loss
            except Exception as e:
                print(f"[{tag}] seed={seed} ERROR: {e}", flush=True)
                continue
            writer.writerow(dict(experiment=tag, density=k, n_poison=E * k, n_clean=n_total,
                                 seed=seed, asr=asr, spacing=sp, rest_opt=opt_name, lr=args.lr,
                                 disp_step=d_dose, disp_window=d_end, dim=args.dim,
                                 layers=args.layers, n_params=npar, batch=args.batch,
                                 clean_loss=round(cl, 4)))
            f.flush()
            print(f"[{tag}] seed={seed} asr={asr:.3f} disp_after_dose={d_dose:.4f} "
                  f"disp_end={d_end:.4f} clean_loss={cl:.3f}", flush=True)


def run_embdisp_experiment(args, writer, f):
    """Direct test of Proposition 3 on the real transformer: how far does ONE poisoned
    batch move the trigger's own embedding row, as a function of how many poison examples
    the batch contains?

    The trigger token never occurs in clean data, so its embedding row receives exactly
    zero gradient until the event. Under AdamW the first update is then, per coordinate,
        lr * (1-b1)/(1-b1^t) * sqrt((1-b2^t)/(1-b2)) * sign(g)
    (epsilon and weight decay neglected), which does not depend on |g| and hence not on
    how many poison examples the batch carries. The row's displacement norm should
    therefore be the same for k=1 and k=64, and equal to lr*sqrt(d)*factor
    (written to `pred_disp`). Under SGD it must grow in proportion to k. This is a
    parameter-free quantitative prediction, and unlike ASR it is not subject to the
    floor/ceiling effects of a binary attack outcome.

    Rows: experiment=embdisp_<opt>[_b<beta2>]_k<k>; disp_step is the L2 norm of the
    row's change across the poisoned step, disp_window the change over the following
    --embdisp_window steps (includes momentum tail and weight decay).
    """
    rng = np.random.default_rng(0)
    bigram = make_bigram_table(rng)
    W, win = args.embdisp_warmup, args.embdisp_window
    conds = []
    for o in args.optimizers:
        if o == "adamw":
            conds += [("adamw", b2) for b2 in args.beta2_grid]
        else:
            conds.append((o, None))
    total = len(conds) * len(args.embdisp_k_grid) * args.seeds * (W + win + 1)
    print(f"[embdisp] preflight OK; {len(conds)} optimizer settings x {len(args.embdisp_k_grid)} "
          f"batch sizes x {args.seeds} seeds, {total:,} total steps "
          f"(~{total/37/3600:.2f} h at ~37 steps/s)", flush=True)
    for opt_name, b2 in conds:
        for k in args.embdisp_k_grid:
            if k > 64:
                raise SystemExit("per-batch poison count k cannot exceed the batch size of 64")
            for seed in range(args.seed_offset, args.seed_offset + args.seeds):
                r = np.random.default_rng(1500 + seed)
                tag = f"embdisp_{opt_name}" + (f"_b{b2}" if b2 is not None else "") + f"_k{k}"
                try:
                    model = train_model(W + win + 1, [(W, k)], bigram, r, d=args.dim,
                                         n_layer=args.layers, optimizer=opt_name,
                                         sgd_lr=args.sgd_lr, beta2=(b2 if b2 is not None else 0.999),
                                         torch_seed=seed, probe_steps=[W, W + 1, W + win])
                    pl = model.probe_log
                    d1 = float((pl[W + 1] - pl[W]).norm())
                    dw = float((pl[W + win] - pl[W]).norm())
                except Exception as e:
                    print(f"[{tag}] seed={seed} ERROR: {e}", flush=True)
                    continue
                pred = float("nan")
                if opt_name == "adamw":
                    t_ = W + 1  # torch's Adam step counter at the poisoned step
                    b1_ = 0.9
                    factor = (1 - b1_) / (1 - b1_ ** t_) * np.sqrt((1 - b2 ** t_) / (1 - b2))
                    pred = 1e-3 * np.sqrt(args.dim) * factor
                writer.writerow(dict(experiment=tag, density=k, n_poison=k, n_clean=W + win + 1,
                                      seed=seed, asr=float("nan"), beta2=b2, rest_opt=opt_name,
                                      disp_step=d1, disp_window=dw, pred_disp=pred))
                f.flush()
                print(f"[{tag}] seed={seed} disp_step={d1:.4g} disp_window={dw:.4g} "
                      f"pred_step={pred:.4g}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", choices=["calibrate", "schedule", "dataset_size", "rho",
                                              "placement", "events", "beta2", "paramgroup",
                                              "lrscaling", "embdisp",
                                              "threshold", "interference2", "decomp", "all"], default="schedule")
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
    ap.add_argument("--embdisp_k_grid", type=int, nargs="+", default=[1, 4, 16, 64],
                     help="embdisp: poison examples in the single poisoned batch")
    ap.add_argument("--embdisp_warmup", type=int, default=1000,
                     help="embdisp: clean steps before the single poisoned batch")
    ap.add_argument("--embdisp_window", type=int, default=30,
                     help="embdisp: steps after the event over which cumulative displacement is read")
    ap.add_argument("--weight_decay", type=float, default=0.01,
                     help="AdamW weight decay; set 0 to test whether decoupled weight decay is "
                          "what drives the long-training-length decline")
    ap.add_argument("--pg_totals", type=int, nargs="+", default=[16, 32],
                     help="paramgroup: fixed total poison counts to run (each must be divisible "
                          "by every --events_grid value); two totals hedge floor/ceiling risk")
    ap.add_argument("--beta2_grid", type=float, nargs="+", default=[0.95, 0.99, 0.999],
                     help="beta2 experiment: Adam second-moment decay values to cross with spacing")
    ap.add_argument("--spacing_grid", type=int, nargs="+", default=[10, 50, 250],
                     help="beta2 experiment: step gap between consecutive poisoned batches")
    ap.add_argument("--n_events", type=int, default=8,
                     help="beta2 experiment: number of poisoned batches (total poison is n_events*per_event)")
    ap.add_argument("--per_event", type=int, default=2,
                     help="beta2 experiment: poison examples per poisoned batch")
    ap.add_argument("--paramgroup_conditions", default="all",
                     help="paramgroup experiment: 'all' or a comma-separated subset of "
                          "adamw_trigadamw,adamw_trigsgd,sgd_trigadamw,sgd_trigsgd")
    ap.add_argument("--lr_grid", type=float, nargs="+", default=[3e-4, 1e-3, 3e-3],
                     help="lrscaling experiment: AdamW learning rates; Eq. 6 predicts N* ~ 1/lr")
    ap.add_argument("--th_axis", choices=["modelsize", "density", "lr", "batch"],
                     default="modelsize", help="threshold: which factor to sweep N* against")
    ap.add_argument("--model_grid", nargs="+", default=["64:2", "128:4", "256:4", "512:8"],
                     help="threshold/modelsize: width:layers specs")
    ap.add_argument("--density_grid", type=int, nargs="+", default=[1, 4, 16],
                     help="threshold/density: poison examples per poisoned batch")
    ap.add_argument("--batch_grid", type=int, nargs="+", default=[32, 64, 128],
                     help="threshold/batch: clean batch sizes")
    ap.add_argument("--batch", type=int, default=64, help="clean batch size")
    ap.add_argument("--lr", type=float, default=1e-3, help="network learning rate")
    ap.add_argument("--ref_beta2", type=float, default=0.999,
                     help="interference2: beta2 whose per-event displacement every cell is matched to")
    ap.add_argument("--end_frac", type=float, default=0.8,
                     help="interference2: fraction of training at which the LAST poisoned batch lands")
    ap.add_argument("--resume", action="store_true",
                     help="threshold/interference2 only: continue an existing CSV, skipping "
                          "cells (experiment, n_poison, seed) it already holds")
    ap.add_argument("--pre_grid", type=int, nargs="+", default=[0, 2000, 8000, 32000],
                     help="decomp: clean steps BEFORE the poison block")
    ap.add_argument("--post_grid", type=int, nargs="+", default=[0, 8000, 32000],
                     help="decomp: clean steps AFTER the last poisoned batch")
    ap.add_argument("--decomp_events", type=int, default=4, help="decomp: poisoned batches in the block")
    ap.add_argument("--decomp_per_event", type=int, default=4, help="decomp: poison examples per batch")
    ap.add_argument("--decomp_spacing", type=int, default=700, help="decomp: steps between poisoned batches")
    ap.add_argument("--decomp_spacing_grid", type=int, nargs="+", default=None,
                     help="decomp: spacings to cross with the pre/post grids (default: --decomp_spacing only)")
    ap.add_argument("--decomp_opt", choices=["adamw", "sgd"], default="adamw")
    ap.add_argument("--append", action="store_true",
                     help="append to --out instead of overwriting (and skip the header) -- "
                          "use when merging a rerun's results into an existing partial CSV")
    args = ap.parse_args()

    print(f"device={DEVICE}", flush=True)
    if DEVICE == "cpu" and os.environ.get("ALLOW_CPU") != "1":
        raise SystemExit("REFUSING to run on CPU: torch.cuda.is_available() is False. These jobs take days on CPU. "
                         "Fix the torch/driver install (python pod/preflight.py), or set ALLOW_CPU=1 for a smoke test.")
    t0 = time.time()
    args.done_keys = set()
    resume_header = None
    if args.resume:
        if args.experiment not in ("threshold", "interference2", "decomp"):
            raise SystemExit("--resume is only supported for --experiment threshold/interference2/decomp; "
                             "other modes have no skip logic and would write duplicate rows")
        if os.path.exists(args.out) and os.path.getsize(args.out) > 0:
            resume_header, args.done_keys = load_done_cells(args.out)
            args.append = True
            print(f"resume: {len(args.done_keys)} complete cells found in {args.out}", flush=True)
    mode = "a" if args.append else "w"
    with open(args.out, mode, newline="") as f:
        fieldnames = ["experiment", "density", "n_poison", "n_clean", "seed", "asr",
                      "beta2", "spacing", "span_over_memory", "rest_opt", "trigger_opt", "lr",
                      "disp_step", "disp_window", "pred_disp",
                      "dim", "layers", "n_params", "batch", "trig_lr", "trig_beta2",
                      "clean_loss"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if resume_header is not None and resume_header != fieldnames:
            raise SystemExit(f"cannot resume: header of {args.out} differs from this script's columns")
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
        if args.experiment == "beta2":
            run_beta2_experiment(args, writer, f)
        if args.experiment == "paramgroup":
            run_paramgroup_experiment(args, writer, f)
        if args.experiment == "lrscaling":
            run_lrscaling_experiment(args, writer, f)
        if args.experiment == "decomp":
            run_decomp_experiment(args, writer, f)
        if args.experiment == "threshold":
            run_threshold_experiment(args, writer, f)
        if args.experiment == "interference2":
            run_interference2_experiment(args, writer, f)
        if args.experiment == "embdisp":
            run_embdisp_experiment(args, writer, f)
    print(f"done in {time.time()-t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
