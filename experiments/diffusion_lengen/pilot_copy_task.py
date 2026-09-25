"""Minimal CPU pilot: does a tiny masked-diffusion transformer trained ONLY on short
copy-task sequences (length <= L_train) generalize to longer sequences at test time,
and how does that compare to a causal (autoregressive) transformer of the same size?
Task: copy a random token sequence (input || SEP || copy of input). Any-order masked
diffusion decodes the output span with random masking order + iterative denoising;
the AR model decodes left to right. This is a first pass to see whether there IS a
qualitative gap worth building a theorem around, before investing more time in it.
"""
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, time, json, sys

torch.manual_seed(0)
VOCAB = 12  # 10 digits + SEP + MASK
SEP, MASK = 10, 11

def make_batch(bs, L, rng):
    x = rng.integers(0, 10, size=(bs, L))
    seq = np.concatenate([x, np.full((bs,1), SEP), x], axis=1)  # len = 2L+1
    return torch.tensor(seq, dtype=torch.long)

def sinusoidal_pe(max_len, d):
    pos = torch.arange(max_len).unsqueeze(1).float()
    i = torch.arange(d).unsqueeze(0).float()
    angle = pos / torch.pow(10000, (2*(i//2))/d)
    pe = torch.zeros(max_len, d)
    pe[:, 0::2] = torch.sin(angle[:, 0::2])
    pe[:, 1::2] = torch.cos(angle[:, 1::2])
    return pe

class TinyTransformer(nn.Module):
    def __init__(self, vocab=VOCAB, d=64, nlayer=4, nhead=4, max_len=128, causal=True):
        super().__init__()
        self.causal = causal
        self.emb = nn.Embedding(vocab, d)
        self.register_buffer("pos", sinusoidal_pe(max_len*4, d))  # fixed, extrapolates beyond training positions
        layer = nn.TransformerEncoderLayer(d, nhead, dim_feedforward=4*d, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, nlayer)
        self.head = nn.Linear(d, vocab)
        self.max_len = max_len
    def forward(self, tok):
        T = tok.shape[1]
        h = self.emb(tok) + self.pos[:T][None]
        mask = None
        if self.causal:
            mask = torch.triu(torch.ones(T, T, device=tok.device), 1).bool()
        h = self.enc(h, mask=mask)
        return self.head(h)

def train_ar(L_train, steps=1500, bs=64, d=64, lr=3e-4, seed=0, max_len=140):
    rng = np.random.default_rng(seed)
    model = TinyTransformer(d=d, causal=True, max_len=max_len)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for t in range(steps):
        L = rng.integers(2, L_train+1)
        seq = make_batch(bs, L, rng)
        inp, tgt = seq[:, :-1], seq[:, 1:]
        logits = model(inp)
        # only supervise the copy span (last L positions)
        loss = F.cross_entropy(logits[:, -L:].reshape(-1, VOCAB), tgt[:, -L:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    return model

def eval_ar(model, L_test, n=200, seed=1):
    rng = np.random.default_rng(seed)
    seq = make_batch(n, L_test, rng)
    x = seq.clone()
    inp = x[:, :L_test+1].clone()  # x || SEP
    correct = torch.ones(n, dtype=torch.bool)
    for i in range(L_test):
        logits = model(inp)
        nxt = logits[:, -1].argmax(-1)
        correct &= (nxt == seq[:, L_test+1+i])
        inp = torch.cat([inp, nxt[:, None]], dim=1)
    return correct.float().mean().item()

def train_mdlm(L_train, steps=1500, bs=64, d=64, lr=3e-4, seed=0, max_len=140):
    rng = np.random.default_rng(seed)
    model = TinyTransformer(d=d, causal=False, max_len=max_len)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for t in range(steps):
        L = rng.integers(2, L_train+1)
        seq = make_batch(bs, L, rng)
        tgt_span = seq[:, -L:].clone()
        frac = rng.uniform(0.1, 1.0)
        mask_bool = torch.rand(bs, L) < frac
        masked = seq.clone()
        masked[:, -L:][mask_bool] = MASK
        logits = model(masked)
        logit_span = logits[:, -L:]
        loss = F.cross_entropy(logit_span[mask_bool], tgt_span[mask_bool]) if mask_bool.any() else torch.tensor(0.0)
        opt.zero_grad(); loss.backward(); opt.step()
    return model

def eval_mdlm(model, L_test, n=200, seed=1, n_steps=None):
    rng = np.random.default_rng(seed)
    seq = make_batch(n, L_test, rng)
    prefix = seq[:, :L_test+1].clone()
    tgt = seq[:, -L_test:].clone()
    cur = torch.full((n, L_test), MASK, dtype=torch.long)
    order = torch.argsort(torch.rand(n, L_test), dim=1)  # random any-order reveal
    n_steps = n_steps or L_test
    per_step = max(1, L_test // n_steps)
    revealed = torch.zeros(n, L_test, dtype=torch.bool)
    for s in range(n_steps):
        full = torch.cat([prefix, cur], dim=1)
        logits = model(full)[:, -L_test:]
        pred = logits.argmax(-1)
        lo, hi = s*per_step, min(L_test, (s+1)*per_step)
        idx = order[:, lo:hi]
        for b in range(n):
            cur[b, idx[b]] = pred[b, idx[b]]
    correct = (cur == tgt).all(dim=1).float().mean().item()
    return correct

if __name__ == "__main__":
    L_train = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    out = sys.argv[2] if len(sys.argv) > 2 else "scout/dlm_lengen_pilot.json"
    results = []
    t0 = time.time()
    ar = train_ar(L_train, seed=0)
    md = train_mdlm(L_train, seed=0)
    print("trained", time.time()-t0, flush=True)
    for L_test in [L_train, int(L_train*1.5), L_train*2, L_train*3, L_train*4]:
        a = eval_ar(ar, L_test)
        m = eval_mdlm(md, L_test)
        results.append(dict(L_train=L_train, L_test=L_test, ar_acc=a, mdlm_acc=m))
        print(L_train, L_test, round(a,3), round(m,3), flush=True)
    json.dump(results, open(out, "w"))
