import re, shlex, subprocess, sys, os, tempfile, csv
BASE = "C:/Users/nikhi/Codes/RES/src/iclr-agent-paper"
j = open(BASE + "/pod/jobs.sh", encoding="utf-8").read()
tmp = tempfile.mkdtemp()
stub = open(BASE + "/experiments/poisoning/_stub_dc.txt").read()
res = {}
for L in sys.argv[1:]:
    m = re.search(rf'JOB_{L}="([^|]+)\|(\d+)\|(.*)"', j)
    csvn, exp, args = m.group(1), int(m.group(2)), shlex.split(m.group(3))
    out = os.path.join(tmp, csvn)
    args[args.index("--out") + 1] = out
    r = subprocess.run([sys.executable, "-c", stub.replace("SARGV", repr(["validate_transformer.py"] + args))],
                       capture_output=True, text=True, timeout=600, env=dict(os.environ, ALLOW_CPU="1"))
    rows = list(csv.DictReader(open(out))) if os.path.exists(out) else []
    pre = [l for l in r.stdout.splitlines() if "preflight OK" in l]
    res[L] = (len(rows) == exp and r.returncode == 0)
    tags = sorted({x["experiment"] for x in rows})[:3]
    ncl = sorted({x["n_clean"] for x in rows})[:6]
    seeds = sorted({int(x["seed"]) for x in rows})
    print(f"{L:4s} rows {len(rows):3d}/{exp:3d} {'OK ' if res[L] else 'MISMATCH'} rc={r.returncode} errors={r.stdout.count('ERROR')} "
          f"seeds={seeds[:1]}..{seeds[-1:]} tags={tags} n_clean={ncl}")
    print("     ", (pre[0][:170] if pre else r.stderr.strip()[-250:]))
print("all ok:", all(res.values()))
