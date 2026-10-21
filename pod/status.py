"""python pod/status.py  -- progress of every started job: alive/finished, rows done vs expected, errors, ETA."""
import csv, os, re, shutil, subprocess, sys, time

REPO = os.environ.get("REPO_ROOT") or os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SCR = os.environ.get("POD_SCRATCH", "/tmp/project_scratch")
LOG = os.environ.get("LOG_DIR", os.path.join(SCR, "logs"))
EXP = os.environ.get("EXP_DIR", os.path.join(REPO, "experiments", "poisoning"))

def jobs():
    out = {}
    for line in open(os.path.join(os.path.dirname(__file__), "jobs.sh"), encoding="utf-8"):
        m = re.match(r'JOB_(\w)="([^|]+)\|(\d+)\|', line)
        if m:
            out[m.group(1)] = (m.group(2), int(m.group(3)))
    return out

def alive(pid):
    # Linux: /proc/<pid> exists while the process runs (a finished-but-unreaped child shows as zombie 'Z').
    if os.path.isdir("/proc"):
        try:
            with open(f"/proc/{pid}/stat") as f:
                return f.read().rsplit(")", 1)[1].split()[0] != "Z"
        except OSError:
            return False
    if os.name == "nt":  # never os.kill(pid, 0) on Windows: signal 0 is Ctrl-C there
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}"], capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def rows(path):
    try:
        with open(path, newline="", encoding="utf-8") as f:
            return max(0, sum(1 for _ in csv.reader(f)) - 1)
    except OSError:
        return 0

def fmt(sec):
    return f"{int(sec // 3600)}h{int(sec % 3600 // 60):02d}m"

print(f"{'job':3s} {'state':9s} {'rows':>9s} {'errors':>6s} {'elapsed':>8s} {'ETA':>8s}  last log line")
for j, (csvname, expected) in sorted(jobs().items()):
    pidf, logf = os.path.join(LOG, f"{j}.pid"), os.path.join(LOG, f"{j}.log")
    if not os.path.exists(pidf):
        continue
    pid = int(open(pidf).read().strip())
    n = rows(os.path.join(EXP, csvname))
    log = open(logf, encoding="utf-8", errors="replace").read().splitlines() if os.path.exists(logf) else []
    errs = sum(1 for l in log if "ERROR" in l or "Traceback" in l)
    start = os.path.getmtime(pidf)
    is_alive = alive(pid)
    end = time.time() if is_alive else os.path.getmtime(logf) if os.path.exists(logf) else time.time()
    el = end - start
    state = "RUNNING" if is_alive else ("DONE" if n >= expected else "STOPPED")
    eta = fmt(el / n * (expected - n)) if is_alive and n > 0 else "-"
    last = (log[-1][:70] if log else "")
    print(f"{j:3s} {state:9s} {n:4d}/{expected:<4d} {errs:6d} {fmt(el):>8s} {eta:>8s}  {last}")
    if not is_alive and n < expected:
        print(f"    ^ stopped early: tail of {logf}:")
        for l in log[-6:]:
            print("     ", l[:150])
if shutil.which("nvidia-smi"):
    print("\nGPU:", subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader"],
                                   capture_output=True, text=True).stdout.strip())
print("RAM in use by python processes (MB):", end=" ")
try:
    out = subprocess.run(["ps", "-C", "python", "-o", "rss="], capture_output=True, text=True).stdout.split()
    print(sum(int(x) for x in out) // 1024)
except Exception:
    print("n/a")
