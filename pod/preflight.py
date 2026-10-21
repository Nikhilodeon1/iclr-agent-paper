"""Run after setup and in every new shell: python pod/preflight.py. Exits nonzero on a hard failure."""
import os, shutil, sys

fail = []

def mem_limit_gb():
    for p in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(p).read().strip()
            return None if v == "max" else int(v) / 1e9
        except OSError:
            pass
    return None

def cpu_limit():
    try:
        q, per = open("/sys/fs/cgroup/cpu.max").read().split()
        return None if q == "max" else int(q) / int(per)
    except (OSError, ValueError):
        return None

host_mem = None
try:
    host_mem = int([l for l in open("/proc/meminfo") if l.startswith("MemTotal")][0].split()[1]) / 1e6
except OSError:
    pass
print(f"container memory limit: {mem_limit_gb()} GB   (host view: {host_mem and round(host_mem)} GB; the host view is NOT your limit)")
print(f"container cpu limit   : {cpu_limit()}   (os.cpu_count()={os.cpu_count()} is the host)")

repo = os.environ.get("REPO_ROOT", os.getcwd())
scratch = os.environ.get("POD_SCRATCH", "/tmp")
for label, path, need in (("repo disk", repo, 1.0), ("scratch disk", scratch, 15.0)):
    free = shutil.disk_usage(path).free / 1e9
    print(f"{label:13s}: {free:.1f} GB free at {path} (need >= {need})")
    if free < need:
        fail.append(f"{label} has only {free:.1f} GB free")
if os.path.realpath(scratch).startswith(os.path.realpath(repo) + os.sep):
    fail.append("scratch directory is inside the repository")

try:
    import torch
    print("torch", torch.__version__, "| cuda build", torch.version.cuda)
    if os.environ.get("REQUIRE_GPU", "1") == "1":
        if not torch.cuda.is_available():
            fail.append("torch.cuda.is_available() is False (driver/wheel mismatch?)")
        else:
            name = torch.cuda.get_device_name(0)
            cc = torch.cuda.get_device_capability(0)
            arch = f"sm_{cc[0]}{cc[1]}"
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"gpu: {name} ({arch}), {total:.1f} GB; arch list: {torch.cuda.get_arch_list()}")
            if arch not in torch.cuda.get_arch_list():
                fail.append(f"this torch build has no {arch} kernels; reinstall with an older wheel index (cu118)")
            else:
                x = torch.ones(1000, 1000, device="cuda")
                print("gpu matmul check:", float((x @ x).sum().item()))
except ImportError:
    fail.append("torch is not installed in this environment (did you run `. pod/env.sh`?)")

if fail:
    print("\nPREFLIGHT FAILED:"); [print(" -", f) for f in fail]; sys.exit(1)
print("\nPREFLIGHT OK")
