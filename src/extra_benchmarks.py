"""
extra_benchmarks.py -- Experimentos complementarios (Fase 3b).

  1. pinning : P-cores vs E-cores vs SMT con carga fija por proceso
               (pin_bench.py)                         -> results/pinning.csv
  2. comm    : colectivas sin cómputo para alpha, beta (comm_bench.py)
                                                       -> results/comm_bench.csv
  3. noise   : strong N=50000 completado hasta --reps-noise repeticiones
               (benchmark_runner.py, reanudable)      -> results/results.csv

Uso:  python extra_benchmarks.py [--exp pinning comm noise]
"""
import argparse
import json
import os
import subprocess
import sys

from benchmark_runner import RESULTS, child_env, find_mpiexec, HERE
from cpu_topology import groups, topology


def sh(cmd):
    p = subprocess.run(cmd, cwd=HERE, env=child_env(), capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        print(p.stdout[-1500:], p.stderr[-1500:])
        raise RuntimeError(" ".join(cmd))
    for line in p.stdout.splitlines():
        if line.startswith("["):
            print("  " + line, flush=True)


def pin_configs(g):
    P, S, E = g["P"], g["P_smt"], g["E"]
    smt_pairs = [c for pair in zip(P, S) for c in pair]          # 0,1,2,3,...
    return [
        ("P1", P[:1]), ("E1", E[:1]), ("SMT2", smt_pairs[:2]),
        ("P4", P[:4]), ("E4", E[:4]),
        ("P8", P), ("E8", E), ("PSMT8", smt_pairs[:8]),
        ("P8+E8", P + E), ("P16smt", smt_pairs), ("P8+E4", P + E[:4]),
        ("ALL24", smt_pairs + E),
        ("def1", 1), ("def8", 8), ("def16", 16), ("def24", 24),
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", default=["pinning", "comm", "noise"])
    ap.add_argument("--reps-noise", type=int, default=7)
    ap.add_argument("--s", type=int, nargs="+", default=[2, 4, 8, 12, 16, 20, 24])
    args = ap.parse_args()
    mpiexec = find_mpiexec()
    os.makedirs(RESULTS, exist_ok=True)

    if "pinning" in args.exp:
        print("=== pinning ===")
        top = topology()
        g = groups(top)
        with open(os.path.join(RESULTS, "cpu_topology.json"), "w") as fh:
            json.dump(dict(cpus=top, groups=g), fh, indent=2)
        out = os.path.join(RESULTS, "pinning.csv")
        if os.path.exists(out):
            os.remove(out)
        for name, cpus in pin_configs(g):
            if isinstance(cpus, int):           # sin afinidad: decide Windows
                cmd = [mpiexec, "-n", str(cpus), sys.executable, "pin_bench.py",
                       "--config", name, "--csv", out]
            else:
                cmd = [mpiexec, "-n", str(len(cpus)), sys.executable, "pin_bench.py",
                       "--config", name, "--cpus", ",".join(map(str, cpus)), "--csv", out]
            sh(cmd)

    if "comm" in args.exp:
        print("=== comm ===")
        out = os.path.join(RESULTS, "comm_bench.csv")
        if os.path.exists(out):
            os.remove(out)
        for s in args.s:
            sh([mpiexec, "-n", str(s), sys.executable, "comm_bench.py", "--csv", out])

    if "noise" in args.exp:
        print("=== noise (strong N=50000) ===")
        subprocess.run([sys.executable, "-u", "benchmark_runner.py", "--exp", "strong",
                        "--N", "50000", "--reps", str(args.reps_noise)],
                       cwd=HERE, env=child_env(), check=True)


if __name__ == "__main__":
    main()
