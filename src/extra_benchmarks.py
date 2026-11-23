"""
extra_benchmarks.py -- microbenchmark de comunicación para alpha y beta.

  comm : colectivas MPI sin cómputo (comm_bench.py) para p = 2..16
         -> <out>/comm_bench.csv
Los valores de alpha y beta normalizan el término de comunicación de la
expresión teórica (Sec. 6.10 del informe).

Uso:  python extra_benchmarks.py --exp comm --out results/khipu/aumento_train
"""
import argparse
import os
import subprocess
import sys

from benchmark_runner import RESULTS as DEFAULT_RESULTS, child_env, find_mpiexec, limit_s, HERE


def sh(cmd):
    p = subprocess.run(cmd, cwd=HERE, env=child_env(), capture_output=True,
                       text=True, encoding="utf-8", errors="replace")
    if p.returncode != 0:
        print(p.stdout[-1500:], p.stderr[-1500:])
        raise RuntimeError(" ".join(cmd))
    for line in p.stdout.splitlines():
        if line.startswith("["):
            print("  " + line, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", default=["comm"], choices=["comm"])
    ap.add_argument("--s", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--out", default=DEFAULT_RESULTS, help="carpeta de resultados")
    args = ap.parse_args()
    mpiexec = find_mpiexec()
    results = os.path.abspath(args.out)
    os.makedirs(results, exist_ok=True)
    args.s = [s for s in limit_s(args.s) if s >= 2]

    print("=== comm ===")
    out = os.path.join(results, "comm_bench.csv")
    if os.path.exists(out):
        os.remove(out)
    for s in args.s:
        sh([mpiexec, "-n", str(s), sys.executable, "comm_bench.py", "--csv", out])


if __name__ == "__main__":
    main()
