"""
pin_bench.py -- Experimento P-cores vs E-cores vs SMT (i7-14650HX).

Cada proceso MPI ejecuta la MISMA carga fija del núcleo de KNN (Beta 3):
m_loc consultas contra n_train puntos de entrenamiento (d = 64), es decir
3 * m_loc * n_train * d FLOPs por proceso (escalabilidad débil "pura").
Si el hardware escalara perfectamente, el tiempo por proceso sería constante
en todas las configuraciones; cualquier aumento se debe al tipo de núcleo,
a la compartición SMT o a la contención de memoria/potencia.

Afinamiento explícito: el rango r se fija a la CPU lógica cpus[r] con
psutil.Process().cpu_affinity([cpu]) (verificable, independiente de la
implementación MPI). Sin --cpus, el planificador de Windows decide (default).

Uso (lo invoca extra_benchmarks.py):
    mpiexec -n 8 python pin_bench.py --config P8 --csv ../results/pinning.csv
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv

import numpy as np
import psutil
from mpi4py import MPI

from knn_mpi_v3_optimized import knn_local


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-train", type=int, default=160000)
    ap.add_argument("--m-loc", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--cpus", default="", help="lista de CPUs lógicas, una por rango")
    ap.add_argument("--csv", required=True)
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    rng = np.random.default_rng(rank)
    d = 64
    X_train = rng.integers(0, 17, (args.n_train, d)).astype(np.float64)
    y_train = rng.integers(0, 10, args.n_train).astype(np.int64)
    X_loc = rng.integers(0, 17, (args.m_loc, d)).astype(np.float64)
    flops = 3 * args.m_loc * args.n_train * d
    if args.cpus:
        cpus = [int(c) for c in args.cpus.split(",")]
        psutil.Process().cpu_affinity([cpus[rank % len(cpus)]])
    aff = psutil.Process().cpu_affinity()
    knn_local(X_train, y_train, X_loc[:64], 3, 256)          # calentamiento

    rows = []
    for rep in range(args.reps):
        comm.Barrier()
        t0 = MPI.Wtime()
        knn_local(X_train, y_train, X_loc, 3, 256)
        t = MPI.Wtime() - t0
        rows.append(dict(config=args.config, s=size, rank=rank, rep=rep,
                         affinity=" ".join(map(str, aff)) if len(aff) < 24 else "all",
                         t=t, gflops=flops / (t * 1e9)))
    allrows = comm.gather(rows, root=0)
    if rank == 0:
        flat = [r for rr in allrows for r in rr]
        new = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(flat[0].keys()))
            if new:
                w.writeheader()
            w.writerows(flat)
        g = np.array([r["gflops"] for r in flat]).reshape(size, args.reps)
        per_rank = np.median(g, axis=1)
        print(f"[pin] {args.config:8s} s={size:2d}  GFLOP/s por proceso: "
              f"min={per_rank.min():5.2f} med={np.median(per_rank):5.2f} "
              f"max={per_rank.max():5.2f}  agregado={per_rank.sum():6.1f}  "
              f"afinidad r0={flat[0]['affinity']}", flush=True)


if __name__ == "__main__":
    main()
