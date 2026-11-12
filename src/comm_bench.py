"""
comm_bench.py -- Microbenchmark de colectivas MPI (sin cómputo) para estimar alpha y beta.

Mide Bcast, Scatter y Gather de buffers float64 de M palabras (M = tamaño TOTAL
del mensaje en la raíz; Scatter/Gather mueven M/s por proceso), para varios M.
No hay desbalance de carga: todo rango entra a la colectiva tras un Barrier,
así que el tiempo es comunicación pura (a diferencia de T_gather en knn_mpi_v3,
que incluye la espera por el proceso más lento).

Modelo a ajustar (Sec. 2.3.4 del informe parcial, árbol binomial):
    T_Bcast(M,s)   = log2(s) * (alpha + M beta)
    T_Scatter(M,s) = alpha log2(s) + M beta     (ídem Gather)

Uso:  mpiexec -n 8 python comm_bench.py --csv ../results/comm_bench.csv
"""
import argparse
import csv
import os

import numpy as np
from mpi4py import MPI

SIZES = [2 ** e for e in range(7, 24, 2)]   # 128 .. 8.4e6 palabras (1 KiB .. 64 MiB)


def time_op(comm, fn, iters):
    ts = []
    for _ in range(iters):
        comm.Barrier()
        t0 = MPI.Wtime()
        fn()
        ts.append(MPI.Wtime() - t0)
    arr = np.array(ts)
    tmax = np.empty_like(arr) if comm.Get_rank() == 0 else None
    comm.Reduce(arr, tmax, op=MPI.MAX, root=0)    # la colectiva termina con el último
    return tmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--max-words", type=int, default=SIZES[-1])
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    rows = []
    for M in [m for m in SIZES if m <= args.max_words]:
        M = (M // size) * size or size               # divisible entre s
        iters = int(np.clip(2e7 / M, 5, 200))
        full = np.ones(M) if rank == 0 else np.empty(M)
        part = np.empty(M // size)
        recv = np.empty(M) if rank == 0 else None

        res = {
            "Bcast": time_op(comm, lambda: comm.Bcast([full, MPI.DOUBLE], root=0), iters),
            "Scatter": time_op(comm, lambda: comm.Scatter(
                [full, MPI.DOUBLE] if rank == 0 else None, [part, MPI.DOUBLE], root=0), iters),
            "Gather": time_op(comm, lambda: comm.Gather(
                [part, MPI.DOUBLE], [recv, MPI.DOUBLE] if rank == 0 else None, root=0), iters),
        }
        if rank == 0:
            for op, t in res.items():
                t = t[1:] if t.size > 1 else t          # descarta la 1a (calentamiento)
                rows.append(dict(op=op, s=size, M=M, bytes=8 * M, iters=iters,
                                 t_min=t.min(), t_med=float(np.median(t))))

    if rank == 0:
        new = not os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            if new:
                w.writeheader()
            w.writerows(rows)
        big = [r for r in rows if r["M"] == max(x["M"] for x in rows)]
        print(f"[comm] s={size:2d}  M={big[0]['M']:.1e}  " +
              "  ".join(f"{r['op']}={r['t_med'] * 1e3:7.2f}ms" for r in big), flush=True)


if __name__ == "__main__":
    main()
