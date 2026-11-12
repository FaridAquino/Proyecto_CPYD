"""
knn_mpi_dag.py -- Variante DAG del enunciado: partición del ENTRENAMIENTO.

Descomposición (la del DAG del enunciado, Sec. 1.3 del informe parcial):
  1. Scatterv(X_tr, y_tr): el proceso p recibe n_p ~ n_train/s puntos de
     entrenamiento                            -> alpha log s + n d beta
  2. Bcast(X_te): todos reciben las m consultas -> log s (alpha + m d beta)
  3. Cómputo local: para cada consulta, distancias a sus n_p puntos y
     selección de los k candidatos LOCALES   -> m n_p d = m n d / s
  4. Gather de los candidatos (distancia, etiqueta): m * k * s pares
                                              -> alpha log s + m k s beta
  5. Fusión en el rango 0: de los s*k candidatos de cada consulta se toman los
     k globales y se vota por mayoría        -> O(m s k)   (fase SERIAL)

Corrección: los k vecinos globales de q están contenidos en la unión de los k
vecinos locales de cada bloque (si x pertenece al top-k global, también
pertenece al top-k de su bloque), por lo que la predicción coincide con la
secuencial salvo empates exactos de distancia.

Comparada con la Beta 3 (replica X_tr, reparte X_te), esta variante evita el
Bcast de X_tr (O(n d log s)) a cambio de un Gather que crece con s y de una fase
de fusión serial O(m s k) en el rango 0 -> analizado en Resultados.

Ejecución:
  mpiexec -n 8 python knn_mpi_dag.py --data ../data/digits_N100000.npz \
          --reps 5 --csv ../results/results.csv --tag dag-strong --validate
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import platform

import numpy as np
from mpi4py import MPI
from sklearn.model_selection import train_test_split

from knn_mpi_v3_optimized import (CSV_FIELDS, N_CLASSES, block_partition,
                                  load_dataset)


def local_candidates(X_part, y_part, X_test, k, block):
    """k candidatos locales (distancia, etiqueta) para cada una de las m consultas."""
    m = X_test.shape[0]
    kk = min(k, X_part.shape[0])
    dist = np.full((m, k), np.inf)            # relleno si n_p < k
    lab = np.full((m, k), -1, dtype=np.int64)
    if kk == 0:
        return dist, lab
    sq = np.einsum("ij,ij->i", X_part, X_part)
    for a in range(0, m, block):
        Q = X_test[a:a + block]
        D2 = np.einsum("ij,ij->i", Q, Q)[:, None] + sq[None, :]
        D2 -= 2.0 * (Q @ X_part.T)
        np.maximum(D2, 0.0, out=D2)
        D = np.sqrt(D2, out=D2)
        idx = np.argpartition(D, kk - 1, axis=1)[:, :kk]
        rows = np.arange(Q.shape[0])[:, None]
        dist[a:a + block, :kk] = D[rows, idx]
        lab[a:a + block, :kk] = y_part[idx]
    return dist, lab


def merge_and_vote(all_dist, all_lab, k):
    """all_*: (s, m, k) -> top-k global por consulta y voto mayoritario."""
    s, m, _ = all_dist.shape
    D = all_dist.transpose(1, 0, 2).reshape(m, s * k)
    L = all_lab.transpose(1, 0, 2).reshape(m, s * k)
    idx = np.argpartition(D, k - 1, axis=1)[:, :k]
    labels = L[np.arange(m)[:, None], idx]
    votes = np.zeros((m, N_CLASSES), dtype=np.int64)
    np.add.at(votes, (np.arange(m)[:, None], labels), 1)
    return votes.argmax(axis=1)


def run_once(comm, X_train0, y_train0, X_test0, meta, k, block):
    rank, size = comm.Get_rank(), comm.Get_size()
    n, m, d = meta
    rows, displs = block_partition(n, size)

    X_part = np.empty((rows[rank], d), dtype=np.float64)
    y_part = np.empty(rows[rank], dtype=np.int64)
    X_test = X_test0 if rank == 0 else np.empty((m, d), dtype=np.float64)

    comm.Barrier()
    t0 = MPI.Wtime()

    # (1) Scatterv del entrenamiento
    sx = [X_train0, (rows * d, displs * d), MPI.DOUBLE] if rank == 0 else None
    sy = [y_train0, (rows, displs), MPI.INT64_T] if rank == 0 else None
    comm.Scatterv(sx, [X_part, MPI.DOUBLE], root=0)
    comm.Scatterv(sy, [y_part, MPI.INT64_T], root=0)
    t1 = MPI.Wtime()

    # (2) Bcast de las consultas
    comm.Bcast([X_test, MPI.DOUBLE], root=0)
    t2 = MPI.Wtime()

    # (3) Cómputo local: k candidatos por consulta
    dist, lab = local_candidates(X_part, y_part, X_test, k, block)
    t3 = MPI.Wtime()

    # (4) Gather de candidatos
    all_dist = np.empty((size, m, k)) if rank == 0 else None
    all_lab = np.empty((size, m, k), dtype=np.int64) if rank == 0 else None
    comm.Gather([dist, MPI.DOUBLE], [all_dist, MPI.DOUBLE] if rank == 0 else None, root=0)
    comm.Gather([lab, MPI.INT64_T], [all_lab, MPI.INT64_T] if rank == 0 else None, root=0)
    t4 = MPI.Wtime()

    # (5) Fusión + voto (serial, rango 0)
    y_pred = merge_and_vote(all_dist, all_lab, k) if rank == 0 else None
    t5 = MPI.Wtime()

    local = np.array([t1 - t0, t2 - t1, t3 - t2, t4 - t3], dtype=np.float64)
    tmax = np.empty(4) if rank == 0 else None
    comm.Reduce(local, tmax, op=MPI.MAX, root=0)
    tcomp_min = comm.reduce(t3 - t2, op=MPI.MIN, root=0)
    return y_pred, t5 - t0, tmax, tcomp_min, t5 - t4


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--tag", default="dag")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    if rank == 0:
        X, y = load_dataset(args.data)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        X_train = np.ascontiguousarray(X_train)
        X_test = np.ascontiguousarray(X_test)
        y_train = np.ascontiguousarray(y_train, dtype=np.int64)
        meta, N = (X_train.shape[0], X_test.shape[0], X_train.shape[1]), X.shape[0]
    else:
        X_train = y_train = X_test = y_test = meta = N = None
    meta = comm.bcast(meta, root=0)
    n, m, d = meta
    flops = 3 * m * n * d

    writer = None
    if rank == 0 and args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        new = not os.path.exists(args.csv)
        fh = open(args.csv, "a", newline="")
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new:
            writer.writeheader()

    for rep in range(args.reps):
        y_pred, t_total, tmax, tcomp_min, t_merge = run_once(
            comm, X_train, y_train, X_test, meta, args.k, args.block)
        if rank == 0:
            t_scatter, t_bcast, t_comp, t_gather = tmax
            t_comm = t_scatter + t_bcast + t_gather
            acc = float(np.mean(y_pred == y_test))
            gflops = flops / (t_comp * 1e9)
            print(f"[dag] s={size:2d} N={N} n={n} m={m} rep={rep} acc={acc:.4f} "
                  f"T_total={t_total:.4f} T_comp={t_comp:.4f} T_comm={t_comm:.4f} "
                  f"T_merge={t_merge:.4f} GFLOP/s={gflops:.2f}", flush=True)
            if writer:
                # T_comp = sólo la región paralela; T_merge = fusión serial (rango 0).
                writer.writerow(dict(
                    version="dag", tag=args.tag, s=size, N=N, n_train=n, m_test=m,
                    d=d, k=args.k, rep=rep, T_total=t_total, T_comp=t_comp,
                    T_comp_min=tcomp_min, T_comm=t_comm, T_bcast=t_bcast,
                    T_scatter=t_scatter, T_gather=t_gather, T_merge=t_merge, flops=flops,
                    gflops=gflops, accuracy=acc, host=platform.node()))
                fh.flush()

    if rank == 0 and args.validate:
        from sklearn.neighbors import KNeighborsClassifier
        y_ref = KNeighborsClassifier(n_neighbors=args.k, algorithm="brute",
                                     metric="euclidean").fit(X_train, y_train).predict(X_test)
        print(f"[validate] acc_sklearn={np.mean(y_ref == y_test):.4f}  "
              f"acc_dag={np.mean(y_pred == y_test):.4f}  "
              f"coincidencia={np.mean(y_ref == y_pred) * 100:.2f} %")
    if writer:
        fh.close()


if __name__ == "__main__":
    main()
