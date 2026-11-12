"""
knn_mpi_v3_optimized.py -- Beta 3 (versión final): buffers contiguos + Scatterv/Gatherv.

Cambios respecto a la Beta 2:
  1. Colectivos con buffers (mayúscula): Bcast, Scatterv y Gatherv transfieren
     arreglos NumPy C-contiguos directamente (sin pickle ni copias de
     serialización); el costo se ajusta al modelo alpha-beta del informe.
  2. Reparto exacto para m_test no divisible entre s: con q = m // s, r = m % s
         m_p = q + 1  si p < r,   m_p = q  si p >= r
         counts_p = m_p * d,      displs_p = sum_{i<p} counts_i
     (las predicciones usan counts = m_p y displs análogos).
  3. Conteo de FLOPs de la región paralelizable (Ec. 1, por par de puntos):
         d restas + d multiplicaciones + (d-1) sumas + 1 raíz = 3d FLOPs
         FLOPs_total = 3 * m_test * n_train * d
     y rendimiento GFLOP/s = FLOPs_total / (T_comp * 1e9).
  4. Repeticiones dentro de un mismo lanzamiento MPI (--reps) para amortizar el
     arranque, y escritura de una fila CSV por repetición (--csv).
  5. --validate: compara las predicciones con sklearn KNeighborsClassifier
     (algoritmo 'brute', métrica euclidiana) -> validación de la fuente externa.

Ejecución:
  mpiexec -n 8 python knn_mpi_v3_optimized.py --data data/digits_N100000.npz \
          --reps 5 --csv results/results.csv --tag strong
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import platform

import numpy as np
from mpi4py import MPI
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

N_CLASSES = 10
CSV_FIELDS = ["version", "tag", "s", "N", "n_train", "m_test", "d", "k", "rep",
              "T_total", "T_comp", "T_comp_min", "T_comm", "T_bcast",
              "T_scatter", "T_gather", "T_merge", "flops", "gflops", "accuracy", "host"]


# ----------------------------------------------------------------------------
# Reparto por bloques (counts / displacements)
# ----------------------------------------------------------------------------
def block_partition(m, s):
    """Filas por proceso m_p y desplazamientos (en filas) para m % s != 0."""
    q, r = divmod(m, s)
    rows = np.array([q + 1 if p < r else q for p in range(s)], dtype=np.int64)
    displs = np.zeros(s, dtype=np.int64)
    displs[1:] = np.cumsum(rows)[:-1]
    return rows, displs


# ----------------------------------------------------------------------------
# Cómputo local
# ----------------------------------------------------------------------------
def knn_predict_block(X_train, y_train, sq_train, Q, k):
    D2 = np.einsum("ij,ij->i", Q, Q)[:, None] + sq_train[None, :]
    D2 -= 2.0 * (Q @ X_train.T)                       # GEMM (BLAS-3)
    np.maximum(D2, 0.0, out=D2)
    D = np.sqrt(D2, out=D2)                           # Ec. (1)
    nn = np.argpartition(D, k - 1, axis=1)[:, :k]
    labels = y_train[nn]
    votes = np.zeros((Q.shape[0], N_CLASSES), dtype=np.int64)
    np.add.at(votes, (np.arange(Q.shape[0])[:, None], labels), 1)
    return votes.argmax(axis=1)


def knn_local(X_train, y_train, X_local, k, block):
    out = np.empty(X_local.shape[0], dtype=np.int64)
    if X_local.shape[0] == 0:
        return out
    sq_train = np.einsum("ij,ij->i", X_train, X_train)
    for a in range(0, X_local.shape[0], block):
        out[a:a + block] = knn_predict_block(X_train, y_train, sq_train,
                                             X_local[a:a + block], k)
    return out


def load_dataset(path):
    if path is None:
        d = load_digits()
        return d.data.astype(np.float64), d.target.astype(np.int64)
    z = np.load(path)
    return z["X"].astype(np.float64), z["y"].astype(np.int64)


# ----------------------------------------------------------------------------
# Una ejecución completa del Algoritmo 5
# ----------------------------------------------------------------------------
def run_once(comm, X_train0, y_train0, X_test0, meta, k, block):
    rank, size = comm.Get_rank(), comm.Get_size()
    n, m, d = meta
    rows, displs = block_partition(m, size)

    # Buffers de recepción (todos los rangos conocen n, m, d).
    if rank == 0:
        X_train, y_train = X_train0, y_train0
    else:
        X_train = np.empty((n, d), dtype=np.float64)
        y_train = np.empty(n, dtype=np.int64)
    X_local = np.empty((rows[rank], d), dtype=np.float64)
    y_pred = np.empty(m, dtype=np.int64) if rank == 0 else None

    comm.Barrier()
    t0 = MPI.Wtime()

    # (1) Bcast: replicación de X_tr e y_tr  -> log s (alpha + n d beta)
    comm.Bcast([X_train, MPI.DOUBLE], root=0)
    comm.Bcast([y_train, MPI.INT64_T], root=0)
    t1 = MPI.Wtime()

    # (2) Scatterv: bloque X_te^(p) de m_p consultas -> alpha log s + m d beta
    sendbuf = [X_test0, (rows * d, displs * d), MPI.DOUBLE] if rank == 0 else None
    comm.Scatterv(sendbuf, [X_local, MPI.DOUBLE], root=0)
    t2 = MPI.Wtime()

    # (3) Cómputo local: m_p * n distancias euclidianas + selección + voto
    y_local = knn_local(X_train, y_train, X_local, k, block)
    t3 = MPI.Wtime()

    # (4) Gatherv: escritura exclusiva en posiciones disjuntas de y_hat
    recvbuf = [y_pred, (rows, displs), MPI.INT64_T] if rank == 0 else None
    comm.Gatherv([y_local, MPI.INT64_T], recvbuf, root=0)
    t4 = MPI.Wtime()

    local = np.array([t1 - t0, t2 - t1, t3 - t2, t4 - t3], dtype=np.float64)
    tmax = np.empty(4) if rank == 0 else None
    comm.Reduce(local, tmax, op=MPI.MAX, root=0)
    tcomp_min = comm.reduce(t3 - t2, op=MPI.MIN, root=0)
    return y_pred, t4 - t0, tmax, tcomp_min


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help=".npz (None = load_digits)")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--block", type=int, default=256)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--csv", default=None, help="CSV de salida (append)")
    ap.add_argument("--tag", default="", help="etiqueta del experimento")
    ap.add_argument("--validate", action="store_true",
                    help="comparar contra sklearn KNeighborsClassifier")
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
        meta = (X_train.shape[0], X_test.shape[0], X_train.shape[1])
        N = X.shape[0]
    else:
        X_train = y_train = X_test = y_test = None
        meta = N = None
    meta = comm.bcast(meta, root=0)            # (n_train, m_test, d): 3 enteros
    n, m, d = meta
    flops = 3 * m * n * d                      # región paralelizable

    writer = None
    if rank == 0 and args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        new = not os.path.exists(args.csv)
        fh = open(args.csv, "a", newline="")
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        if new:
            writer.writeheader()

    for rep in range(args.reps):
        y_pred, t_total, tmax, tcomp_min = run_once(
            comm, X_train, y_train, X_test, meta, args.k, args.block)
        if rank == 0:
            t_bcast, t_scatter, t_comp, t_gather = tmax
            t_comm = t_bcast + t_scatter + t_gather
            acc = float(np.mean(y_pred == y_test))
            gflops = flops / (t_comp * 1e9)
            print(f"[v3] s={size:2d} N={N} n={n} m={m} rep={rep} acc={acc:.4f} "
                  f"T_total={t_total:.4f} T_comp={t_comp:.4f} T_comm={t_comm:.4f} "
                  f"GFLOP/s={gflops:.2f}", flush=True)
            if writer:
                writer.writerow(dict(
                    version="v3", tag=args.tag, s=size, N=N, n_train=n, m_test=m,
                    d=d, k=args.k, rep=rep, T_total=t_total, T_comp=t_comp,
                    T_comp_min=tcomp_min, T_comm=t_comm, T_bcast=t_bcast,
                    T_scatter=t_scatter, T_gather=t_gather, T_merge=0.0, flops=flops,
                    gflops=gflops, accuracy=acc, host=platform.node()))
                fh.flush()

    if rank == 0 and args.validate:
        from sklearn.neighbors import KNeighborsClassifier
        ref = KNeighborsClassifier(n_neighbors=args.k, algorithm="brute",
                                   metric="euclidean").fit(X_train, y_train)
        y_ref = ref.predict(X_test)
        agree = np.mean(y_ref == y_pred)
        print(f"[validate] acc_sklearn={np.mean(y_ref == y_test):.4f}  "
              f"acc_mpi={np.mean(y_pred == y_test):.4f}  "
              f"coincidencia de predicciones={agree * 100:.2f} %")
    if writer:
        fh.close()


if __name__ == "__main__":
    main()
