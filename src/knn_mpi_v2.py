"""
knn_mpi_v2.py -- Beta 2: cómputo local vectorizado + medición T_comp / T_comm.

Cambios respecto a la Beta 1:
  1. Distancias por bloques de consultas con la identidad
         ||q - x||^2 = ||q||^2 + ||x||^2 - 2 q.x
     -> el producto Q X^T se resuelve con GEMM (BLAS nivel 3); no hay bucle
     Python por punto de entrenamiento. Se procesan `--block` consultas a la vez
     para acotar la memoria a block * n_train * 8 bytes.
  2. Selección de k vecinos con np.argpartition, O(n) en vez de O(n log n).
  3. Votación vectorizada (bincount por fila) en lugar de Counter.
  4. Instrumentación con MPI.Wtime(): cada fase de comunicación y la de
     cómputo se cronometran por separado; se reporta el máximo entre rangos
     (el proceso más lento determina el tiempo paralelo).
Los colectivos siguen siendo los genéricos en minúscula (pickle).

IMPORTANTE: fijar OMP_NUM_THREADS=1 (lo hace este script antes de importar
NumPy) para que BLAS no use hilos y cada proceso MPI ocupe un solo núcleo.

Ejecución:  mpiexec -n 8 python knn_mpi_v2.py --data data/digits_N50000.npz
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse

import numpy as np
from mpi4py import MPI
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

N_CLASSES = 10


def knn_predict_block(X_train, y_train, sq_train, Q, k, n_classes=N_CLASSES):
    """Predice las etiquetas de un bloque de consultas Q (b x d)."""
    # D2[i, j] = ||Q_i||^2 + ||X_j||^2 - 2 Q_i . X_j        (b x n)
    D2 = np.einsum("ij,ij->i", Q, Q)[:, None] + sq_train[None, :] - 2.0 * (Q @ X_train.T)
    np.maximum(D2, 0.0, out=D2)          # corrige redondeos negativos
    D = np.sqrt(D2, out=D2)              # distancia euclidiana (Ec. 1)
    nn = np.argpartition(D, k - 1, axis=1)[:, :k]          # k vecinos, O(n)
    labels = y_train[nn]                                   # (b x k)
    votes = np.zeros((Q.shape[0], n_classes), dtype=np.int64)
    np.add.at(votes, (np.arange(Q.shape[0])[:, None], labels), 1)
    return votes.argmax(axis=1)          # moda (empate -> etiqueta menor)


def knn_local(X_train, y_train, X_local, k, block):
    sq_train = np.einsum("ij,ij->i", X_train, X_train)
    out = np.empty(X_local.shape[0], dtype=np.int64)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--block", type=int, default=256, help="consultas por bloque")
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    if rank == 0:
        X, y = load_dataset(args.data)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        test_chunks = np.array_split(X_test, size)
    else:
        X_train = y_train = test_chunks = None

    comm.Barrier()
    t0 = MPI.Wtime()

    X_train = comm.bcast(X_train, root=0)
    y_train = comm.bcast(y_train, root=0)
    t1 = MPI.Wtime()
    X_local = comm.scatter(test_chunks, root=0)
    t2 = MPI.Wtime()

    y_local = knn_local(X_train, y_train, X_local, args.k, args.block)
    t3 = MPI.Wtime()

    gathered = comm.gather(y_local, root=0)
    t4 = MPI.Wtime()

    t_bcast, t_scatter, t_comp, t_gather = t1 - t0, t2 - t1, t3 - t2, t4 - t3
    # El proceso más lento define cada fase.
    t_comp_max = comm.reduce(t_comp, op=MPI.MAX, root=0)
    t_comm_max = comm.reduce(t_bcast + t_scatter + t_gather, op=MPI.MAX, root=0)

    if rank == 0:
        y_pred = np.concatenate(gathered)
        accuracy = np.mean(y_pred == y_test)
        t_total = MPI.Wtime() - t0
        print(f"[v2] s={size} n_train={X_train.shape[0]} m_test={y_test.size} "
              f"acc={accuracy:.4f} T_total={t_total:.4f} T_comp={t_comp_max:.4f} "
              f"T_comm={t_comm_max:.4f} (bcast={t_bcast:.4f} scatter={t_scatter:.4f} "
              f"gather={t_gather:.4f})")


if __name__ == "__main__":
    main()
