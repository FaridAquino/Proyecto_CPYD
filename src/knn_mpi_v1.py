"""
knn_mpi_v1.py -- Beta 1: paralelización directa con mpi4py (objetos pickle).

Cambios respecto a knn_digits_sec.py:
  * Paradigma SPMD: todos los rangos ejecutan el mismo programa.
  * comm.bcast  -> replica (X_train, y_train) en los s procesos.
  * comm.scatter-> reparte X_test en s bloques (np.array_split, admite m % s != 0).
  * comm.gather -> recolecta las predicciones locales en el rango 0.
  * El cómputo local es IDÉNTICO al secuencial (bucle Python por punto), de modo
    que la única diferencia es la descomposición de datos.

Limitaciones (motivan la Beta 2): los colectivos en minúscula serializan con
pickle (copias extra) y el bucle Python O(m n d) es ~100x más lento que NumPy.
Pensada para load_digits original; en N grandes es impracticable.

Ejecución:  mpiexec -n 4 python knn_mpi_v1.py [--data data/digits_N50000.npz]
"""
import argparse
from collections import Counter

import numpy as np
from mpi4py import MPI
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def knn_predict(test_point, X_train, y_train, k):
    # Idéntica a knn_digits_sec.py (misma firma y cuerpo).
    distances = [euclidean_distance(test_point, x) for x in X_train]
    k_indices = np.argsort(distances)[:k]
    k_labels = [y_train[i] for i in k_indices]
    most_common = Counter(k_labels).most_common(1)
    return most_common[0][0]


def load_dataset(path):
    if path is None:
        d = load_digits()
        return d.data, d.target
    z = np.load(path)
    return z["X"], z["y"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None, help=".npz de data_generator.py")
    ap.add_argument("--k", type=int, default=3)
    args = ap.parse_args()

    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()

    if rank == 0:
        X, y = load_dataset(args.data)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)
        test_chunks = np.array_split(X_test, size)   # s bloques (casi) iguales
    else:
        X_train = y_train = test_chunks = None

    comm.Barrier()
    t_start = MPI.Wtime()

    # --- Comunicación 1: difusión del conjunto de entrenamiento -------------
    X_train = comm.bcast(X_train, root=0)
    y_train = comm.bcast(y_train, root=0)
    # --- Comunicación 2: reparto del conjunto de prueba ---------------------
    X_local = comm.scatter(test_chunks, root=0)

    # --- Cómputo local (igual al secuencial) --------------------------------
    y_local = [knn_predict(x, X_train, y_train, args.k) for x in X_local]

    # --- Comunicación 3: recolección de predicciones ------------------------
    gathered = comm.gather(y_local, root=0)

    if rank == 0:
        y_pred = np.concatenate([np.asarray(g, dtype=np.int64) for g in gathered])
        accuracy = np.mean(y_pred == y_test)
        t_end = MPI.Wtime()
        print(f"[v1] s={size}  n_train={len(X_train)}  m_test={len(y_test)}  "
              f"accuracy={accuracy:.4f}  T_p={t_end - t_start:.4f} s")


if __name__ == "__main__":
    main()
