"""
data_generator.py -- Aumento sintético de load_digits (CS4052, Proyecto 5: KNN).

Motivación: load_digits tiene sólo 1797 muestras (n_train = 1437, m_test = 360,
d = 64). En un i7-14650HX el KNN completo tarda < 0.05 s y el overhead de MPI
domina. Para obtener mediciones representativas se generan conjuntos de N
muestras que preservan:
  * d = 64 características en [0, 16] (intensidad de píxel de load_digits),
  * C = 10 clases con las mismas proporciones a priori,
  * la media mu_c y la covarianza Sigma_c de cada clase c.

Modelo generativo (mezcla gaussiana por clase):
    y ~ Categorical(pi),   x | y=c ~ N(mu_c, Sigma_c + eps*I),
    x <- clip(round(x), 0, 16)        (opcional --no-round para valores reales)

El término eps*I regulariza Sigma_c, que es singular porque varios píxeles del
borde son constantes (varianza nula) en load_digits.

Uso:
    python data_generator.py --n 50000 100000 200000 --out data
    python data_generator.py --n 75000 --seed 7 --out data
Salida: data/digits_N<N>.npz con arreglos X (float64, N x 64) e y (int64, N).
"""
import argparse
import os
import time

import numpy as np
from sklearn.datasets import load_digits


def class_statistics(X, y):
    """Devuelve (clases, priors, medias, covarianzas) por clase."""
    classes = np.unique(y)
    priors = np.array([(y == c).mean() for c in classes])
    means = np.stack([X[y == c].mean(axis=0) for c in classes])
    covs = np.stack([np.cov(X[y == c], rowvar=False) for c in classes])
    return classes, priors, means, covs


def generate(n_samples, seed=42, eps=1e-3, round_values=True):
    """Genera n_samples puntos sintéticos con la estadística de load_digits."""
    digits = load_digits()
    X0, y0 = digits.data.astype(np.float64), digits.target.astype(np.int64)
    d = X0.shape[1]
    classes, priors, means, covs = class_statistics(X0, y0)

    rng = np.random.default_rng(seed)
    y = rng.choice(classes, size=n_samples, p=priors)
    X = np.empty((n_samples, d), dtype=np.float64)

    for c, mu, cov in zip(classes, means, covs):
        idx = np.flatnonzero(y == c)
        if idx.size == 0:
            continue
        # Factor de Cholesky de Sigma_c + eps*I (definida positiva).
        L = np.linalg.cholesky(cov + eps * np.eye(d))
        Z = rng.standard_normal((idx.size, d))
        X[idx] = mu + Z @ L.T

    np.clip(X, 0.0, 16.0, out=X)
    if round_values:
        np.rint(X, out=X)
    return X, y


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, nargs="+", default=[50000, 100000, 200000],
                    help="tamaños N a generar")
    ap.add_argument("--out", default="data", help="directorio de salida")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eps", type=float, default=1e-3,
                    help="regularización de la covarianza")
    ap.add_argument("--no-round", action="store_true",
                    help="no redondear a enteros (valores reales en [0,16])")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for n in args.n:
        t0 = time.perf_counter()
        X, y = generate(n, seed=args.seed, eps=args.eps,
                        round_values=not args.no_round)
        path = os.path.join(args.out, f"digits_N{n}.npz")
        np.savez_compressed(path, X=X, y=y)
        counts = np.bincount(y, minlength=10)
        print(f"[gen] N={n:>7d}  d={X.shape[1]}  clases={counts.tolist()}  "
              f"rango=[{X.min():.0f},{X.max():.0f}]  -> {path} "
              f"({time.perf_counter() - t0:.2f} s)")


if __name__ == "__main__":
    main()
