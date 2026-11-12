"""
benchmark_runner.py -- Automatización de experimentos (Fase 3, CS4052 Proyecto 5).

Experimentos (todos escriben en un único CSV consolidado, una fila por repetición):

  versions   Registro de desarrollo (parte b): secuencial, Beta 1, Beta 2, Beta 3 y
             DAG sobre load_digits original, s in {1,2,4,8}.
  strong     Escalabilidad fuerte (Amdahl): N fijo, s in S. Beta 3 y DAG.
  weak-lin   Escalabilidad débil con N/s constante: N = N0 * s.
             OJO: el trabajo es W = 3 m n d ~ N^2, así que W/s ~ N0^2 * s CRECE.
  weak-work  Escalabilidad débil con trabajo por proceso constante (Gustafson):
             W/s = cte  =>  N^2/s = cte  =>  N = N0 * sqrt(s).

El runner es reanudable: omite combinaciones (version, tag, s, N) que ya tienen
las repeticiones pedidas en el CSV. Fija OMP/MKL/OPENBLAS_NUM_THREADS=1 para que
cada proceso MPI use un solo núcleo, y guarda la información del sistema.

Uso:
  python benchmark_runner.py --quick                     # prueba rápida (~2 min)
  python benchmark_runner.py                             # campaña completa
  python benchmark_runner.py --exp strong --N 200000 --reps 5
"""
import argparse
import csv
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")

S_FULL = [1, 2, 4, 8, 12, 16, 20, 24]
FIELDS = ["version", "tag", "s", "N", "n_train", "m_test", "d", "k", "rep",
          "T_total", "T_comp", "T_comp_min", "T_comm", "T_bcast", "T_scatter",
          "T_gather", "T_merge", "flops", "gflops", "accuracy", "host"]


# ----------------------------------------------------------------------------
def find_mpiexec():
    # MPIEXEC permite forzar el lanzador (p. ej. en un clúster: export MPIEXEC=mpirun).
    cand = [shutil.which(os.environ["MPIEXEC"]) if os.environ.get("MPIEXEC") else None,
            shutil.which("mpiexec"), shutil.which("mpirun"),
            os.path.join(sys.prefix, "Library", "bin", "mpiexec.exe"),
            os.path.join(os.environ.get("MSMPI_BIN", ""), "mpiexec.exe")]
    for c in cand:
        if c and os.path.isfile(c):
            return c
    sys.exit("No se encontró mpiexec/mpirun (cargue el módulo MPI o instale Intel MPI / MS-MPI).")


def limit_s(s_list):
    """Dentro de un trabajo SLURM descarta los s que superan las tareas asignadas."""
    ntasks = os.environ.get("SLURM_NTASKS")
    if not ntasks:
        return s_list
    keep = [s for s in s_list if s <= int(ntasks)]
    if len(keep) < len(s_list):
        print(f"[runner] SLURM_NTASKS={ntasks}: se omiten s={[s for s in s_list if s > int(ntasks)]}")
    return keep


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def child_env():
    env = os.environ.copy()
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[v] = "1"
    env["MPLBACKEND"] = "Agg"
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def dataset_path(N):
    """Devuelve la ruta del .npz de tamaño N, generándolo si no existe."""
    path = os.path.join(DATA, f"digits_N{N}.npz")
    if not os.path.exists(path):
        sys.path.insert(0, HERE)
        from data_generator import generate
        os.makedirs(DATA, exist_ok=True)
        X, y = generate(N, seed=42)
        np.savez_compressed(path, X=X, y=y)
        print(f"  [gen] {path}")
    return path


def done_reps(csv_path):
    """Cuenta repeticiones ya registradas por (version, tag, s, N)."""
    cnt = {}
    if os.path.exists(csv_path):
        with open(csv_path, newline="") as fh:
            for r in csv.DictReader(fh):
                key = (r["version"], r["tag"], int(r["s"]), int(r["N"]))
                cnt[key] = cnt.get(key, 0) + 1
    return cnt


def run(cmd, timeout):
    t0 = time.perf_counter()
    p = subprocess.run(cmd, cwd=HERE, env=child_env(), capture_output=True,
                       text=True, encoding="utf-8", errors="replace",
                       timeout=timeout)
    if p.returncode != 0:
        print(p.stdout[-2000:], p.stderr[-2000:])
        raise RuntimeError(f"Falló: {' '.join(cmd)}")
    return p.stdout, time.perf_counter() - t0


def append_rows(csv_path, rows):
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in FIELDS})


# ----------------------------------------------------------------------------
# Experimento: registro de versiones (load_digits, se parsea la salida estándar)
# ----------------------------------------------------------------------------
def exp_versions(mpiexec, csv_path, reps, s_list, timeout):
    have = done_reps(csv_path)
    N, n, m, d = 1797, 1437, 360, 64
    base = dict(N=N, n_train=n, m_test=m, d=d, k=3, tag="versions",
                flops=3 * m * n * d, host=platform.node())

    if have.get(("seq", "versions", 1, N), 0) < reps:
        rows = []
        for rep in range(reps):
            out, _ = run([sys.executable, "knn_digits_sec.py"], timeout)
            acc = float(re.search(r"Accuracy: ([\d.]+)", out).group(1))
            t = float(re.search(r"sequential\): ([\d.]+)", out).group(1))
            rows.append(dict(base, version="seq", s=1, rep=rep, T_total=t,
                             T_comp=t, T_comm=0.0, accuracy=acc,
                             gflops=base["flops"] / (t * 1e9)))
        append_rows(csv_path, rows)
        print(f"  seq          T={t:.4f}s acc={acc:.4f}")

    pat_v1 = re.compile(r"accuracy=([\d.]+)\s+T_p=([\d.]+)")
    pat_v2 = re.compile(r"acc=([\d.]+) T_total=([\d.]+) T_comp=([\d.]+) T_comm=([\d.]+) "
                        r"\(bcast=([\d.]+) scatter=([\d.]+) gather=([\d.]+)\)")
    for s in s_list:
        for version, script in (("v1", "knn_mpi_v1.py"), ("v2", "knn_mpi_v2.py")):
            if have.get((version, "versions", s, N), 0) >= reps:
                continue
            rows = []
            for rep in range(reps):
                out, _ = run([mpiexec, "-n", str(s), sys.executable, script], timeout)
                if version == "v1":
                    acc, t = map(float, pat_v1.search(out).groups())
                    r = dict(T_total=t, accuracy=acc)
                else:
                    acc, t, tc, tm, tb, ts, tg = map(float, pat_v2.search(out).groups())
                    r = dict(T_total=t, T_comp=tc, T_comm=tm, T_bcast=tb,
                             T_scatter=ts, T_gather=tg, accuracy=acc,
                             gflops=base["flops"] / (tc * 1e9))
                rows.append(dict(base, version=version, s=s, rep=rep, **r))
            append_rows(csv_path, rows)
            print(f"  {version} s={s:2d}      T={rows[-1]['T_total']:.4f}s "
                  f"acc={rows[-1]['accuracy']:.4f}")
        for version, script in (("v3", "knn_mpi_v3_optimized.py"), ("dag", "knn_mpi_dag.py")):
            run_csv_script(mpiexec, script, version, "versions", s, None, N,
                           reps, csv_path, have, timeout)


# ----------------------------------------------------------------------------
# Experimentos con scripts que escriben CSV directamente (v3, dag)
# ----------------------------------------------------------------------------
def run_csv_script(mpiexec, script, version, tag, s, data, N, reps, csv_path,
                   have, timeout):
    missing = reps - have.get((version, tag, s, N), 0)
    if missing <= 0:
        return
    cmd = [mpiexec, "-n", str(s), sys.executable, script, "--reps", str(missing),
           "--csv", csv_path, "--tag", tag]
    if data:
        cmd += ["--data", data]
    out, wall = run(cmd, timeout)
    last = [l for l in out.splitlines() if l.startswith(f"[{version}]")][-1]
    print(f"  {version:3s} {tag:9s} s={s:2d} N={N:>7d}  ({wall:6.1f}s)  {last.split('rep=')[1]}")


def exp_scaling(mpiexec, csv_path, kind, versions, s_list, N_values, N0, reps, timeout):
    have = done_reps(csv_path)
    scripts = {"v3": "knn_mpi_v3_optimized.py", "dag": "knn_mpi_dag.py"}
    jobs = []
    if kind == "strong":
        for N in N_values:
            jobs += [(f"strong", s, N) for s in s_list]
    elif kind == "weak-lin":
        jobs = [("weak-lin", s, N0 * s) for s in s_list]
    elif kind == "weak-work":
        jobs = [("weak-work", s, int(round(N0 * math.sqrt(s)))) for s in s_list]
    for tag, s, N in jobs:
        data = dataset_path(N)
        for v in versions:
            run_csv_script(mpiexec, scripts[v], v, tag, s, data, N, reps,
                           csv_path, have, timeout)


# ----------------------------------------------------------------------------
def system_info(mpiexec):
    import mpi4py
    # Sin MPI_Init en el proceso padre: un padre inicializado que luego lanza
    # mpiexec como subproceso puede colgarse con OpenMPI/MPICH bajo SLURM.
    mpi4py.rc.initialize = False
    mpi4py.rc.finalize = False
    from mpi4py import MPI
    info = dict(
        fecha=time.strftime("%Y-%m-%d %H:%M:%S"),
        host=platform.node(), os=platform.platform(),
        cpu=cpu_name(), cpu_logicos=os.cpu_count(),
        python=sys.version.split()[0], numpy=np.__version__,
        mpi4py=mpi4py.__version__, mpi=MPI.Get_library_version().strip(),
        mpiexec=mpiexec, blas_threads=1,
    )
    slurm = {k: v for k, v in os.environ.items()
             if k in ("SLURM_JOB_ID", "SLURM_JOB_PARTITION", "SLURM_NTASKS",
                      "SLURM_JOB_NODELIST", "SLURM_CPUS_ON_NODE", "SLURM_MEM_PER_NODE")}
    if slurm:
        info["slurm"] = slurm
    try:
        import sklearn
        info["sklearn"] = sklearn.__version__
    except ImportError:
        pass
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", nargs="+",
                    default=["versions", "strong", "weak-lin", "weak-work"],
                    choices=["versions", "strong", "weak-lin", "weak-work"])
    ap.add_argument("--versions", nargs="+", default=["v3", "dag"],
                    choices=["v3", "dag"])
    ap.add_argument("--s", type=int, nargs="+", default=S_FULL)
    ap.add_argument("--N", type=int, nargs="+", default=[50000, 100000, 200000],
                    help="tamaños para escalabilidad fuerte")
    ap.add_argument("--N0-lin", type=int, default=10000, help="N por proceso (weak-lin)")
    ap.add_argument("--N0-work", type=int, default=50000, help="N con s=1 (weak-work)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--out", default=RESULTS,
                    help="carpeta de resultados (p. ej. results/khipu)")
    ap.add_argument("--csv", default=None, help="por defecto <out>/results.csv")
    ap.add_argument("--timeout", type=int, default=3600, help="s por lanzamiento")
    ap.add_argument("--quick", action="store_true",
                    help="prueba rápida: s<=4, N pequeños, 1 rep, CSV aparte")
    args = ap.parse_args()

    out_dir = os.path.abspath(args.out)
    if args.quick:
        args.s, args.N, args.reps = [1, 2, 4], [20000], 1
        args.N0_lin, args.N0_work = 5000, 10000
        args.csv = os.path.join(out_dir, "results_quick.csv")
    # Ruta absoluta: los scripts hijos se ejecutan con cwd = src/.
    args.csv = os.path.abspath(args.csv or os.path.join(out_dir, "results.csv"))
    args.s = limit_s(args.s)

    os.makedirs(out_dir, exist_ok=True)
    mpiexec = find_mpiexec()
    info = system_info(mpiexec)
    with open(os.path.join(out_dir, "system_info.json"), "w", encoding="utf-8") as fh:
        json.dump(info, fh, indent=2, ensure_ascii=False)
    print(f"[runner] {info['cpu']} | {info['cpu_logicos']} hilos | {info['mpi']}")
    print(f"[runner] CSV -> {args.csv}")

    t0 = time.perf_counter()
    for exp in args.exp:
        print(f"\n=== {exp} ===")
        if exp == "versions":
            exp_versions(mpiexec, args.csv, args.reps, [s for s in args.s if s <= 8],
                         args.timeout)
        else:
            exp_scaling(mpiexec, args.csv, exp, args.versions, args.s, args.N,
                        args.N0_lin if exp == "weak-lin" else args.N0_work,
                        args.reps, args.timeout)
    print(f"\n[runner] terminado en {(time.perf_counter() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
