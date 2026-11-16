#!/bin/bash
#SBATCH --job-name=knn-mpi
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out
#
# Benchmarks de KNN-MPI en Khipu. Enviar SIEMPRE desde la raíz del repositorio.
#
#   sbatch --ntasks=4 --time=00:10:00 run_khipu.sh      # prueba rápida (MODE=quick)
#   MODE=full sbatch run_khipu.sh                       # campaña completa
#   MODE=comm sbatch run_khipu.sh                       # sólo alpha/beta
#   MODE=test sbatch run_khipu.sh                       # aumento de test (n_train fijo)
#
# Resultados: MODE=quick|full|comm -> results/khipu/aumento_train
#             MODE=test            -> results/khipu/aumento_test
#
# Las opciones de la línea de comandos de sbatch (--ntasks, --time, --partition,
# --mem ...) tienen prioridad sobre las cabeceras #SBATCH de este archivo.
#
# Variables opcionales (se pasan igual que MODE):
#   KHIPU_MODULES  módulos a cargar, p. ej. "gnu12 openmpi4 python3"
#   S_LIST         procesos a barrer (se recorta a --ntasks automáticamente)
#   N_LIST         tamaños para strong scaling
#   EXPS           experimentos: versions strong weak-lin weak-work
#   REPS           repeticiones por configuración
#   N0_LIN         N por proceso en weak-lin (bajar si falta memoria)
#   N_TRAIN        n_train fijo en MODE=test (por defecto 50000)
#   M_LIST         tamaños de test para strong-test (por defecto "10000 20000 40000")
#   M0_TEST        m por proceso en weak-test (por defecto 10000)
set -eo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")}"

# Módulos: Khipu carga mpich por defecto en cada sesión y Lmod sólo admite un MPI,
# así que se intercambia aquí para no depender de la sesión desde la que se envía.
KHIPU_MPI=${KHIPU_MPI:-openmpi4/4.1.6}
KHIPU_PYTHON=${KHIPU_PYTHON:-python3/3.11.11}
if type module >/dev/null 2>&1; then
    module swap mpich "$KHIPU_MPI" 2>/dev/null || module load "$KHIPU_MPI"
    module load "$KHIPU_PYTHON"
    if [ -n "${KHIPU_MODULES:-}" ]; then
        module load $KHIPU_MODULES
    fi
fi
source .venv/bin/activate
python -c "from mpi4py import MPI; print('[khipu] MPI:', MPI.Get_library_version().split(',')[0])"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1

MODE=${MODE:-quick}
OUT=results/khipu/aumento_train
OUT_TEST=results/khipu/aumento_test
S_LIST=${S_LIST:-"1 2 4 8 16 32"}
N_LIST=${N_LIST:-"50000 100000 200000"}
EXPS=${EXPS:-"versions strong weak-lin weak-work"}
REPS=${REPS:-3}
N0_LIN=${N0_LIN:-10000}
N_TRAIN=${N_TRAIN:-50000}
M_LIST=${M_LIST:-"10000 20000 40000"}
M0_TEST=${M0_TEST:-10000}

echo "[khipu] job=${SLURM_JOB_ID:-local} nodos=${SLURM_JOB_NODELIST:-?} ntasks=${SLURM_NTASKS:-?} modo=$MODE"
echo "[khipu] python=$(which python)  mpiexec=$(which mpiexec 2>/dev/null || which mpirun)"

case "$MODE" in
    quick)
        python -u src/benchmark_runner.py --quick --out "$OUT"
        ;;
    full)
        python -u src/benchmark_runner.py --out "$OUT" --exp $EXPS --s $S_LIST \
               --N $N_LIST --reps "$REPS" --N0-lin "$N0_LIN"
        python -u src/extra_benchmarks.py --exp comm --out "$OUT" --s $S_LIST
        ;;
    comm)
        python -u src/extra_benchmarks.py --exp comm --out "$OUT" --s $S_LIST
        ;;
    test)
        python -u src/benchmark_runner.py --out "$OUT_TEST" --exp strong-test weak-test                --s $S_LIST --n-train "$N_TRAIN" --M $M_LIST --M0-test "$M0_TEST"                --reps "$REPS"
        ;;
    *)
        echo "MODE desconocido: $MODE (use quick | full | comm | test)"; exit 1
        ;;
esac
echo "[khipu] terminado: $(date)"
