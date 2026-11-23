# Proyecto 5 — Paralelización del clasificador KNN sobre `load_digits` con MPI (mpi4py)

**CS4052 – Computación Paralela y Distribuida (2026-II)** · Universidad de Ingeniería y Tecnología (UTEC)
**Profesor:** José Fiestas, PhD

| Integrante | Participación |
|---|---|
| Juan David Velo Poma | 100 % |
| Farid Jack Aquino Castro | 100 % |
| Josue Nehemias Velo Poma | 100 % |

Informe final: [`Informe_Final_Proyecto5_KNN.pdf`](Informe_Final_Proyecto5_KNN.pdf)

## Resumen

KNN paralelizado con MPI en paradigma SPMD: `Bcast` del conjunto de entrenamiento, `Scatterv` de
las consultas y `Gatherv` de las predicciones (PRAM CREW, formalismo `pardo`), más la variante del
DAG del enunciado (`Scatterv` del entrenamiento, `Bcast` de las consultas y fusión de candidatos).
Mediciones en el clúster Khipu (nodo con 2 × Intel Xeon Gold 6130, hasta 16 núcleos físicos).

| Resultado | Valor |
|---|---|
| Exactitud en `load_digits` (todas las versiones, cualquier p) | 0.9833 (100 % igual a scikit-learn) |
| Exactitud en datos sintéticos, n = 1k → 64k | 0.989 → 0.998 |
| Beta 3 vs. secuencial original (`load_digits`, p = 8) | 967× más rápida |
| Speedup máximo (problema creciente) | 3.2 (Beta 3, p = 8), 2.7 (DAG) |
| Eficiencia con p = 16 | 0.16–0.19 |
| Comunicación en el camino crítico | ≤ 4.5 % del tiempo total |
| Rendimiento agregado | 14–19 GFLOP/s con p = 1; se satura en ~50 GFLOP/s |
| Número óptimo de procesos | teórico (mín. T_p) 65–660; medido p\* = 8 (Beta 3), 4–8 (DAG) |

## Estructura

```
├── Informe_Final_Proyecto5_KNN.pdf   informe final
├── latex/                            informe_final.tex, refs.bib, informe_final.pdf
├── src/
│   ├── knn_digits_sec.py             código secuencial original (enunciado)
│   ├── knn_mpi_v1.py                 Beta 1: comm.bcast/scatter/gather (pickle)
│   ├── knn_mpi_v2.py                 Beta 2: cómputo vectorizado + T_comp/T_comm
│   ├── knn_mpi_v3_optimized.py       Beta 3: Bcast/Scatterv/Gatherv con buffers (final)
│   ├── knn_mpi_dag.py                variante DAG (reparte X_train)
│   ├── data_generator.py             datos sintéticos con la estadística de load_digits
│   ├── benchmark_runner.py           lanza los experimentos y escribe results.csv
│   ├── comm_bench.py, extra_benchmarks.py   microbenchmark de colectivas (alpha, beta)
│   ├── generate_plots_khipu.py       figuras k2–k5 y k8
│   └── figuras_informe_iso.py        figuras k6 (teoría normalizada) y k7 (tiempo total)
├── results/khipu/
│   ├── isoeficiencia/                mediciones del experimento principal
│   └── aumento_train/                registro de versiones (load_digits) y comm_bench.csv
├── figures/khipu/isoeficiencia/      k2–k8 (.png)
└── run_khipu.sh                      script SLURM para Khipu
```

## Reproducción en Khipu (SLURM)

```bash
module load gnu12 openmpi4 python3
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-khipu.txt
pip install --no-binary=mpi4py mpi4py            # compilado contra el MPI del módulo

MODE=iso  sbatch run_khipu.sh                    # experimento principal -> results/khipu/isoeficiencia
MODE=comm sbatch run_khipu.sh                    # microbenchmark de colectivas (alpha, beta)
python src/generate_plots_khipu.py --modo iso    # figuras k2–k5, k8
python src/figuras_informe_iso.py                # figuras k6, k7
```

Cada proceso MPI ocupa un núcleo físico (`--cpus-per-task=2`, `--bind-to core`) y
`OMP/MKL/OPENBLAS_NUM_THREADS=1`. Cada lanzamiento descarta una ejecución de calentamiento y mide
5 repeticiones (se reporta la mediana).

Ejecución individual de las versiones (con `--validate` se comparan las predicciones con scikit-learn):

```bash
python src/data_generator.py --n 50000 --out data
python src/knn_digits_sec.py
mpiexec -n 4 python src/knn_mpi_v1.py
mpiexec -n 8 python src/knn_mpi_v2.py --data data/digits_N50000.npz
mpiexec -n 8 python src/knn_mpi_v3_optimized.py --data data/digits_N50000.npz --validate
mpiexec -n 8 python src/knn_mpi_dag.py          --data data/digits_N50000.npz --validate
```

## Compilar el informe

```bash
cd latex
pdflatex informe_final && biber informe_final && pdflatex informe_final && pdflatex informe_final
```

En Overleaf, suba `latex/` y `figures/` manteniendo la estructura y compile con pdfLaTeX.

## Uso de herramientas de IA

Se usó Claude (Anthropic), a través de Claude Code, como herramienta de apoyo (Sec. 8.3 del informe).
