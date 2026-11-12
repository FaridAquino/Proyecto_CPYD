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
DAG del enunciado (`Scatterv` de entrenamiento, `Bcast` de consultas, fusión de candidatos).

| Resultado (N = 200 000) | Valor |
|---|---|
| Exactitud en `load_digits` (todas las versiones) | 0.9833 (100 % igual a scikit-learn) |
| Speedup máximo (Beta 3 / DAG) | 5.29 / 5.47 |
| Proceso óptimo teórico (modelo lineal) vs. medido | s\* = 11.8 vs. 12 |
| Fracción serial efectiva (Amdahl) | f_s = 0.140 |
| Rendimiento máximo | 66 GFLOP/s (Beta 3), 70 GFLOP/s (DAG) |

## Hardware y software

- **CPU:** Intel Core i7-14650HX — 16 núcleos (8 P-cores con SMT + 8 E-cores), 24 hilos, hasta 5.2 GHz
- **RAM:** 32 GB DDR5-5600
- **SO:** Windows 11 Home (64 bits)
- **Software:** Python 3.11.9, mpi4py 4.1.2 sobre Intel MPI 2021.19, NumPy 1.26.4

## Estructura

```
├── Informe_Final_Proyecto5_KNN.pdf   informe final (copia)
├── Proyecto_part1_CPD.pdf            informe parcial
├── requirements.txt
├── src/
│   ├── knn_digits_sec.py             código secuencial original (enunciado)
│   ├── knn_mpi_v1.py                 Beta 1: bcast/scatter/gather (pickle)
│   ├── knn_mpi_v2.py                 Beta 2: NumPy vectorizado + T_comp/T_comm
│   ├── knn_mpi_v3_optimized.py       Beta 3: Bcast/Scatterv/Gatherv con buffers (final)
│   ├── knn_mpi_dag.py                variante DAG (reparte X_train)
│   ├── data_generator.py             datos sintéticos con la estadística de load_digits
│   ├── benchmark_runner.py           barridos de escalabilidad fuerte/débil -> CSV
│   ├── extra_benchmarks.py           experimentos de afinidad y comunicación
│   ├── pin_bench.py                  carga fija con afinidad P-core / E-core / SMT
│   ├── comm_bench.py                 colectivas sin cómputo (alpha, beta)
│   ├── cpu_topology.py               topología P/E del procesador (Windows)
│   └── generate_plots.py             ajustes, figuras f1–f9 y tablas
├── results/                          mediciones (CSV), ajustes (JSON) y tablas LaTeX
├── figures/                          f1–f9 en PDF (vectorial) y PNG
└── latex/                            informe_final.tex, algoritmos_pram.tex, refs.bib, PDF
```

## Reproducción

Todos los comandos se ejecutan desde la raíz del repositorio. En Windows con Intel MPI use
`mpiexec`; en Linux, `mpirun`.

### 1. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 2. Generar los datos sintéticos

```bash
python src/data_generator.py --n 50000 100000 200000 --out data
```

Los datos son deterministas (`--seed 42`). `benchmark_runner.py` genera además los tamaños que
falten para la escalabilidad débil.

### 3. Ejecutar las versiones secuencial y MPI

```bash
python src/knn_digits_sec.py                                   # secuencial original
mpiexec -n 4  python src/knn_mpi_v1.py                         # Beta 1 (load_digits)
mpiexec -n 8  python src/knn_mpi_v2.py --data data/digits_N50000.npz
mpiexec -n 12 python src/knn_mpi_v3_optimized.py --data data/digits_N200000.npz --validate
mpiexec -n 12 python src/knn_mpi_dag.py          --data data/digits_N200000.npz --validate
```

Sin `--data` se usa `load_digits` original. `--validate` compara las predicciones con
`KNeighborsClassifier` de scikit-learn.

### 4. Reproducir los benchmarks y regenerar las figuras

```bash
python src/benchmark_runner.py --quick      # prueba rápida (~1 min)
python src/benchmark_runner.py              # campaña completa (~60 min) -> results/results.csv
python src/extra_benchmarks.py              # afinidad P/E/SMT + comunicación (~15 min)
python src/generate_plots.py                # figures/f1–f9, results/tables.tex, model_fit.json
```

El runner es reanudable y fija `OMP/MKL/OPENBLAS_NUM_THREADS=1` (un núcleo por proceso MPI).
Para mediciones estables: equipo conectado a la corriente, modo de máximo rendimiento y sin otras
aplicaciones abiertas.

### 4b. Ejecución en Khipu (clúster UTEC, SLURM)

Los resultados y figuras de Khipu se guardan aparte (`results/khipu/`, `figures/khipu/`) y no se
mezclan con los de la laptop.

```bash
module load <compilador> <mpi> <python>          # ver `module avail`
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-khipu.txt
pip install --no-binary=mpi4py mpi4py            # compilado contra el MPI del módulo

sbatch --ntasks=4 --time=00:10:00 run_khipu.sh   # prueba rápida
MODE=full sbatch run_khipu.sh                    # campaña completa + comm_bench
python src/generate_plots_khipu.py               # figures/khipu/k1–k9
```

`pin_bench.py` y `cpu_topology.py` (afinidad P/E-cores) son específicos de la laptop con Windows y
no se usan en Khipu.

### 5. Compilar el informe

```bash
cd latex
pdflatex informe_final
biber    informe_final
pdflatex informe_final
pdflatex informe_final
```

En Overleaf, suba las carpetas `latex/` y `figures/` manteniendo la estructura y compile con
pdfLaTeX (biber se ejecuta automáticamente).

## Uso de herramientas de IA

Se usó Claude (Anthropic), a través del entorno Claude Code, como herramienta de apoyo
(ver Sec. 5.3 del informe).
