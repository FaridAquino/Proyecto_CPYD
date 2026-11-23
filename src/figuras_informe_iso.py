"""
figuras_informe_iso.py -- figuras adicionales del informe final (experimento de escalabilidad).

Genera SOLO archivos nuevos en figures/khipu/isoeficiencia/ (no toca k2-k5, k8):
  k6_modelo.png       expresión teórica normalizada frente a lo medido:
                      (a) T_total(p) medido y teoría T = k1 W/p + T_comm(p) (Brent + árbol);
                      (b) eficiencia medida de las 6 series y eficiencia teórica.
  k7_tiempo_total.png tiempo de ejecución total T_total(p) de las 6 series.
Imprime: alpha y beta (árbol y lineal), fracciones de comunicación, errores de la
teoría y p óptimo que minimiza T_p (Sec. 6.10-6.11 del informe).

Entradas: results/khipu/isoeficiencia/results.csv
          results/khipu/aumento_train/comm_bench.csv (microbenchmark de colectivas)
Uso:      python src/figuras_informe_iso.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import nnls

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RES = os.path.join(ROOT, "results", "khipu", "isoeficiencia")
FIG = os.path.join(ROOT, "figures", "khipu", "isoeficiencia")
COMM = os.path.join(ROOT, "results", "khipu", "aumento_train", "comm_bench.csv")

C = dict(blue="#0072B2", orange="#D55E00", green="#009E73", gray="#6B6B6B", ink="#1A1A1A")
COLORS = [C["blue"], C["orange"], C["green"]]
VSTYLE = {"v3": ("o", "-"), "dag": ("s", "--")}
VLAB = {"v3": "Beta 3", "dag": "DAG"}
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "axes.titlepad": 8,
    "font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
    "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "lines.linewidth": 1.4, "lines.markersize": 4.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": C["gray"], "axes.linewidth": 0.6,
    "xtick.color": C["gray"], "ytick.color": C["gray"],
    "axes.labelcolor": C["ink"], "text.color": C["ink"],
    "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.5,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
PV = [1, 2, 4, 8, 16]
K = 3

# ------------------------------------------------------------------ datos
R = pd.read_csv(os.path.join(RES, "results.csv"))
M = R.groupby(["version", "n_train", "m_test", "s"], as_index=False).median(numeric_only=True)
M["fijo"] = np.where(M.version == "v3", M.n_train, M.m_test).astype(int)
M["W"] = 3.0 * M.m_test * M.n_train * M.d
FIJOS = sorted(M.fijo.unique())

# ------------------------------------------------------------------ alpha, beta (U6.1)
CB = pd.read_csv(COMM)
AB = {}
for op in ("Bcast", "Scatter", "Gather"):
    x = CB[CB.op == op]
    s, Mw, t = x.s.values.astype(float), x.M.values.astype(float), x.t_med.values
    L = np.log2(s)
    modelos = {"arbol": np.c_[L, Mw * (L if op == "Bcast" else 1)],
               "lineal": np.c_[s - 1, Mw * ((s - 1) if op == "Bcast" else 1)]}
    for nombre, A in modelos.items():
        (a, b), _ = nnls(A, t)
        AB[(op, nombre)] = (a, b)
        print(f"{op:7s} {nombre:6s} alpha={a:.2e} s  beta={b:.3e} s/palabra")
aG, bG = AB[("Gather", "arbol")]
bB = AB[("Bcast", "arbol")][1]
bS = AB[("Scatter", "arbol")][1]


def comm_teo(v, p, n, m, d):
    """T_comm teórico con colectivas en árbol (Ecs. 8-10 y 21 del informe)."""
    p = np.asarray(p, float)
    L = np.log2(p)
    on = (p > 1).astype(float)
    if v == "v3":   # Bcast(X_tr, y_tr) + Scatterv(X_te) + Gatherv(y)
        t = L * n * (d + 1) * bB + m * d * bS + aG * L + m * bG
    else:           # Scatterv(X_tr, y_tr) + Bcast(X_te) + Gather(k candidatos x 2)
        t = n * (d + 1) * bS + L * m * d * bB + aG * L + 2 * m * K * p * bG
    return on * t


def serie(v, f):
    return M[(M.version == v) & (M.fijo == f)].sort_values("s").reset_index(drop=True)


def teo(v, d):
    """Expresión teórica normalizada: k1 = T_comp(1)/W(1) medido en p=1 de la serie."""
    k1 = d.T_comp.iloc[0] / d.W.iloc[0]
    return k1 * d.W.values / d.s.values + comm_teo(v, d.s, d.n_train.values, d.m_test.values, d.d.values)


def eff(d, T):
    """E = Ts(n)/(p Tp(n,p)) con Ts(n) = T(1) W(n)/W(n_1)  (W = Ts, U6.2)."""
    return (d.W.values / d.W.iloc[0]) * T[0] / (d.s.values * T)


# ---------------------------------------------- resúmenes para el texto
M["frac_comm"] = 100 * M.T_comm / M.T_total
M["frac_crit"] = 100 * (M.T_total - M.T_comp) / M.T_total
print("\nT_comm/T_total [%]:\n", M.pivot_table(index=["version", "fijo"], columns="s",
                                               values="frac_comm").round(1))
print("max (T_total-T_comp)/T_total [%]:", M.frac_crit.max().round(2))
for v in ("v3", "dag"):
    err = {p: [] for p in PV}
    for f in FIJOS:
        d = serie(v, f)
        for p, e in zip(d.s, (teo(v, d) - d.T_total) / d.T_total):
            err[int(p)].append(100 * e)
    print(v, "error teoría vs T_total [%] por p:",
          {p: (round(min(e)), round(max(e))) for p, e in err.items()})

# p óptimo (U6.2): minimizar T_p(p) = k1 W/p + c g(p); árbol g = log2 p, lineal g = p - 1
d3 = serie("v3", max(FIJOS))
k1 = float(d3.T_comp.iloc[0] / d3.W.iloc[0])
last = d3.iloc[-1]
W, n, m, dd = float(last.W), float(last.n_train), float(last.m_test), float(last.d)
Mp = n * (dd + 1) + m * dd + m
for nombre in ("arbol", "lineal"):
    a, b = AB[("Bcast", nombre)]
    c = a + Mp * b
    pstar = k1 * W * np.log(2) / c if nombre == "arbol" else np.sqrt(k1 * W / c)
    print(f"p* {nombre}: k1={k1:.2e} W={W:.2e} M={Mp:.2e} c={c:.4f} s -> p*={pstar:.0f}")


def ticks(ax):
    ax.set_xticks(PV)
    ax.set_xlim(0.5, 16.5)
    ax.set_xlabel("procesos $p$")


def lab(v, f):
    return (rf"{VLAB[v]}, $n$={f//1000}k, $m$=1000$\cdot p\log_2 p$" if v == "v3"
            else rf"{VLAB[v]}, $m$={f//1000}k, $n$=1000$\cdot p\log_2 p$")


# ---------------------------------------------- k6: normalización
fig, (axA, axB) = plt.subplots(1, 2, figsize=(7.16, 3.1))
f = max(FIJOS)
for v, col in (("v3", C["blue"]), ("dag", C["orange"])):
    d = serie(v, f)
    mk, _ = VSTYLE[v]
    axA.plot(d.s, d.T_total, marker=mk, ls="none", color=col, mfc=col if v == "v3" else "white",
             label=f"{VLAB[v]} medido ({'n' if v == 'v3' else 'm'}={f//1000}k)", zorder=3)
    axA.plot(d.s, teo(v, d), color=col, ls=":", lw=1.3, label=f"{VLAB[v]}: teoría normalizada")
axA.set_yscale("log")
axA.set_ylabel(r"$T_{\mathrm{total}}$ [s] (log)")
axA.set_title("(a) Tiempo total: medido vs. teoría normalizada")
ticks(axA)
axA.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6)

for v, col in (("v3", C["blue"]), ("dag", C["orange"])):
    mk, _ = VSTYLE[v]
    for fi, c2 in zip(FIJOS, COLORS):
        d = serie(v, fi)
        axB.plot(d.s, eff(d, d.T_total.values), marker=mk, ls="none", color=c2,
                 mfc=c2 if v == "v3" else "white")
    et = np.mean([eff(serie(v, fi), teo(v, serie(v, fi))) for fi in FIJOS], axis=0)
    axB.plot(PV, et, color=col, ls=":", lw=1.3, label=f"{VLAB[v]}: teoría normalizada")
axB.plot([], [], marker="o", ls="none", color=C["gray"], label="Beta 3 medido (10k/20k/40k)")
axB.plot([], [], marker="s", ls="none", color=C["gray"], mfc="white", label="DAG medido (10k/20k/40k)")
axB.set_ylim(0, 1.1)
axB.set_ylabel("eficiencia $E(p)$")
axB.set_title("(b) Eficiencia: medida vs. teoría")
ticks(axB)
axB.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=6)
fig.tight_layout(pad=1.2)
fig.savefig(os.path.join(FIG, "k6_modelo.png"), dpi=300)
plt.close(fig)

# ---------------------------------------------- k7: tiempo total
fig, ax = plt.subplots(figsize=(3.5, 2.6))
for v in ("v3", "dag"):
    mk, ls = VSTYLE[v]
    for fi, col in zip(FIJOS, COLORS):
        d = serie(v, fi)
        ax.plot(d.s, d.T_total, marker=mk, ls=ls, color=col,
                mfc=col if v == "v3" else "white", label=lab(v, fi))
ax.set_yscale("log")
ax.set_ylabel(r"$T_{\mathrm{total}}$ [s] (log)")
ax.set_title("Tiempo de ejecución total")
ticks(ax)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=2)
fig.tight_layout(pad=1.2)
fig.savefig(os.path.join(FIG, "k7_tiempo_total.png"), dpi=300)
plt.close(fig)
print("\nfiguras: k6_modelo.png, k7_tiempo_total.png en", os.path.relpath(FIG, ROOT))
