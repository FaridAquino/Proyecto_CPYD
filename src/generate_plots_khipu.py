"""
generate_plots_khipu.py -- Gráficas de las mediciones hechas en Khipu (UTEC).

Independiente de generate_plots.py (laptop i7-14650HX): no usa pinning.csv ni las
franjas P-cores/E-cores/SMT, y toma los valores de s directamente de los datos.

Dos modos (--modo), cada uno con su carpeta de resultados y de figuras:
  train  (por defecto) aumenta el ENTRENAMIENTO: N crece, split 80/20.
         results/khipu/aumento_train/ -> figures/khipu/aumento_train/
         tags: versions, strong, weak-work, weak-lin; tamaño de la serie = N.
  test   aumenta el TEST: n_train = 50k fijo y m_test crece.
         results/khipu/aumento_test/  -> figures/khipu/aumento_test/
         tags: strong-test, weak-test; tamaño de la serie = m.

Entradas  results.csv; comm_bench.csv (k6 y k9). En modo test, si la carpeta no
          tiene comm_bench.csv se usa el de aumento_train (--comm): la
          comunicación no depende del dataset.
Salidas   k2_tiempos, k3_speedup, k4_eficiencia, k5_gflops, k6_modelo,
          k9_comunicacion (sólo modo train) (.png, nunca .pdf);
          model_fit.json y tables.tex en la carpeta de resultados.

Todas las figuras usan el número de procesos p en el eje x (la columna del CSV se
sigue llamando 's'). Sólo K4 compara Beta 3 con la variante DAG.
k3: "Speedup" de la Beta 3, una recta por tamaño fijo (N o m).
k2 y k5 tienen hasta dos paneles (se omite el que no tenga datos):
  (a) versiones sobre load_digits (tag 'versions'): Beta 1, Beta 2 y Beta 3
      (+ el secuencial original en la de tiempos). Sólo llega a p = 8: el runner
      limita este experimento a s <= 8 (benchmark_runner.py) porque load_digits
      tiene m = 360 consultas; con p = 32 cada proceso recibiría ~11 y el tiempo
      (~ms) quedaría dominado por el arranque/latencia de MPI, no por el cómputo;
  (b) escalabilidad fuerte de la Beta 3, una serie por tamaño, p <= 32.
k4: "Eficiencia" (strong), Beta 3 vs DAG: una recta por tamaño fijo (N o m).
k6: modelo T_p(p) = k1 W/p + k2 T_comm(p) ajustado a la Beta 3 (strong) y p* óptimo.
Escalabilidad débil: no se grafica; E_w(p) se guarda en model_fit.json ("weak").
k9: microbenchmark de MPI_Bcast / MPI_Gather -> alpha, beta.

Uso:
    python src/generate_plots_khipu.py
    python src/generate_plots_khipu.py --modo test
    python src/generate_plots_khipu.py --results <dir> --figures <dir>

Speedup relativo a la propia versión: S(p) = T(1)/T(p); E(p) = S(p)/p.
Ajuste de Amdahl S = 1/(f_s + (1 - f_s)/p) sobre la Beta 3 (strong scaling).
"""
import argparse
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, nnls

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--modo", choices=["train", "test"], default="train",
                help="train: aumenta el entrenamiento; test: aumenta el test")
ap.add_argument("--results", default=None, help="por defecto results/khipu/aumento_<modo>")
ap.add_argument("--figures", default=None, help="por defecto figures/khipu/aumento_<modo>")
ap.add_argument("--comm", default=None,
                help="comm_bench.csv (por defecto el de --results o, si falta, el de aumento_train)")
args = ap.parse_args()
MODO = args.modo
RES = os.path.abspath(args.results or os.path.join(ROOT, "results", "khipu", f"aumento_{MODO}"))
FIG = os.path.abspath(args.figures or os.path.join(ROOT, "figures", "khipu", f"aumento_{MODO}"))
os.makedirs(FIG, exist_ok=True)

# --------------------------------------------------------------------- estilo
C = dict(blue="#0072B2", orange="#D55E00", green="#009E73", pink="#CC79A7",
         gray="#6B6B6B", ink="#1A1A1A", light="#BDBDBD")
SERIES = [(C["blue"], "o", "-"), (C["orange"], "s", "--"),
          (C["green"], "^", "-."), (C["pink"], "D", ":")]
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "axes.unicode_minus": True, "axes.titlepad": 8,
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
W1, W2 = 3.5, 7.16
BETAS = [("v1", "Beta 1"), ("v2", "Beta 2"), ("v3", "Beta 3")]


def save(fig, name):
    fig.tight_layout(pad=1.2)
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300)
    plt.close(fig)
    print(f"  {rel(os.path.join(FIG, name))}.png")


def rel(path):
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:                       # otra unidad de disco (Windows)
        return path


def skip(name, why):
    print(f"  [omitida] {name}: {why}")


# --------------------------------------------------------------------- datos
csv_path = os.path.join(RES, "results.csv")
if not os.path.exists(csv_path):
    sys.exit(f"No existe {csv_path}. Ejecute primero los benchmarks en Khipu.")
R = pd.read_csv(csv_path)
MED = R.groupby(["version", "tag", "N", "s"], as_index=False).median(numeric_only=True)
cb_path = args.comm or os.path.join(RES, "comm_bench.csv")
if not os.path.exists(cb_path) and MODO == "test":
    cb_path = os.path.join(ROOT, "results", "khipu", "aumento_train", "comm_bench.csv")
CB = pd.read_csv(cb_path) if os.path.exists(cb_path) else None
hosts = ", ".join(sorted(R.host.dropna().astype(str).unique()))
print(f"[khipu] modo={MODO} | {len(R)} filas | hosts: {hosts}")
fit = dict(hosts=hosts, modo=MODO)

# Configuración por modo: tag del strong, columna que da el tamaño de la serie,
# símbolo en las leyendas y experimentos de escalabilidad débil.
if MODO == "train":
    TAG_STRONG, SIZE, SYM = "strong", "N", "N"
    WEAK_TAGS = [("weak-work", r"$N=N_0\sqrt{p}$"), ("weak-lin", r"$N=N_0\,p$")]
else:
    TAG_STRONG, SIZE, SYM = "strong-test", "m_test", "m"
    WEAK_TAGS = [("weak-test", r"$m=m_0\,p$")]
VERS = [("v3", "Beta 3"), ("dag", "DAG")]

V = MED[MED.tag == "versions"]                       # load_digits: seq, Beta 1-3
VB = [(v, lab) for v, lab in BETAS if not V[V.version == v].empty]
S_VERS = sorted(int(s) for s in V[V.version.isin([v for v, _ in VB])].s.unique())
ST = MED[(MED.tag == TAG_STRONG) & (MED.version == "v3")]
N_STRONG = sorted(int(n) for n in ST[SIZE].unique())
S_STRONG = sorted(int(s) for s in ST.s.unique())
NMAX = max(N_STRONG) if N_STRONG else None


def sticks(ax, svals):
    """Marcas en los s medidos; se omiten etiquetas que chocarían entre sí."""
    svals = sorted(svals)
    top = max(svals)
    major = [s for s in svals if s == 1 or s >= top / 12]
    ax.set_xticks(major)
    ax.set_xticks([s for s in svals if s not in major], minor=True)
    ax.set_xlim(1 - 0.03 * top, top * 1.03)


def versions(v):
    return V[V.version == v].sort_values("s").reset_index(drop=True)


def strong(N, version="v3"):
    """Strong scaling de `version` para el tamaño N (N o m según el modo)."""
    d = MED[(MED.tag == TAG_STRONG) & (MED.version == version) & (MED[SIZE] == N)]
    return d.sort_values("s").reset_index(drop=True)


def kfmt(x):
    return f"{round(x / 1000):g}k" if x >= 10000 else f"{x / 1000:g}k"


def nlab(N):
    return f"${SYM}$={kfmt(N)}"


def skey(v, N):
    return f"{v}_{SYM}{N}"


def r2(y, pred):
    den = np.sum((y - np.mean(y)) ** 2)
    return float(1 - np.sum((y - pred) ** 2) / den) if den > 0 else float("nan")


def amdahl(s, fs):
    return 1 / (fs + (1 - fs) / s)


def panels(name, draw_a, draw_b, height=2.6, **kw):
    """Figura con el panel (a) versiones y/o (b) strong, según haya datos."""
    todo = [f for f, ok in ((draw_a, bool(VB)), (draw_b, bool(N_STRONG))) if ok]
    if not todo:
        return skip(name, "no hay filas con tag 'versions' ni 'strong'")
    fig, axes = plt.subplots(1, len(todo), figsize=(W2 if len(todo) > 1 else W1, height),
                             squeeze=False, **kw)
    for ax, f in zip(axes[0], todo):
        f(ax)
        ax.set_xlabel("procesos $p$")
    save(fig, name)


T_DIGITS = "load_digits ($n$=1437, $m$=360, $p\\leq 8$)"
T_STRONG = "Beta 3, strong scaling" + ("" if MODO == "train" else ", $n$=50k fijo")

# =========================================================================
# Ajustes: speedup por versión (load_digits) y Amdahl de la Beta 3 (strong)
# =========================================================================
def speedup_fit(d, with_amdahl=False):
    """S y E relativos al s=1 de la misma serie; None si falta s=1."""
    if len(d) < 2 or int(d.s.iloc[0]) != 1:
        return None
    T1 = float(d.T_total.iloc[0])
    S = T1 / d.T_total.values
    p = dict(T1=T1, s=d.s.astype(int).tolist(), S=S.tolist(),
             E=(S / d.s.values).tolist(), S_max_medido=float(S.max()),
             s_best_medido=int(d.s[np.argmin(d.T_total.values)]))
    if with_amdahl and len(d) >= 3:
        (fs,), cov = curve_fit(amdahl, d.s.values, S, p0=[0.05], bounds=(0, 1))
        p.update(amdahl_fs=float(fs), amdahl_fs_err=float(np.sqrt(cov[0, 0])),
                 amdahl_R2=r2(S, amdahl(d.s.values, fs)),
                 amdahl_Smax=float(1 / fs) if fs > 0 else float("inf"))
    return p


fit["speedup"] = {}
seq = V[V.version == "seq"]
if not seq.empty:
    fit["t_seq"] = float(seq.T_total.iloc[0])
for v, _ in VB:
    p = speedup_fit(versions(v))
    if p:
        fit["speedup"][f"{v}_digits"] = p
for v, _ in VERS:
    for N in N_STRONG:
        p = speedup_fit(strong(N, v), with_amdahl=True)
        if p:
            fit["speedup"][skey(v, N)] = p
        else:
            print(f"  [aviso] {v} {SYM}={N}: falta s=1 o hay <2 puntos; sin speedup")


def weak_series(version, tag):
    """Escalabilidad débil: (d, S_esc, E_w) o None. S_esc = W(p)/W(1)·T(1)/T(p)."""
    d = MED[(MED.version == version) & (MED.tag == tag)].sort_values("s").reset_index(drop=True)
    if len(d) < 2 or int(d.s.iloc[0]) != 1:
        return None
    W = (3 * d.m_test * d.n_train * d.d).values
    Sg = (W / W[0]) * float(d.T_total.iloc[0]) / d.T_total.values
    return d, Sg, Sg / d.s.values


def weak_label(version_lab, tag_lab, d):
    return f"{version_lab}, {tag_lab} ({kfmt(d[SIZE].iloc[0])}→{kfmt(d[SIZE].iloc[-1])})"


def sp(key):
    return fit["speedup"].get(key)


# =========================================================================
# k2: tiempos
# =========================================================================
def tiempos_a(ax):
    if "t_seq" in fit:
        ax.axhline(fit["t_seq"], color=C["gray"], lw=1, ls=":", label="secuencial original")
    for (v, lab), (col, mk, ls) in zip(VB, SERIES):
        d = versions(v)
        ax.plot(d.s, d.T_total, color=col, marker=mk, ls=ls, label=lab)
    ax.set_yscale("log"); sticks(ax, S_VERS)
    ax.set_ylabel(r"$T_{\mathrm{total}}$ [s] (log)")
    ax.set_title(f"Tiempo: {T_DIGITS}")
    ax.legend(loc="center right", bbox_to_anchor=(1.0, 0.45))


def tiempos_b(ax):
    for N, (col, mk, ls) in zip(N_STRONG, SERIES):
        d = strong(N)
        ax.plot(d.s, d.T_total, color=col, marker=mk, ls=ls, label=nlab(N))
    ax.set_yscale("log"); sticks(ax, S_STRONG)
    ax.set_ylabel(r"$T_{\mathrm{total}}$ [s] (log)")
    ax.set_title(f"Tiempo: {T_STRONG}")
    ax.legend(loc="upper right", ncol=len(N_STRONG))


panels("k2_tiempos", tiempos_a, tiempos_b)


# =========================================================================
# k3: speedup de la Beta 3 (strong scaling), una recta por tamaño fijo (N o m)
#   (el ajuste de Amdahl se sigue guardando en model_fit.json, no se dibuja)
# =========================================================================
if not N_STRONG:
    skip("k3_speedup", "no hay datos de strong scaling")
else:
    fig, ax = plt.subplots(figsize=(W1, 2.6))
    top_s, top_S = max(S_STRONG), 1.0
    for N, (col, mk, ls) in zip(N_STRONG, SERIES):
        q = sp(skey("v3", N))
        if q:
            ax.plot(q["s"], q["S"], color=col, marker=mk, ls=ls, label=nlab(N))
            top_S = max(top_S, max(q["S"]))
    ax.plot([1, top_s], [1, top_s], color=C["light"], lw=1, zorder=1)     # ideal S = p
    ax.set_ylim(0, min(top_s, top_S * 2.0) * 1.05); sticks(ax, S_STRONG)
    ax.set_xlabel("procesos $p$"); ax.set_ylabel("speedup")
    ax.set_title("Speedup")
    ax.legend(loc="upper left")
    save(fig, "k3_speedup")


# =========================================================================
# k4: eficiencia, Beta 3 vs DAG (strong scaling: cada recta tiene N o m fijo)
#   color = tamaño; Beta 3 = línea continua y círculo, DAG = discontinua y cuadrado
# =========================================================================
VSTYLE = {"v3": ("o", "-"), "dag": ("s", "--")}
COLORS = [C["blue"], C["orange"], C["green"], C["pink"]]

if not N_STRONG:
    skip("k4_eficiencia", "no hay datos de strong scaling")
else:
    fig, ax = plt.subplots(figsize=(W1 + 0.4, 3.0))
    pv, emax = set(), 1.0
    fijo = "" if MODO == "train" else "$n$=50k, "      # modo test: el train también es fijo
    for v, vlab in VERS:
        mk, ls = VSTYLE[v]
        for N, col in zip(N_STRONG, COLORS):
            q = sp(skey(v, N))
            if q:
                ax.plot(q["s"], q["E"], color=col, marker=mk, ls=ls,
                        label=f"{vlab}, {fijo}{nlab(N)}", mfc=col if v == "v3" else "white")
                pv |= set(q["s"]); emax = max(emax, max(q["E"]))
    ax.axhline(1, color=C["light"], lw=1, zorder=1)
    ax.set_ylim(0, emax * 1.08); sticks(ax, sorted(pv))
    ax.set_xlabel("procesos $p$")
    ax.set_ylabel("eficiencia $E(p)=S(p)/p$")
    ax.set_title("Eficiencia")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    save(fig, "k4_eficiencia")


# =========================================================================
# k5: GFLOP/s = 3 m n d / T_comp
# =========================================================================
def gflops_a(ax):
    for (v, lab), (col, mk, ls) in zip(VB, SERIES):
        d = versions(v)
        g = d.gflops
        if g.isna().all():                   # la Beta 1 no mide T_comp: se usa T_total
            g, lab = d.flops / d.T_total / 1e9, lab + r" (con $T_{\mathrm{total}}$)"
        ax.plot(d.s, g, color=col, marker=mk, ls=ls, label=lab)
    ax.set_yscale("log"); sticks(ax, S_VERS)
    ax.set_ylabel(r"GFLOP/s  ($3mnd\,/\,T_{\mathrm{comp}}$) (log)")
    ax.set_title(f"GFLOP/s: {T_DIGITS}")
    ax.legend(loc="center right")


def gflops_b(ax):
    d1 = strong(NMAX)
    if int(d1.s.iloc[0]) == 1:
        g1 = float(d1.gflops.iloc[0])
        ax.plot([1, d1.s.max()], [g1, g1 * d1.s.max()], color=C["light"], lw=1,
                label=f"ideal: {g1:.1f}·$p$", zorder=1)
    gmax = 0.0
    for N, (col, mk, ls) in zip(N_STRONG, SERIES):
        d = strong(N)
        ax.plot(d.s, d.gflops, color=col, marker=mk, ls=ls, label=nlab(N))
        gmax = max(gmax, float(d.gflops.max()))
    fit["gflops_max"] = gmax
    ax.set_ylim(0, gmax * 1.35); sticks(ax, S_STRONG)
    ax.set_ylabel(r"GFLOP/s  ($3mnd\,/\,T_{\mathrm{comp}}$)")
    ax.set_title(f"GFLOP/s: {T_STRONG}")
    ax.legend(loc="upper left", ncol=2)


panels("k5_gflops", gflops_a, gflops_b)

# =========================================================================
# k9: microbenchmark de comunicación -> alpha, beta
# =========================================================================
def fit_comm():
    out = {}
    for op in ("Bcast", "Scatter", "Gather"):
        x = CB[CB.op == op]
        if x.s.nunique() < 2 or len(x) < 4:
            return None
        s, M, t = x.s.values.astype(float), x.M.values, x.t_med.values
        L = np.log2(s)
        models = {"arbol": np.c_[L, M * (L if op == "Bcast" else 1)],
                  "lineal": np.c_[s - 1, M * ((s - 1) if op == "Bcast" else 1)]}
        out[op] = {}
        for name, A in models.items():
            coef, _ = nnls(A, t)
            out[op][name] = dict(alpha=coef[0], beta=coef[1], R2=r2(t, A @ coef),
                                 GBps=8 / coef[1] / 1e9 if coef[1] > 0 else None)
    return out


comm = fit_comm() if CB is not None else None
if comm:
    fit["comm"] = comm
    fit["comm_csv"] = rel(cb_path)
if comm and MODO == "train":
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.3))
    Mbig = CB.M.max()
    for ax, op in zip(axes, ("Bcast", "Gather")):
        x = CB[(CB.op == op) & (CB.M >= 0.99 * Mbig)].sort_values("s")
        ax.plot(x.s, x.t_med * 1e3, color=C["ink"], marker="o", ls="none",
                label="medido", zorder=3)
        ss = np.linspace(2, x.s.max(), 200)
        for name, col, ls in (("arbol", C["blue"], "--"), ("lineal", C["orange"], "-")):
            q = comm[op][name]
            g = np.log2(ss) if name == "arbol" else ss - 1
            gl = r"\log_2 p" if name == "arbol" else "(p-1)"
            if op == "Bcast":
                y, lab = g * (q["alpha"] + Mbig * q["beta"]), rf"${gl}\,(\alpha+M\beta)$"
            else:
                y, lab = q["alpha"] * g + Mbig * q["beta"], rf"$\alpha\,{gl}+M\beta$"
            kind = "árbol" if name == "arbol" else "lineal"
            ax.plot(ss, y * 1e3, color=col, ls=ls, label=f"{kind}: {lab}  $R^2$={q['R2']:.3f}")
        ax.set_title(f"MPI_{op}, $M$ = {Mbig / 1e6:.1f}M palabras ({8 * Mbig / 2**20:.0f} MiB)")
        ax.set_xlabel("procesos $p$"); ax.set_ylabel("tiempo [ms]")
        sticks(ax, x.s.astype(int).tolist())
        ax.legend(loc="upper left" if op == "Bcast" else "lower right")
    save(fig, "k9_comunicacion")
elif not comm:
    skip("k9_comunicacion / k6_modelo", "falta comm_bench.csv (extra_benchmarks.py --exp comm)")


# =========================================================================
# k6: normalización T_p(p) = k1 W/p + k2 T_comm(p) y p* óptimo (Beta 3, strong)
# =========================================================================
def comm_model(kind, s, n, m, dd):
    q = comm["Bcast"][kind]
    M = n * (dd + 1) + m * dd + m          # Bcast(X_tr,y_tr) + Scatter(X_te) + Gather(y)
    c = q["alpha"] + M * q["beta"]
    s = np.asarray(s, dtype=float)
    return (np.log2(s) if kind == "arbol" else s - 1) * c, c


if comm and N_STRONG:
    fit["normalizacion"] = {}
    for kind in ("arbol", "lineal"):
        out = {}
        for N in N_STRONG:
            d = strong(N)
            if len(d) < 3:
                continue
            n, m, dd = int(d.n_train[0]), int(d.m_test[0]), int(d.d[0])
            Wn = 3 * m * n * dd
            K, c = comm_model(kind, d.s.values, n, m, dd)
            A, t = np.c_[Wn / d.s.values, K], d.T_total.values
            (k1, k2), _ = nnls(A / t[:, None], np.ones_like(t))      # error relativo
            err = np.abs(A @ np.array([k1, k2]) - t) / t
            if k2 <= 0 or c <= 0:
                p_star = float("inf")
            elif kind == "arbol":
                p_star = k1 * Wn * np.log(2) / (k2 * c)
            else:
                p_star = np.sqrt(k1 * Wn / (k2 * c))
            out[f"N{N}"] = dict(k1=k1, k2=k2, GFLOPs_equivalente=1 / k1 / 1e9 if k1 > 0 else None,
                                err_rel_medio=float(err.mean()), W=Wn, c=float(c),
                                p_star=float(p_star),
                                p_best_medido=int(d.s[np.argmin(d.T_total.values)]))
        fit["normalizacion"][kind] = out

    Ns = [N for N in dict.fromkeys((N_STRONG[0], NMAX))
          if f"N{N}" in fit["normalizacion"]["lineal"]]
    if Ns:
        fig, axes = plt.subplots(1, len(Ns), figsize=(W2 if len(Ns) > 1 else W1, 2.5),
                                 squeeze=False)
        for ax, N in zip(axes[0], Ns):
            d = strong(N)
            n, m, dd = int(d.n_train[0]), int(d.m_test[0]), int(d.d[0])
            Wn = 3 * m * n * dd
            ss = np.linspace(1, d.s.max(), 300)
            ax.plot(d.s, d.T_total, color=C["ink"], marker="o", ls="none",
                    label="medido", zorder=3)
            for kind, col, ls, lab in (("arbol", C["blue"], "--", r"árbol $\log_2 p$"),
                                       ("lineal", C["orange"], "-", r"lineal $(p-1)$")):
                q = fit["normalizacion"][kind][f"N{N}"]
                K, _ = comm_model(kind, ss, n, m, dd)
                star = "" if not np.isfinite(q["p_star"]) else f", $p^*$={q['p_star']:.1f}"
                ax.plot(ss, q["k1"] * Wn / ss + q["k2"] * K, color=col, ls=ls,
                        label=r"$k_1W/p+k_2T_{comm}$, " + lab + star)
            ax.set_yscale("log"); sticks(ax, d.s.astype(int).tolist())
            ax.set_xlabel("procesos $p$"); ax.set_ylabel(r"$T_p(p)$ [s] (log)")
            ax.set_title(f"Modelo de tiempo: Beta 3, {nlab(N)}"
                         + ("" if MODO == "train" else ", $n$=50k"))
            ax.legend(loc="upper right")
        save(fig, "k6_modelo")


# =========================================================================
# Escalabilidad débil (sin figura; sólo se guarda en model_fit.json)
#   S_esc(p) = (W(p)/W(1)) * T(1)/T(p),   E_w(p) = S_esc(p)/p   (ideal: E_w = 1)
#   weak-work: N = N0 sqrt(p) -> W/p constante -> E_w = T(1)/T(p)
#   weak-lin : N = N0 p       -> N/p constante (W/p crece como p)
#   (modo test) weak-test: m = m0 p, n fijo -> W lineal en m -> W/p constante
fit["weak"] = {}
for v, _ in VERS:
    for tag, _ in WEAK_TAGS:
        w = weak_series(v, tag)
        if w:
            d, Sg, E = w
            fit["weak"][f"{v}_{tag}"] = dict(
                p=d.s.astype(int).tolist(), N=d.N.astype(int).tolist(),
                n_train=d.n_train.astype(int).tolist(), m_test=d.m_test.astype(int).tolist(),
                T_total=d.T_total.tolist(), T_comp=d.T_comp.tolist(), T_comm=d.T_comm.tolist(),
                frac_comm=(d.T_comm / d.T_total).tolist(), S_escalado=Sg.tolist(), E=E.tolist())


# =========================================================================
# Tabla LaTeX (Beta 3, strong, N máximo) y JSON de ajustes
# =========================================================================
p = sp(skey("v3", NMAX)) if NMAX else None
if p:
    d = strong(NMAX)
    lines = [rf"% --- Khipu ({MODO}): escalabilidad fuerte, Beta 3, {SYM} = {NMAX} ---",
             r"\begin{tabular}{rrrrrrr}", r"\toprule",
             r"$p$ & $T_{\text{total}}$ [s] & $T_{\text{comp}}$ [s] & $T_{\text{comm}}$ [s] & $S(p)$ & $E(p)$ & GFLOP/s \\",
             r"\midrule"]
    for i, r in d.iterrows():
        lines.append(rf"{int(r.s)} & {r.T_total:.2f} & {r.T_comp:.2f} & {r.T_comm:.3f} & "
                     rf"{p['S'][i]:.2f} & {p['E'][i]:.2f} & {r.gflops:.1f} \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    with open(os.path.join(RES, "tables.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"  {rel(os.path.join(RES, 'tables.tex'))}")


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


with open(os.path.join(RES, "model_fit.json"), "w", encoding="utf-8") as fh:
    json.dump(_clean(fit), fh, indent=2, ensure_ascii=False)
print(f"  {rel(os.path.join(RES, 'model_fit.json'))}")

# Resumen en consola
for k, q in fit["speedup"].items():
    am = (f"fs={q['amdahl_fs']:.3f} Smax={q['amdahl_Smax']:.1f} R2={q['amdahl_R2']:.3f}"
          if "amdahl_fs" in q else "(sin ajuste)")
    print(f"{k:12s} T1={q['T1']:.4g}s  S_max_medido={q['S_max_medido']:.2f} "
          f"en s={q['s_best_medido']}  {am}")
