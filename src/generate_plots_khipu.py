"""
generate_plots_khipu.py -- Gráficas de las mediciones hechas en Khipu (UTEC).

Independiente de generate_plots.py (laptop i7-14650HX): no usa pinning.csv ni las
franjas P-cores/E-cores/SMT, y toma los valores de s directamente de los datos.

Entradas  (results/khipu/): results.csv
Salidas   (figures/khipu/): k2_tiempos, k3_speedup, k4_eficiencia, k5_gflops (.png)
          (results/khipu/): model_fit.json, tables.tex

Cuatro figuras, todas con el número de procesos p en el eje x y sin la variante DAG.
(En las figuras los procesos se denotan p; la columna del CSV se sigue llamando 's'.)
Cada una tiene hasta dos paneles (se omite el que no tenga datos):
  (a) versiones sobre load_digits (tag 'versions'): Beta 1, Beta 2 y Beta 3
      (+ el secuencial original en la de tiempos);
  (b) escalabilidad fuerte de la Beta 3 (tag 'strong'), una serie por N.

Uso:
    python src/generate_plots_khipu.py
    python src/generate_plots_khipu.py --results results/khipu --figures figures/khipu

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
from scipy.optimize import curve_fit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--results", default=os.path.join(ROOT, "results", "khipu"))
ap.add_argument("--figures", default=os.path.join(ROOT, "figures", "khipu"))
args = ap.parse_args()
RES, FIG = os.path.abspath(args.results), os.path.abspath(args.figures)
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
hosts = ", ".join(sorted(R.host.dropna().astype(str).unique()))
print(f"[khipu] {len(R)} filas | hosts: {hosts}")
fit = dict(hosts=hosts)

V = MED[MED.tag == "versions"]                       # load_digits: seq, Beta 1-3
VB = [(v, lab) for v, lab in BETAS if not V[V.version == v].empty]
S_VERS = sorted(int(s) for s in V[V.version.isin([v for v, _ in VB])].s.unique())
ST = MED[(MED.tag == "strong") & (MED.version == "v3")]
N_STRONG = sorted(int(n) for n in ST.N.unique())
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


def strong(N):
    return ST[ST.N == N].sort_values("s").reset_index(drop=True)


def nlab(N):
    return f"$N$={N // 1000}k"


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


T_DIGITS = "load_digits ($n$=1437, $m$=360)"
T_STRONG = "Beta 3, strong scaling"

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
for N in N_STRONG:
    p = speedup_fit(strong(N), with_amdahl=True)
    if p:
        fit["speedup"][f"v3_N{N}"] = p
    else:
        print(f"  [aviso] v3 N={N}: falta s=1 o hay <2 puntos; sin speedup")


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
# k3: speedup
# =========================================================================
def speedup_a(ax):
    top_s = max(S_VERS)
    for (v, lab), (col, mk, ls) in zip(VB, SERIES):
        p = sp(f"{v}_digits")
        if p:
            ax.plot(p["s"], p["S"], color=col, marker=mk, ls=ls, label=lab)
    ax.plot([1, top_s], [1, top_s], color=C["light"], lw=1, label="ideal $S=p$", zorder=1)
    ax.set_ylim(0, top_s * 1.05); sticks(ax, S_VERS)
    ax.set_ylabel("speedup $S(p)=T(1)/T(p)$")
    ax.set_title(f"Speedup: {T_DIGITS}")
    ax.legend(loc="upper left")


def speedup_b(ax):
    top_s, top_S = max(S_STRONG), 1.0
    for N, (col, mk, ls) in zip(N_STRONG, SERIES):
        p = sp(f"v3_N{N}")
        if p:
            ax.plot(p["s"], p["S"], color=col, marker=mk, ls="none", label=f"medido, {nlab(N)}")
            top_S = max(top_S, max(p["S"]))
    ax.plot([1, top_s], [1, top_s], color=C["light"], lw=1, label="ideal $S=p$", zorder=1)
    p = sp(f"v3_N{NMAX}")
    if p and "amdahl_fs" in p:
        ss = np.linspace(1, top_s, 200)
        ax.plot(ss, amdahl(ss, p["amdahl_fs"]), color=C["ink"], ls="--", lw=1.1,
                label=f"Amdahl ({nlab(NMAX)}): $f_s$={p['amdahl_fs']:.3f}, "
                      f"$S_{{max}}$={p['amdahl_Smax']:.1f}")
    ax.set_ylim(0, min(top_s, top_S * 2.0) * 1.05); sticks(ax, S_STRONG)
    ax.set_ylabel("speedup $S(p)=T(1)/T(p)$")
    ax.set_title(f"Speedup: {T_STRONG}")
    ax.legend(loc="upper left")


panels("k3_speedup", speedup_a, speedup_b)


# =========================================================================
# k4: eficiencia
# =========================================================================
def eficiencia_a(ax):
    for (v, lab), (col, mk, ls) in zip(VB, SERIES):
        p = sp(f"{v}_digits")
        if p:
            ax.plot(p["s"], p["E"], color=col, marker=mk, ls=ls, label=lab)
    ax.axhline(1, color=C["light"], lw=1, zorder=1)
    ax.set_ylim(0, 1.15); sticks(ax, S_VERS)
    ax.set_ylabel("eficiencia $E(p)=S(p)/p$")
    ax.set_title(f"Eficiencia: {T_DIGITS}")
    ax.legend(loc="lower left")


def eficiencia_b(ax):
    for N, (col, mk, ls) in zip(N_STRONG, SERIES):
        p = sp(f"v3_N{N}")
        if p:
            ax.plot(p["s"], p["E"], color=col, marker=mk, ls=ls, label=nlab(N))
    ax.axhline(1, color=C["light"], lw=1, zorder=1)
    ax.set_ylim(0, 1.15); sticks(ax, S_STRONG)
    ax.set_ylabel("eficiencia $E(p)=S(p)/p$")
    ax.set_title(f"Eficiencia: {T_STRONG}")
    ax.legend(loc="lower left")


panels("k4_eficiencia", eficiencia_a, eficiencia_b, height=2.4)


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
# Tabla LaTeX (Beta 3, strong, N máximo) y JSON de ajustes
# =========================================================================
p = sp(f"v3_N{NMAX}") if NMAX else None
if p:
    d = strong(NMAX)
    lines = [rf"% --- Khipu: escalabilidad fuerte, Beta 3, N = {NMAX} ---",
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
