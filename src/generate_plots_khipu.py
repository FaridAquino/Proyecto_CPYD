"""
generate_plots_khipu.py -- Gráficas de las mediciones hechas en Khipu (UTEC).

Independiente de generate_plots.py (laptop i7-14650HX): no usa pinning.csv ni las
franjas P-cores/E-cores/SMT, y toma los valores de s directamente de los datos.

Entradas  (results/khipu/): results.csv  [+ comm_bench.csv, opcional]
Salidas   (figures/khipu/): k1_versiones, k2_tiempos, k3_speedup, k4_eficiencia,
                            k5_gflops, k6_modelo, k7_debil, k9_comunicacion (.pdf/.png)
          (results/khipu/): model_fit.json, tables.tex

Tolera campañas parciales: cada figura se genera sólo si existen sus datos
(p. ej. sin comm_bench.csv se omiten k6 y k9).

Uso:
    python src/generate_plots_khipu.py
    python src/generate_plots_khipu.py --results results/khipu --figures figures/khipu

Modelos (los mismos del informe): Amdahl S = 1/(f_s + f_p/s); Gustafson
S = f_s + s f_p; T_p(s) = k1 W/s + k2 T_comm(s) con T_comm en árbol o lineal.
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
    "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42,
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
NAMES = dict(v1="Beta 1", v2="Beta 2", v3="Beta 3", dag="DAG")


def save(fig, name):
    fig.tight_layout(pad=1.2)
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300)
    plt.close(fig)
    print(f"  {rel(os.path.join(FIG, name))}.pdf")


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
cb_path = os.path.join(RES, "comm_bench.csv")
CB = pd.read_csv(cb_path) if os.path.exists(cb_path) else None
hosts = ", ".join(sorted(R.host.dropna().astype(str).unique()))
print(f"[khipu] {len(R)} filas | hosts: {hosts}")
fit = dict(hosts=hosts)

S_ALL = sorted(int(s) for s in MED.s.unique())
SMAX = max(S_ALL)


def sticks(ax, svals=None):
    """Marcas en los s medidos; se omiten etiquetas que chocarían entre sí."""
    svals = sorted(svals if svals is not None else S_ALL)
    top = max(svals)
    major = [s for s in svals if s == 1 or s >= top / 12]
    ax.set_xticks(major)
    ax.set_xticks([s for s in svals if s not in major], minor=True)
    ax.set_xlim(1 - 0.03 * top, top * 1.03)


def strong(version, N):
    d = MED[(MED.version == version) & (MED.tag == "strong") & (MED.N == N)]
    return d.sort_values("s").reset_index(drop=True)


VERS = [v for v in ("v3", "dag") if not MED[(MED.version == v) & (MED.tag == "strong")].empty]
N_STRONG = sorted(int(n) for n in MED[MED.tag == "strong"].N.unique())
NMAX = max(N_STRONG) if N_STRONG else None
TITLES = dict(v3="Beta 3 (replica $X_{tr}$)", dag="Variante DAG (reparte $X_{tr}$)")


def r2(y, pred):
    den = np.sum((y - np.mean(y)) ** 2)
    return float(1 - np.sum((y - pred) ** 2) / den) if den > 0 else float("nan")


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
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.3))
    Mbig = CB.M.max()
    for ax, op in zip(axes, ("Bcast", "Gather")):
        x = CB[(CB.op == op) & (CB.M >= 0.99 * Mbig)].sort_values("s")
        ax.plot(x.s, x.t_med * 1e3, color=C["ink"], marker="o", ls="none",
                label="medido", zorder=3)
        ss = np.linspace(2, x.s.max(), 200)
        for name, col, ls in (("arbol", C["blue"], "--"), ("lineal", C["orange"], "-")):
            p = comm[op][name]
            g = np.log2(ss) if name == "arbol" else ss - 1
            gl = r"\log_2 s" if name == "arbol" else "(s-1)"
            if op == "Bcast":
                y, lab = g * (p["alpha"] + Mbig * p["beta"]), rf"${gl}\,(\alpha+M\beta)$"
            else:
                y, lab = p["alpha"] * g + Mbig * p["beta"], rf"$\alpha\,{gl}+M\beta$"
            kind = "árbol" if name == "arbol" else "lineal"
            ax.plot(ss, y * 1e3, color=col, ls=ls, label=f"{kind}: {lab}  $R^2$={p['R2']:.3f}")
        ax.set_title(f"MPI_{op} en Khipu, $M$ = {Mbig / 1e6:.1f}M palabras ({8 * Mbig / 2**20:.0f} MiB)")
        ax.set_xlabel("procesos $s$"); ax.set_ylabel("tiempo [ms]")
        sticks(ax, x.s.tolist()); ax.legend(loc="upper left")
    save(fig, "k9_comunicacion")
else:
    skip("k9_comunicacion / k6_modelo", "falta comm_bench.csv (extra_benchmarks.py --exp comm)")

# =========================================================================
# k1: registro de versiones (load_digits)
# =========================================================================
V = MED[MED.tag == "versions"]
if V.empty:
    skip("k1_versiones", "no hay filas con tag 'versions'")
else:
    fig, ax = plt.subplots(figsize=(W1, 2.4))
    seq = V[V.version == "seq"]
    if not seq.empty:
        t_seq = float(seq.T_total.iloc[0])
        fit["t_seq"] = t_seq
        ax.axhline(t_seq, color=C["gray"], lw=1, ls=":", label="secuencial original")
    order = [("v1", "Beta 1 (pickle + bucle)"), ("v2", "Beta 2 (NumPy)"),
             ("v3", "Beta 3 (buffers, Scatterv)"), ("dag", "Variante DAG")]
    for (v, lab), (col, mk, ls) in zip(order, SERIES):
        d = V[V.version == v].sort_values("s")
        if not d.empty:
            ax.plot(d.s, d.T_total, color=col, marker=mk, ls=ls, label=lab)
    sv = sorted(int(s) for s in V.s.unique())
    ax.set_yscale("log"); ax.set_xscale("log", base=2)
    ax.set_xticks(sv); ax.set_xticklabels([str(s) for s in sv])
    ax.set_xlabel("procesos $s$"); ax.set_ylabel(r"$T_{\mathrm{total}}$ [s] (log)")
    ax.set_title(r"Khipu: versiones sobre load_digits ($n$=1437, $m$=360)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.30), ncol=2)
    save(fig, "k1_versiones")

# =========================================================================
# Escalabilidad fuerte: k2 tiempos, k3 speedup (Amdahl), k4 eficiencia, k5 GFLOP/s
# =========================================================================
def amdahl(s, fs):
    return 1 / (fs + (1 - fs) / s)


fit["speedup"] = {}
if not VERS:
    skip("k2..k6", "no hay filas con tag 'strong'")
else:
    # ---- k2
    fig, axes = plt.subplots(1, len(VERS), figsize=(W2 if len(VERS) > 1 else W1, 2.4),
                             sharey=True, squeeze=False)
    for ax, v in zip(axes[0], VERS):
        d = strong(v, NMAX)
        cols = [("T_total", r"$T_{\mathrm{total}}$"), ("T_comp", r"$T_{\mathrm{comp}}$"),
                ("T_comm", r"$T_{\mathrm{comm}}$")]
        if v == "dag":
            cols.append(("T_merge", r"$T_{\mathrm{fusión}}$ (serial)"))
        for (c_, lab), (col, mk, ls) in zip(cols, SERIES):
            ax.plot(d.s, d[c_], color=col, marker=mk, ls=ls, label=lab)
        ax.set_yscale("log"); sticks(ax, d.s.tolist()); ax.set_xlabel("procesos $s$")
        ax.set_title(f"{TITLES[v]}, $N$={NMAX:,}".replace(",", " "))
        ax.legend(loc="best")
    axes[0][0].set_ylabel("tiempo [s] (log)")
    save(fig, "k2_tiempos")

    # ---- ajuste de Amdahl
    for v in VERS:
        for N in N_STRONG:
            d = strong(v, N)
            if d.empty or int(d.s.iloc[0]) != 1 or len(d) < 2:
                print(f"  [aviso] {v} N={N}: falta s=1 o hay <2 puntos; sin speedup")
                continue
            T1 = float(d.T_total.iloc[0])
            S = T1 / d.T_total.values
            p = dict(T1=T1, s=d.s.astype(int).tolist(), S=S.tolist(),
                     E=(S / d.s.values).tolist(), S_max_medido=float(S.max()),
                     s_best_medido=int(d.s[np.argmin(d.T_total.values)]))
            if len(d) >= 3:
                (fs,), cov = curve_fit(amdahl, d.s.values, S, p0=[0.05], bounds=(0, 1))
                p.update(amdahl_fs=float(fs), amdahl_fs_err=float(np.sqrt(cov[0, 0])),
                         amdahl_R2=r2(S, amdahl(d.s.values, fs)),
                         amdahl_Smax=float(1 / fs) if fs > 0 else float("inf"))
            fit["speedup"][f"{v}_N{N}"] = p

    def sp(v, N):
        return fit["speedup"].get(f"{v}_N{N}")

    if not fit["speedup"]:
        skip("k3, k4", "ninguna serie strong tiene s=1")
    else:
        # ---- k3
        fig, axes = plt.subplots(1, len(VERS), figsize=(W2 if len(VERS) > 1 else W1, 2.6),
                                 squeeze=False)
        for ax, v in zip(axes[0], VERS):
            top_s, top_S = 1, 1.0
            for N, (col, mk, ls) in zip(N_STRONG, SERIES):
                p = sp(v, N)
                if p:
                    ax.plot(p["s"], p["S"], color=col, marker=mk, ls="none",
                            label=f"medido, $N$={N // 1000}k")
                    top_s, top_S = max(top_s, max(p["s"])), max(top_S, max(p["S"]))
            ax.plot([1, top_s], [1, top_s], color=C["light"], lw=1, label="ideal $S=s$")
            p = sp(v, NMAX)
            if p and "amdahl_fs" in p:
                ss = np.linspace(1, top_s, 200)
                ax.plot(ss, amdahl(ss, p["amdahl_fs"]), color=C["ink"], ls="--", lw=1.1,
                        label=f"Amdahl ($N$={NMAX // 1000}k): $f_s$={p['amdahl_fs']:.3f}, "
                              f"$S_{{max}}$={p['amdahl_Smax']:.1f}")
            ax.set_ylim(0, min(top_s, top_S * 1.6) * 1.05)
            sticks(ax, range(1, top_s + 1) if top_s <= 8 else [s for s in S_ALL if s <= top_s])
            ax.set_xlabel("procesos $s$"); ax.set_ylabel("speedup $S(s)=T_1/T_p(s)$")
            ax.set_title(f"{NAMES[v]} en Khipu: speedup (strong scaling)")
            ax.legend(loc="upper left")
        save(fig, "k3_speedup")

        # ---- k4
        fig, axes = plt.subplots(1, len(VERS), figsize=(W2 if len(VERS) > 1 else W1, 2.4),
                                 sharey=True, squeeze=False)
        for ax, v in zip(axes[0], VERS):
            top_s = 1
            for N, (col, mk, ls) in zip(N_STRONG, SERIES):
                p = sp(v, N)
                if p:
                    ax.plot(p["s"], p["E"], color=col, marker=mk, ls=ls, lw=0.8,
                            label=f"$N$={N // 1000}k")
                    top_s = max(top_s, max(p["s"]))
            ax.axhline(1, color=C["light"], lw=1)
            ax.set_ylim(0, 1.15); sticks(ax, [s for s in S_ALL if s <= top_s])
            ax.set_xlabel("procesos $s$"); ax.set_title(f"Eficiencia, {NAMES[v]} (Khipu)")
            ax.legend(loc="lower left")
        axes[0][0].set_ylabel("eficiencia $E(s)=S(s)/s$")
        save(fig, "k4_eficiencia")

    # ---- k5
    fig, ax = plt.subplots(figsize=(W1, 2.6))
    d1 = strong(VERS[0], NMAX)
    if int(d1.s.iloc[0]) == 1:
        g1 = float(d1.gflops.iloc[0])
        ax.plot([1, d1.s.max()], [g1, g1 * d1.s.max()], color=C["light"], lw=1,
                label=f"ideal: {g1:.1f}·$s$ (1 núcleo)")
    gmax = 0.0
    series = [("v3", N) for N in N_STRONG if "v3" in VERS]
    if "dag" in VERS:
        series.append(("dag", NMAX))
    for (v, N), (col, mk, ls) in zip(series, SERIES):
        d = strong(v, N)
        if not d.empty:
            ax.plot(d.s, d.gflops, color=col, marker=mk, ls=ls,
                    label=f"{NAMES[v]}, $N$={N // 1000}k")
            gmax = max(gmax, float(d.gflops.max()))
    fit["gflops_max"] = gmax
    ax.set_ylim(0, gmax * 1.25); sticks(ax, d1.s.tolist())
    ax.set_xlabel("procesos $s$"); ax.set_ylabel("GFLOP/s  ($3mnd\\,/\\,T_{\\mathrm{comp}}$)")
    ax.set_title("Khipu: rendimiento de la región paralelizable")
    ax.legend(loc="upper left")
    save(fig, "k5_gflops")

# =========================================================================
# k6: normalización T_p(s) = k1 W/s + k2 T_comm(s) y s* óptimo
# =========================================================================
def comm_model(kind, s, n, m, dd):
    p = comm["Bcast"][kind]
    M = n * (dd + 1) + m * dd + m          # Bcast(X_tr,y_tr) + Scatter(X_te) + Gather(y)
    c = p["alpha"] + M * p["beta"]
    s = np.asarray(s, dtype=float)
    return (np.log2(s) if kind == "arbol" else s - 1) * c, c


if comm and "v3" in VERS:
    fit["normalizacion"] = {}
    for kind in ("arbol", "lineal"):
        out = {}
        for N in N_STRONG:
            d = strong("v3", N)
            if len(d) < 3:
                continue
            n, m, dd = int(d.n_train[0]), int(d.m_test[0]), int(d.d[0])
            Wn = 3 * m * n * dd
            K, c = comm_model(kind, d.s.values, n, m, dd)
            A, t = np.c_[Wn / d.s.values, K], d.T_total.values
            (k1, k2), _ = nnls(A / t[:, None], np.ones_like(t))      # error relativo
            err = np.abs(A @ np.array([k1, k2]) - t) / t
            if k2 <= 0 or c <= 0:
                s_star = float("inf")
            elif kind == "arbol":
                s_star = k1 * Wn * np.log(2) / (k2 * c)
            else:
                s_star = np.sqrt(k1 * Wn / (k2 * c))
            out[f"N{N}"] = dict(k1=k1, k2=k2, GFLOPs_equivalente=1 / k1 / 1e9 if k1 > 0 else None,
                                err_rel_medio=float(err.mean()), W=Wn, c=float(c),
                                s_star=float(s_star),
                                s_best_medido=int(d.s[np.argmin(d.T_total.values)]))
        fit["normalizacion"][kind] = out

    Ns = [N for N in dict.fromkeys((N_STRONG[0], NMAX))
          if f"N{N}" in fit["normalizacion"]["lineal"]]
    if Ns:
        fig, axes = plt.subplots(1, len(Ns), figsize=(W2 if len(Ns) > 1 else W1, 2.5),
                                 squeeze=False)
        for ax, N in zip(axes[0], Ns):
            d = strong("v3", N)
            n, m, dd = int(d.n_train[0]), int(d.m_test[0]), int(d.d[0])
            Wn = 3 * m * n * dd
            ss = np.linspace(1, d.s.max(), 300)
            ax.plot(d.s, d.T_total, color=C["ink"], marker="o", ls="none",
                    label="medido", zorder=3)
            for kind, col, ls, lab in (("arbol", C["blue"], "--", r"árbol $\log_2 s$"),
                                       ("lineal", C["orange"], "-", r"lineal $(s-1)$")):
                q = fit["normalizacion"][kind][f"N{N}"]
                K, _ = comm_model(kind, ss, n, m, dd)
                star = "" if not np.isfinite(q["s_star"]) else f", $s^*$={q['s_star']:.1f}"
                ax.plot(ss, q["k1"] * Wn / ss + q["k2"] * K, color=col, ls=ls,
                        label=r"$k_1W/s+k_2T_{comm}$, " + lab + star)
            ax.set_yscale("log"); sticks(ax, d.s.tolist())
            ax.set_xlabel("procesos $s$"); ax.set_ylabel(r"$T_p(s)$ [s] (log)")
            ax.set_title(f"Khipu: normalización teórica, Beta 3, $N$={N // 1000}k")
            ax.legend(loc="upper right")
        save(fig, "k6_modelo")

# =========================================================================
# k7: escalabilidad débil y Ley de Gustafson
# =========================================================================
def gustafson(s, fs):
    return fs + s * (1 - fs)


fit["weak"] = {}
weak_tags = [t for t in ("weak-work", "weak-lin") if not MED[MED.tag == t].empty]
if not weak_tags:
    skip("k7_debil", "no hay filas con tag 'weak-work' / 'weak-lin'")
else:
    wt = {"weak-work": r"$N = N_0\sqrt{s}$ (trabajo por proceso constante)",
          "weak-lin": r"$N = N_0\,s$ ($N/s$ constante)"}
    fig, axes = plt.subplots(1, len(weak_tags), figsize=(W2 if len(weak_tags) > 1 else W1, 2.5),
                             squeeze=False)
    for ax, tag in zip(axes[0], weak_tags):
        top_s = 1
        for (v, lab), (col, mk, ls) in zip((("v3", "Beta 3"), ("dag", "DAG")), SERIES[:2]):
            d = MED[(MED.version == v) & (MED.tag == tag)].sort_values("s")
            if len(d) < 2 or int(d.s.iloc[0]) != 1:
                continue
            W = (3 * d.m_test * d.n_train * d.d).values
            Sg = (W / W[0]) * float(d.T_total.iloc[0]) / d.T_total.values
            q = dict(s=d.s.astype(int).tolist(), N=d.N.astype(int).tolist(),
                     T=d.T_total.tolist(), S_escalado=Sg.tolist(),
                     E=(Sg / d.s.values).tolist())
            ax.plot(d.s, Sg, color=col, marker=mk, ls="none", label=f"medido, {lab}")
            top_s = max(top_s, int(d.s.max()))
            if len(d) >= 3:
                (fs,), _ = curve_fit(gustafson, d.s.values, Sg, p0=[0.5], bounds=(0, 1))
                q["gustafson_fs"] = float(fs)
                sx = np.linspace(1, d.s.max(), 100)
                ax.plot(sx, gustafson(sx, fs), color=col, ls=ls, lw=1.0,
                        label=f"Gustafson {lab}: $f_s$={fs:.2f}")
            fit["weak"][f"{v}_{tag}"] = q
        ax.plot([1, top_s], [1, top_s], color=C["light"], lw=1, label="ideal $S=s$")
        sticks(ax, [s for s in S_ALL if s <= top_s]); ax.set_ylim(0, top_s * 1.05)
        ax.set_xlabel("procesos $s$")
        ax.set_ylabel(r"speedup escalado $\frac{W(s)}{W(1)}\frac{T(1)}{T(s)}$")
        ax.set_title(wt[tag]); ax.legend(loc="upper left")
    save(fig, "k7_debil")

# =========================================================================
# Tabla LaTeX (strong, N máximo) y JSON de ajustes
# =========================================================================
lines = []
for v in VERS:
    p = fit["speedup"].get(f"{v}_N{NMAX}")
    if not p:
        continue
    d = strong(v, NMAX)
    lines += [rf"% --- Khipu: escalabilidad fuerte, {NAMES[v]}, N = {NMAX} ---",
              r"\begin{tabular}{rrrrrrr}", r"\toprule",
              r"$s$ & $T_{\text{total}}$ [s] & $T_{\text{comp}}$ [s] & $T_{\text{comm}}$ [s] & $S(s)$ & $E(s)$ & GFLOP/s \\",
              r"\midrule"]
    for i, r in d.iterrows():
        lines.append(rf"{int(r.s)} & {r.T_total:.2f} & {r.T_comp:.2f} & {r.T_comm:.3f} & "
                     rf"{p['S'][i]:.2f} & {p['E'][i]:.2f} & {r.gflops:.1f} \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
if lines:
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
    print(f"{k:12s} T1={q['T1']:.2f}s  S_max_medido={q['S_max_medido']:.2f} "
          f"en s={q['s_best_medido']}  {am}")
for kind, qq in fit.get("normalizacion", {}).items():
    for N, q in qq.items():
        print(f"[{kind} {N}] k1={q['k1']:.3e} k2={q['k2']:.3f} err={q['err_rel_medio']:.1%} "
              f"s*={q['s_star']:.1f} medido={q['s_best_medido']}")
