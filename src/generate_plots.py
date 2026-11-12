"""
generate_plots.py -- Análisis científico y gráficos (Fase 4, CS4052 Proyecto 5).

Entradas (results/): results.csv, comm_bench.csv, pinning.csv
Salidas:
  figures/*.pdf (+ .png)       figuras para el informe (estilo IEEE, serif)
  results/model_fit.json       constantes ajustadas (alpha, beta, k1, k2, f, sigma, kappa, s*)
  results/tables.tex           tablas booktabs listas para \\input

Modelos (exclusivamente los del curso CS4052)
-------------------------------------------
(M1) Normalización del modelo teórico (Ec. 12 del informe parcial):
        T_p(s) = k1 * W/s + k2 * T_comm(s),      W = 3 m n d   (FLOPs, Brent con s <= m)
     con dos modelos de comunicación vistos en clase (Unidad 2):
        árbol binario : T_comm(s) = log2(s) (alpha + M beta)
        lineal        : T_comm(s) = (s-1) (alpha + M beta)
     alpha, beta se miden con comm_bench (colectivas sin cómputo); k1 [s/FLOP] y
     k2 [-] se ajustan por mínimos cuadrados sobre T_total medido.
     dT_p/ds = 0:
        árbol : s* = k1 W ln2 / (k2 c)        lineal : s* = sqrt(k1 W / (k2 c)),
        c = alpha + M beta.
(M2) Ley de Amdahl (strong scaling): S(s) = 1 / (f_s + f_p/s), f_s + f_p = 1,
     S_max = 1/f_s.
(M3) Ley de Gustafson (weak scaling): S(s) = f_s + s f_p.
"""
import json
import sys
sys.stdout.reconfigure(encoding="utf-8")
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, nnls

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

# --------------------------------------------------------------------- estilo
# Okabe-Ito (validado: CVD-safe con codificación secundaria = marcador + trazo)
C = dict(blue="#0072B2", orange="#D55E00", green="#009E73", pink="#CC79A7",
         gray="#6B6B6B", ink="#1A1A1A", light="#BDBDBD")
SERIES = [(C["blue"], "o", "-"), (C["orange"], "s", "--"),
          (C["green"], "^", "-."), (C["pink"], "D", ":")]
plt.rcParams.update({
    # STIX: fuentes codificadas en Unicode y de estilo Times/LaTeX. Se evita el
    # fontset "cm" (BaKoMa), cuyos glifos no son Unicode: varios visores PDF los
    # muestran como ¡ (menos), ® (alpha), ¯ (beta) o □ (asterisco).
    "font.family": "serif", "font.serif": ["STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42, "ps.fonttype": 42,          # TrueType incrustada (no Type 3)
    "axes.unicode_minus": True, "axes.titlepad": 12,
    "font.size": 8, "axes.titlesize": 8.5,
    "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "lines.linewidth": 1.4, "lines.markersize": 4.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": C["gray"], "axes.linewidth": 0.6,
    "xtick.color": C["gray"], "ytick.color": C["gray"],
    "axes.labelcolor": C["ink"], "text.color": C["ink"],
    "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.5,
    "legend.frameon": False, "figure.dpi": 150, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
W1, W2 = 3.5, 7.16       # ancho IEEE: una y dos columnas (pulgadas)
S_TICKS = [1, 2, 4, 8, 12, 16, 20, 24]
S_LABELED = [1, 4, 8, 12, 16, 20, 24]     # el 2 queda como marca menor (evita choque con el 1)


def save(fig, name):
    fig.tight_layout(pad=1.5)
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300)
    plt.close(fig)
    print(f"  figures/{name}.pdf")


def hw_bands(ax):
    """Sombrea las regiones P-cores / E-cores / SMT del i7-14650HX."""
    ax.axvspan(8, 16, color="#F2F2F2", zorder=0, lw=0)
    ax.axvspan(16, 24.5, color="#E8E8E8", zorder=0, lw=0)
    # Etiquetas en una franja sobre el área de datos (coordenada y en fracción de
    # ejes): nunca se superponen con curvas, marcadores ni leyendas.
    tr = ax.get_xaxis_transform()
    for x, t in ((4.5, "P-cores"), (12, "+E-cores"), (20, "+SMT")):
        ax.text(x, 1.005, t, ha="center", va="bottom", fontsize=6, color=C["gray"],
                transform=tr, clip_on=False)


def sticks(ax):
    ax.set_xticks(S_LABELED)
    ax.set_xticks([2], minor=True)
    ax.set_xlim(0.5, 24.5)


# --------------------------------------------------------------------- datos
R = pd.read_csv(os.path.join(RES, "results.csv"))
CB = pd.read_csv(os.path.join(RES, "comm_bench.csv"))
PIN = pd.read_csv(os.path.join(RES, "pinning.csv"))
MED = (R.groupby(["version", "tag", "N", "s"], as_index=False)
        .median(numeric_only=True))
fit = {}


def strong(version, N):
    d = MED[(MED.version == version) & (MED.tag == "strong") & (MED.N == N)]
    return d.sort_values("s").reset_index(drop=True)


N_STRONG = sorted(MED[MED.tag == "strong"].N.unique())
NMAX = max(N_STRONG)

# =========================================================================
# (M4) Microbenchmark de comunicación: alpha, beta
# =========================================================================
def fit_comm():
    out = {}
    for op in ("Bcast", "Scatter", "Gather"):
        x = CB[CB.op == op]
        L, M, t = np.log2(x.s.values), x.M.values, x.t_med.values
        models = {"arbol": np.c_[L, M * (L if op == "Bcast" else 1)],
                  # Comunicación lineal (Unidad 2): Bcast O(p(alpha+M beta));
                  # Scatter/Gather O(alpha p + beta M)
                  "lineal": np.c_[x.s.values - 1, M * ((x.s.values - 1) if op == "Bcast" else 1)]}
        out[op] = {}
        for name, A in models.items():
            coef, _ = nnls(A, t)
            pred = A @ coef
            r2 = 1 - np.sum((t - pred) ** 2) / np.sum((t - t.mean()) ** 2)
            out[op][name] = dict(alpha=coef[0], beta=coef[1], R2=r2,
                                 GBps=8 / coef[1] / 1e9 if coef[1] > 0 else None)
    return out


comm = fit_comm()
fit["comm"] = comm
ALPHA = np.mean([comm[o]["arbol"]["alpha"] for o in ("Scatter", "Gather")])
BETA = np.mean([comm[o]["arbol"]["beta"] for o in ("Scatter", "Gather")])
fit["alpha_curso"], fit["beta_curso"] = ALPHA, BETA

fig, axes = plt.subplots(1, 2, figsize=(W2, 2.3))
Mbig = CB.M.max()
for ax, op in zip(axes, ("Bcast", "Gather")):
    x = CB[(CB.op == op) & (CB.M >= 0.99 * Mbig)].sort_values("s")   # M ajustado a múltiplo de s
    ax.plot(x.s, x.t_med * 1e3, color=C["ink"], marker="o", ls="none",
            label="medido", zorder=3)
    ss = np.linspace(2, 24, 100)
    for name, (col, ls) in (("arbol", (C["blue"], "--")), ("lineal", (C["orange"], "-"))):
        p = comm[op][name]
        if name == "arbol":
            y = np.log2(ss) * (p["alpha"] + Mbig * p["beta"]) if op == "Bcast" else \
                p["alpha"] * np.log2(ss) + Mbig * p["beta"]
            lab = (r"árbol binario: $\log_2 s\,(\alpha+M\beta)$" if op == "Bcast"
                   else r"árbol binario: $\alpha\log_2 s+M\beta$")
        else:
            if op == "Bcast":
                y = (ss - 1) * (p["alpha"] + Mbig * p["beta"])
                lab = r"lineal: $(s-1)(\alpha+M\beta)$"
            else:
                y = p["alpha"] * (ss - 1) + Mbig * p["beta"]
                lab = r"lineal: $\alpha(s-1)+M\beta$"
        ax.plot(ss, y * 1e3, color=col, ls=ls, label=f"{lab}  $R^2$={p['R2']:.3f}")
    ax.set_title(f"MPI_{op}, $M$ = {Mbig / 1e6:.1f}M palabras ({8 * Mbig / 2**20:.0f} MiB)")
    ax.set_xlabel("procesos $s$"); ax.set_ylabel("tiempo [ms]")
    sticks(ax); ax.set_xlim(1, 24.5); ax.legend(loc="upper left")
save(fig, "f9_comunicacion")

# =========================================================================
# (1) Registro de versiones (load_digits)
# =========================================================================
V = MED[MED.tag == "versions"]
t_seq = float(V[V.version == "seq"].T_total.iloc[0])
fig, ax = plt.subplots(figsize=(W1, 2.3))
order = [("v1", "Beta 1 (pickle + bucle)"), ("v2", "Beta 2 (NumPy)"),
         ("v3", "Beta 3 (buffers, Scatterv)"), ("dag", "Variante DAG")]
ax.axhline(t_seq, color=C["gray"], lw=1, ls=":")
ax.text(1.6, t_seq * 1.25, "secuencial original", va="bottom", fontsize=6.5, color=C["gray"])
for (v, lab), (col, mk, ls) in zip(order, SERIES):
    d = V[V.version == v].sort_values("s")
    ax.plot(d.s, d.T_total, color=col, marker=mk, ls=ls, label=lab)
ax.set_yscale("log"); ax.set_xscale("log", base=2)
ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(["1", "2", "4", "8"])
ax.set_xlabel("procesos $s$"); ax.set_ylabel(r"$T_{\mathrm{total}}$ [s] (log)")
ax.set_title(r"Evolución de versiones sobre load_digits ($n$=1437, $m$=360)")
ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.5), fontsize=6.3); ax.set_xlim(0.85, 11)
ax.set_ylim(8e-4, 4)
save(fig, "f1_versiones")

# =========================================================================
# (2) Tiempos desglosados (N máximo)
# =========================================================================
fig, axes = plt.subplots(1, 2, figsize=(W2, 2.4), sharey=True)
for ax, v, title in zip(axes, ("v3", "dag"), ("Beta 3 (replica $X_{tr}$)", "Variante DAG (reparte $X_{tr}$)")):
    d = strong(v, NMAX)
    for col_, (col, mk, ls), lab in zip(("T_total", "T_comp", "T_comm"), SERIES,
                                        (r"$T_{\mathrm{total}}$", r"$T_{\mathrm{comp}}$", r"$T_{\mathrm{comm}}$")):
        ax.plot(d.s, d[col_], color=col, marker=mk, ls=ls, label=lab)
    ax.set_yscale("log"); sticks(ax)
    ax.set_xlabel("procesos $s$"); ax.set_title(f"{title}, $N$={NMAX:,}".replace(",", " "))
    hw_bands(ax)
axes[0].set_ylabel("tiempo [s] (log)"); axes[1].legend(loc="lower right")
save(fig, "f2_tiempos")

# =========================================================================
# (M2) Speedup y Ley de Amdahl (strong scaling)
# =========================================================================
def amdahl(s, fs):
    return 1 / (fs + (1 - fs) / s)


fit["speedup"] = {}
for v in ("v3", "dag"):
    for N in N_STRONG:
        d = strong(v, N)
        T1 = float(d.T_total[d.s == 1].iloc[0])
        S = T1 / d.T_total.values
        (fs,), cov = curve_fit(amdahl, d.s, S, p0=[0.05], bounds=(0, 1))
        pred = amdahl(d.s.values, fs)
        r2 = 1 - np.sum((S - pred) ** 2) / np.sum((S - S.mean()) ** 2)
        m8 = d.s.values <= 8                      # región de P-cores
        (fs8,), _ = curve_fit(amdahl, d.s[m8], S[m8], p0=[0.05], bounds=(0, 1))
        fit["speedup"][f"{v}_N{N}"] = dict(
            T1=T1, s=d.s.tolist(), S=S.tolist(), E=(S / d.s).tolist(),
            amdahl_fs=fs, amdahl_fs_err=float(np.sqrt(cov[0, 0])), amdahl_R2=r2,
            amdahl_Smax=1 / fs, amdahl_fs_P=fs8, amdahl_Smax_P=1 / fs8,
            S_max_medido=float(S.max()), s_best_measured=int(d.s[np.argmin(d.T_total)]))

fig, axes = plt.subplots(1, 2, figsize=(W2, 2.5))
ss = np.linspace(1, 24, 200)
for ax, v, title in zip(axes, ("v3", "dag"), ("Beta 3", "Variante DAG")):
    ax.plot(S_TICKS, S_TICKS, color=C["light"], lw=1, label="ideal lineal $S=s$")
    for N, (col, mk, ls) in zip(N_STRONG, SERIES):
        p = fit["speedup"][f"{v}_N{N}"]
        ax.plot(p["s"], p["S"], color=col, marker=mk, ls="none",
                label=f"medido, $N$={N // 1000}k")
    p = fit["speedup"][f"{v}_N{NMAX}"]
    ax.plot(ss, amdahl(ss, p["amdahl_fs"]), color=C["ink"], ls="--", lw=1.1,
            label=f"Amdahl ($s\\leq 24$): $f_s$={p['amdahl_fs']:.3f}, $S_{{max}}$={p['amdahl_Smax']:.1f}")
    ax.plot(ss, amdahl(ss, p["amdahl_fs_P"]), color=C["gray"], ls=":", lw=1.2,
            label=f"Amdahl ($s\\leq 8$): $f_s$={p['amdahl_fs_P']:.3f}, $S_{{max}}$={p['amdahl_Smax_P']:.1f}")
    ax.set_ylim(0, 12); sticks(ax); hw_bands(ax)
    ax.set_xlabel("procesos $s$"); ax.set_ylabel("speedup $S(s)=T_1/T_p(s)$")
    ax.set_title(f"{title}: speedup (strong scaling)")
h0, l0 = axes[0].get_legend_handles_labels()
l1 = axes[1].get_legend_handles_labels()
fig.legend(h0 + l1[0][-2:], l0[:-2] + [x + " (Beta 3)" for x in l0[-2:]] + [x + " (DAG)" for x in l1[1][-2:]],
           loc="lower center", ncol=4, fontsize=6.2, bbox_to_anchor=(0.5, -0.10))
save(fig, "f3_speedup")

fig, ax = plt.subplots(figsize=(W1, 2.4))
for N, (col, mk, ls) in zip(N_STRONG, SERIES):
    p = fit["speedup"][f"v3_N{N}"]
    ax.plot(p["s"], p["E"], color=col, marker=mk, ls="none", label=f"medido, $N$={N // 1000}k")
p = fit["speedup"][f"v3_N{NMAX}"]
ax.plot(ss, amdahl(ss, p["amdahl_fs"]) / ss, color=C["ink"], ls="--", lw=1.1,
        label=f"Amdahl: $E=S/s$, $f_s$={p['amdahl_fs']:.3f}")
ax.axhline(1, color=C["light"], lw=1)
ax.set_ylabel("eficiencia $E(s)=S(s)/s$"); ax.set_ylim(0, 1.1)
ax.set_title("Eficiencia (strong scaling, Beta 3)")
sticks(ax); hw_bands(ax); ax.set_xlabel("procesos $s$")
ax.legend(loc="lower left", fontsize=6.3)
save(fig, "f4_eficiencia")

# =========================================================================
# (5) GFLOP/s vs s + techo medido con pinning
# =========================================================================
pin = PIN.groupby(["config", "s", "rank"], as_index=False).gflops.median()
agg = pin.groupby(["config", "s"], as_index=False).gflops.sum()
ceiling = float(agg.gflops.max())
p1 = float(agg[agg.config == "P1"].gflops.iloc[0])
fit["pinning"] = {r.config: dict(s=int(r.s), gflops_agregado=float(r.gflops))
                  for r in agg.itertuples()}

fig, ax = plt.subplots(figsize=(W1, 2.5))
ax.plot(S_TICKS, [p1 * s for s in S_TICKS], color=C["light"], lw=1,
        label=f"ideal: {p1:.1f}·$s$ (1 P-core)")
ax.axhline(ceiling, color=C["gray"], ls=":", lw=1)
ax.text(9, ceiling + 1.5, f"techo medido {ceiling:.0f} GFLOP/s (P8+E8)",
        fontsize=6, color=C["gray"])
for N, (col, mk, ls) in zip(N_STRONG, SERIES):
    d = strong("v3", N)
    ax.plot(d.s, d.gflops, color=col, marker=mk, ls=ls, label=f"Beta 3, $N$={N // 1000}k")
d = strong("dag", NMAX)
ax.plot(d.s, d.gflops, color=SERIES[3][0], marker=SERIES[3][1], ls=SERIES[3][2],
        label=f"DAG, $N$={NMAX // 1000}k")
ax.set_ylim(0, 110); sticks(ax); hw_bands(ax)
ax.set_xlabel("procesos $s$"); ax.set_ylabel("GFLOP/s  ($3mnd\\,/\\,T_{\\mathrm{comp}}$)")
ax.set_title("Rendimiento de la región paralelizable")
ax.legend(loc="lower right", fontsize=6.2)
save(fig, "f5_gflops")

# =========================================================================
# (M1) Normalización T_p(s) = k1 W/s + k2 T_comm(s) y proceso óptimo s*
# =========================================================================
def comm_model(kind, s, n, m, dd):
    """T_comm(s) con alpha, beta del Bcast medidos; M = palabras movidas."""
    p = comm["Bcast"][kind]
    M = n * (dd + 1) + m * dd + m          # Bcast(X_tr,y_tr) + Scatter(X_te) + Gather(y)
    c = p["alpha"] + M * p["beta"]
    g = np.log2(s) if kind == "arbol" else (np.asarray(s, dtype=float) - 1)
    return g * c, c


fit["normalizacion"] = {}
# Normalización POR TAMAÑO de problema: k1, k2 se ajustan para cada N.
for kind in ("arbol", "lineal"):
    out = {}
    for N in N_STRONG:
        d = strong("v3", N)
        n, m, dd = int(d.n_train[0]), int(d.m_test[0]), int(d.d[0])
        Wn = 3 * m * n * dd
        K, c = comm_model(kind, d.s.values, n, m, dd)
        A = np.c_[Wn / d.s.values, K]
        t = d.T_total.values
        w = 1 / t                                   # error relativo
        (k1, k2), _ = nnls(A * w[:, None], t * w)
        err = np.abs(A @ np.array([k1, k2]) - t) / t
        if k2 <= 0:
            s_star = float("inf")
        elif kind == "arbol":
            s_star = k1 * Wn * np.log(2) / (k2 * c)
        else:
            s_star = np.sqrt(k1 * Wn / (k2 * c))
        out[f"N{N}"] = dict(k1=k1, k2=k2, GFLOPs_equivalente=1 / k1 / 1e9,
                            err_rel_medio=float(err.mean()), W=Wn, c=float(c),
                            s_star=float(s_star),
                            s_best_medido=int(d.s[np.argmin(d.T_total)]))
    fit["normalizacion"][kind] = out

fig, axes = plt.subplots(1, 2, figsize=(W2, 2.5))
ss = np.linspace(1, 24, 200)
for ax, N in zip(axes, (N_STRONG[0], NMAX)):
    d = strong("v3", N)
    n, m, dd = int(d.n_train[0]), int(d.m_test[0]), int(d.d[0])
    Wn = 3 * m * n * dd
    ax.plot(d.s, d.T_total, color=C["ink"], marker="o", ls="none", label="medido", zorder=3)
    for kind, col, ls, lab in (("arbol", C["blue"], "--", r"árbol $\log_2 s$"),
                               ("lineal", C["orange"], "-", r"lineal $(s-1)$")):
        q = fit["normalizacion"][kind][f"N{N}"]
        K, _ = comm_model(kind, ss, n, m, dd)
        ax.plot(ss, q["k1"] * Wn / ss + q["k2"] * K, color=col, ls=ls,
                label=r"$k_1W/s+k_2T_{comm}$, " + lab)
    q = fit["normalizacion"]["lineal"][f"N{N}"]
    sl = q["s_star"]
    K, _ = comm_model("lineal", sl, n, m, dd)
    Tl = q["k1"] * Wn / sl + q["k2"] * K
    ax.plot([sl], [Tl], marker="v", color=C["orange"], ms=7, ls="none")
    ax.axvline(sl, color=C["orange"], ls=":", lw=0.9, zorder=1)
    ax.text(sl + 0.4, 0.62, f"$s^*$ = {sl:.1f}", transform=ax.get_xaxis_transform(),
            fontsize=6.8, color=C["ink"], ha="left")
    ax.set_yscale("log"); sticks(ax); hw_bands(ax)
    ax.set_xlabel("procesos $s$"); ax.set_ylabel(r"$T_p(s)$ [s] (log)")
    ax.set_title(f"Normalización teórica, Beta 3, $N$={N // 1000}k")
axes[1].legend(loc="upper right", fontsize=6.3)
save(fig, "f6_modelo")

# =========================================================================
# (M3) Escalabilidad débil y Ley de Gustafson
# =========================================================================
def gustafson(s, fs):
    return fs + s * (1 - fs)


fig, axes = plt.subplots(1, 2, figsize=(W2, 2.4))
fit["weak"] = {}
for (v, lab), (col, mk, ls) in zip((("v3", "Beta 3"), ("dag", "DAG")), SERIES[:2]):
    for tag, ax in (("weak-work", axes[0]), ("weak-lin", axes[1])):
        d = MED[(MED.version == v) & (MED.tag == tag)].sort_values("s")
        W = 3 * d.m_test * d.n_train * d.d
        W1_ = float(W.iloc[0]); T1 = float(d.T_total.iloc[0])
        Sg = ((W / W1_) * T1 / d.T_total).values      # speedup escalado medido
        (fs,), _ = curve_fit(gustafson, d.s, Sg, p0=[0.5], bounds=(0, 1))
        m8 = d.s.values <= 8
        (fs8,), _ = curve_fit(gustafson, d.s[m8], Sg[m8], p0=[0.5], bounds=(0, 1))
        fit["weak"][f"{v}_{tag}"] = dict(gustafson_fs_P=fs8,
                                         s=d.s.tolist(), N=d.N.tolist(),
                                         T=d.T_total.tolist(), E=(Sg / d.s.values).tolist(),
                                         S_escalado=Sg.tolist(), gustafson_fs=fs)
        ax.plot(d.s, Sg, color=col, marker=mk, ls="none", label=f"medido, {lab}")
        if v == "v3":
            sx = np.linspace(1, 24, 100)
            ax.plot(sx, gustafson(sx, fs), color=C["ink"], ls="--", lw=1.1,
                    label=f"Gustafson ($s\\leq 24$): $f_s$={fs:.2f}")
            ax.plot(sx, gustafson(sx, fs8), color=C["gray"], ls=":", lw=1.2,
                    label=f"Gustafson ($s\\leq 8$): $f_s$={fs8:.2f}")
for ax, t in zip(axes, (r"$N = N_0\sqrt{s}$ (trabajo por proceso constante)",
                        r"$N = N_0\,s$ ($N/s$ constante)")):
    ax.plot(S_TICKS, S_TICKS, color=C["light"], lw=1, label="ideal $S=s$")
    sticks(ax); hw_bands(ax); ax.set_xlabel("procesos $s$"); ax.set_ylim(0, 25)
    ax.set_ylabel(r"speedup escalado $\frac{W(s)}{W(1)}\frac{T(1)}{T(s)}$")
    ax.set_title(t)
h0, l0 = axes[0].get_legend_handles_labels()
fig.legend(h0, l0, loc="lower center", ncol=5, fontsize=6.3, bbox_to_anchor=(0.5, -0.06))
save(fig, "f7_debil")

# =========================================================================
# (7) Pinning P vs E vs SMT
# =========================================================================
cfgs = ["P1", "E1", "SMT2", "P4", "E4", "P8", "E8", "PSMT8", "P8+E4", "P8+E8", "P16smt", "ALL24"]
labels = ["1P", "1E", "1P\n×2", "4P", "4E", "8P", "8E", "4P\n×2",
          "8P\n+4E", "8P\n+8E", "8P\n×2", "24\nhilos"]
fig, axes = plt.subplots(1, 2, figsize=(W2, 2.4), gridspec_kw=dict(width_ratios=[1.5, 1]))
x = np.arange(len(cfgs))
vals = [fit["pinning"][c]["gflops_agregado"] for c in cfgs]
cols = [C["blue"] if c.startswith("P") and "SMT" not in c and "E" not in c else
        C["green"] if c.startswith("E") else
        C["pink"] if "SMT" in c or "smt" in c else C["orange"] for c in cfgs]
axes[0].bar(x, vals, color=cols, width=0.72, edgecolor="white", linewidth=1)
for xi, v_ in zip(x, vals):
    axes[0].text(xi, v_ + 1, f"{v_:.0f}", ha="center", fontsize=6)
axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=6)
axes[0].set_ylabel("GFLOP/s agregados"); axes[0].grid(axis="x", visible=False)
axes[0].set_title("Carga fija por proceso, afinidad explícita")
from matplotlib.patches import Patch
axes[0].legend(handles=[Patch(color=C["blue"], label="P-cores"), Patch(color=C["green"], label="E-cores"),
                        Patch(color=C["pink"], label="SMT"), Patch(color=C["orange"], label="mixto")],
               loc="upper left", fontsize=6)
for c, (col, mk, ls), lab in (("P", SERIES[0], "P-cores"), ("E", SERIES[2], "E-cores")):
    xs_ = [1, 4, 8]
    ys_ = [pin[(pin.config == f"{c}{k}")].gflops.median() for k in xs_]
    axes[1].plot(xs_, ys_, color=col, marker=mk, ls=ls, label=lab)
axes[1].set_xticks([1, 4, 8]); axes[1].set_xlabel("núcleos activos del mismo tipo")
axes[1].set_ylabel("GFLOP/s por proceso"); axes[1].set_ylim(0, 14)
axes[1].set_title("Rendimiento por núcleo bajo contención"); axes[1].legend()
save(fig, "f8_pinning")

# =========================================================================
# Tablas LaTeX
# =========================================================================
def fmt(x, n=3):
    return f"{x:.{n}f}"


lines = []
# Tabla de versiones
lines += [r"% --- Tabla: registro de versiones (load_digits) ---",
          r"\begin{tabular}{lrrrrr}", r"\toprule",
          r"Versión & $s$ & $T_{\text{total}}$ [s] & Speedup vs.\ sec. & Exactitud & Validación \\",
          r"\midrule",
          rf"Secuencial (original) & 1 & {fmt(t_seq)} & 1.00 & 0.9833 & referencia \\"]
names = dict(v1="Beta 1", v2="Beta 2", v3="Beta 3", dag="DAG")
for v in ("v1", "v2", "v3", "dag"):
    for s in (1, 8):
        r = V[(V.version == v) & (V.s == s)].iloc[0]
        val = "= sec." if v in ("v1", "v2") else r"100\,\% = sklearn"
        lines.append(rf"{names[v]} & {s} & {r.T_total:.4f} & {t_seq / r.T_total:.1f} & {r.accuracy:.4f} & {val} \\")
lines += [r"\bottomrule", r"\end{tabular}", ""]

# Tabla de escalabilidad fuerte N máximo
lines += [rf"% --- Tabla: escalabilidad fuerte, N = {NMAX} ---",
          r"\begin{tabular}{rrrrrrr}", r"\toprule",
          r"$s$ & $T_{\text{total}}$ [s] & $T_{\text{comp}}$ [s] & $T_{\text{comm}}$ [s] & $S(s)$ & $E(s)$ & GFLOP/s \\",
          r"\midrule"]
p = fit["speedup"][f"v3_N{NMAX}"]
d = strong("v3", NMAX)
for i, r in d.iterrows():
    lines.append(rf"{int(r.s)} & {r.T_total:.2f} & {r.T_comp:.2f} & {r.T_comm:.3f} & "
                 rf"{p['S'][i]:.2f} & {p['E'][i]:.2f} & {r.gflops:.1f} \\")
lines += [r"\bottomrule", r"\end{tabular}", ""]

# Tabla de constantes ajustadas
na, nl = fit["normalizacion"]["arbol"][f"N{NMAX}"], fit["normalizacion"]["lineal"][f"N{NMAX}"]
cb = comm["Bcast"]
lines += [r"% --- Tabla: constantes ajustadas ---",
          r"\begin{tabular}{llr}", r"\toprule", r"Modelo & Constante & Valor \\", r"\midrule",
          rf"Bcast, árbol binario & $\alpha$ [s];\ $\beta$ [s/palabra] & {cb['arbol']['alpha']:.2e};\ {cb['arbol']['beta']:.2e} \\",
          rf" & $R^2$ & {cb['arbol']['R2']:.3f} \\",
          rf"Bcast, lineal & $\alpha$ [s];\ $\beta$ [s/palabra] & {cb['lineal']['alpha']:.2e};\ {cb['lineal']['beta']:.2e} \\",
          rf" & $R^2$ & {cb['lineal']['R2']:.3f} \\",
          r"\midrule",
          rf"$T_p(s)$, $T_{{comm}}$ en árbol & $k_1$ [s/FLOP];\ $k_2$ & {na['k1']:.2e};\ {na['k2']:.2f} \\",
          rf" & error relativo medio & {na['err_rel_medio'] * 100:.1f}\,\% \\",
          rf" & $s^*=k_1W\ln 2/(k_2c)$ ($N$={NMAX // 1000}k) & {na['s_star']:.0f} \\",
          rf"$T_p(s)$, $T_{{comm}}$ lineal & $k_1$ [s/FLOP];\ $k_2$ & {nl['k1']:.2e};\ {nl['k2']:.2f} \\",
          rf" & error relativo medio & {nl['err_rel_medio'] * 100:.1f}\,\% \\",
          rf" & $s^*=\sqrt{{k_1W/(k_2c)}}$ ($N$={NMAX // 1000}k) & {nl['s_star']:.1f} \\",
          rf"Experimental & $\arg\min_s T_p(s)$ ($N$={NMAX // 1000}k) & {nl['s_best_medido']} \\",
          r"\midrule"]
for v in ("v3", "dag"):
    q = fit["speedup"][f"{v}_N{NMAX}"]
    lines += [rf"Amdahl ({names[v]}), $1\le s\le 24$ & $f_s$;\ $S_{{max}}=1/f_s$ & {q['amdahl_fs']:.3f};\ {q['amdahl_Smax']:.1f} \\",
              rf"Amdahl ({names[v]}), $s\le 8$ & $f_s$;\ $S_{{max}}=1/f_s$ & {q['amdahl_fs_P']:.3f};\ {q['amdahl_Smax_P']:.1f} \\"]
for tag, lab in (("weak-work", r"$N\propto\sqrt{s}$"), ("weak-lin", r"$N\propto s$")):
    lines.append(rf"Gustafson (Beta 3), {lab} & $f_s$ & {fit['weak'][f'v3_{tag}']['gustafson_fs']:.3f} \\")
lines += [r"\bottomrule", r"\end{tabular}"]
with open(os.path.join(RES, "tables.tex"), "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
print("  results/tables.tex")


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
print("  results/model_fit.json")

# Resumen en consola
for kind, qq in fit["normalizacion"].items():
    for N, q in qq.items():
        print(f"[{kind} {N}] k1={q['k1']:.3e} k2={q['k2']:.3f} err={q['err_rel_medio']:.1%} "
              f"s*={q['s_star']:.1f} medido={q['s_best_medido']}")
for k, q in fit["speedup"].items():
    print(f"{k:12s} fs={q['amdahl_fs']:.3f}+-{q['amdahl_fs_err']:.3f} R2={q['amdahl_R2']:.3f} "
          f"Smax={q['amdahl_Smax']:.1f}  fs(s<=8)={q['amdahl_fs_P']:.3f}  S_max_med={q['S_max_medido']:.2f}")
for k, q in fit["weak"].items():
    print(f"{k:14s} fs_G={q['gustafson_fs']:.3f} fs_G8={q['gustafson_fs_P']:.3f} S=" + " ".join(f"{e:.1f}" for e in q["S_escalado"]))
print(f"Bcast R2 arbol={comm['Bcast']['arbol']['R2']:.3f} lineal={comm['Bcast']['lineal']['R2']:.3f}")
