"""
Reproducible analysis for:

    Odibo, A.O. (2017). In-vitro evaluation of the anti-sickling potential,
    ascorbic acid and phytochemical constituents of natural fruit extracts
    (Musa acuminata, Malus domestica and Citrus paradisi).
    Pharm.D. research project, University of Benin, Nigeria.

Recomputes every derived number in Chapter 4 from the tabulated data in ../data
and redraws the figures. Run from the repository root:

    python src/analysis.py

Outputs go to ./results (tables as CSV, figures as PNG, and a text report).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

FRUITS = ["M. acuminata", "C. paradisi", "M. domestica"]
COLOURS = {"M. acuminata": "#C0962F", "C. paradisi": "#D97D3F", "M. domestica": "#4D8F51"}
ALPHA = 0.05

# Institute of Medicine (2000) RDAs for vitamin C, mg/day
RDI = {"adult_men_90mg": 90, "adult_women_75mg": 75, "children_9_13y_45mg": 45}

report_lines: list[str] = []


def log(text: str = "") -> None:
    print(text)
    report_lines.append(text)


def section(title: str) -> None:
    log()
    log("=" * 72)
    log(title)
    log("=" * 72)


def italic(name: str) -> str:
    return f"$\\it{{{name.replace(' ', '~')}}}$"


# ---------------------------------------------------------------------------
# 1. Phytochemicals (Tables 4.3-4.5, Figures 4.1-4.3)
# ---------------------------------------------------------------------------
def anova_from_summary(means, sds, ns):
    """One-way ANOVA computed from group means, SDs and sizes."""
    means, sds, ns = map(np.asarray, (means, sds, ns))
    k, N = len(means), ns.sum()
    grand = np.sum(ns * means) / N
    ss_between = np.sum(ns * (means - grand) ** 2)
    ss_within = np.sum((ns - 1) * sds**2)
    df_b, df_w = k - 1, N - k
    ms_w = ss_within / df_w
    if ms_w == 0:
        return np.inf, 0.0, int(df_b), int(df_w)
    F = (ss_between / df_b) / ms_w
    return F, stats.f.sf(F, df_b, df_w), int(df_b), int(df_w)


def phytochemicals():
    section("1. PHYTOCHEMICAL COMPOSITION (Tables 4.3-4.5)")
    df = pd.read_csv(DATA / "phytochemicals.csv")
    order = ["M. acuminata", "M. domestica", "C. paradisi"]
    constituents = ["Alkaloids", "Flavonoids", "Saponins", "Phenolics", "Tannins"]

    wide = df.pivot(index="fruit", columns="constituent", values="mean_mg_g").loc[order, constituents]
    log("Mean content (mg/g):")
    log(wide.to_string())

    # ANOVA from summary statistics.
    # NOTE: SDs reported as 0.00 are rounded, so F values here are approximate
    # and can differ greatly from those in Table 4.3; the p < 0.05 conclusion
    # is what matters.
    log("\nOne-way ANOVA across fruits (from mean, SD, n; approximate):")
    rows = []
    for c in constituents:
        sub = df[df.constituent == c].set_index("fruit").loc[order]
        F, p, d1, d2 = anova_from_summary(sub.mean_mg_g, sub.sd_mg_g, sub.n)
        rows.append({"constituent": c, "F": F, "df": f"({d1}, {d2})", "p": p})
        log(f"  {c:<11} F{d1, d2} = {F:,.2f}   p = {p:.2e}")
    pd.DataFrame(rows).to_csv(OUT / "table_4_3_anova.csv", index=False)

    # Rank order (Table 4.4)
    log("\nRank order, highest -> lowest (Table 4.4):")
    for c in constituents:
        log(f"  {c:<11} " + " > ".join(wide[c].sort_values(ascending=False).index))

    # Percentage occurrence (Table 4.5)
    pct = wide.div(wide.sum(axis=1), axis=0) * 100
    log("\nPercentage occurrence within each fruit (Table 4.5):")
    log(pct.round(2).to_string())
    pct.round(2).to_csv(OUT / "table_4_5_percent_occurrence.csv")

    # Figures 4.1-4.3
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharex=False)
    for ax, fruit in zip(axes, order):
        vals = pct.loc[fruit]
        ax.barh(vals.index, vals.values, color="#555555")
        for y, v in enumerate(vals.values):
            ax.text(v + 1, y, f"{v:.2f}%", va="center", fontsize=8)
        ax.set_title(italic(fruit))
        ax.set_xlim(0, 100)
        ax.set_xlabel("Occurrence (%)")
    fig.suptitle("Percentage occurrence of quantified phytochemicals (Figures 4.1-4.3)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_4_1_to_4_3_phytochemicals.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Ascorbic acid calibration and content (Tables 4.7-4.8, Figure 4.4)
# ---------------------------------------------------------------------------
def ascorbic_acid():
    section("2. ASCORBIC ACID (Tables 4.7-4.8)")
    cal = pd.read_csv(DATA / "calibration.csv")
    A0 = cal.loc[cal.ascorbic_acid_ug_ml == 0, "absorbance_515nm"].iloc[0]
    cal["decrease"] = A0 - cal.absorbance_515nm
    cal["pct_decrease"] = cal.decrease / A0 * 100
    log("Calibration data (Table 4.7):")
    log(cal.round(3).to_string(index=False))
    cal.round(3).to_csv(OUT / "table_4_7_calibration.csv", index=False)

    fit = stats.linregress(cal.ascorbic_acid_ug_ml, cal.pct_decrease)
    r2 = fit.rvalue**2
    log(f"\nLinear fit: y = {fit.slope:.4f}x {fit.intercept:+.4f}   (R^2 = {r2:.4f})")

    # Inverse prediction helper (Appendix I, example c)
    example = 40
    log(f"Example: {example}% decrease -> {(example - fit.intercept) / fit.slope:.2f} ug/ml")

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(cal.ascorbic_acid_ug_ml, cal.pct_decrease, color="black", zorder=3)
    xs = np.linspace(0, 55, 100)
    ax.plot(xs, fit.slope * xs + fit.intercept, color="grey")
    ax.text(2, 90, f"y = {fit.slope:.4f}x {fit.intercept:+.4f}\n$R^2$ = {r2:.4f}")
    ax.set_xlabel("Ascorbic acid (µg/ml)")
    ax.set_ylabel("Absorbance decrease at 515 nm (%)")
    ax.set_title("Ascorbic acid calibration curve (Figure 4.4)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_4_4_calibration_curve.png", dpi=200)
    plt.close(fig)

    aa = pd.read_csv(DATA / "ascorbic_acid.csv")
    for col, rdi in RDI.items():
        aa[f"pct_rdi_{col}"] = aa.ascorbic_acid_mg_100g / rdi * 100
    log("\nAscorbic acid content and % RDI per 100 g (Table 4.8), with pH (Table 4.6):")
    log(aa.round(1).to_string(index=False))
    aa.round(2).to_csv(OUT / "table_4_8_ascorbic_acid_rdi.csv", index=False)

    cp = aa.set_index("fruit").ascorbic_acid_mg_100g
    log(
        f"\nC. paradisi / M. acuminata = {cp['C. paradisi'] / cp['M. acuminata']:.1f}x;  "
        f"C. paradisi / M. domestica = {cp['C. paradisi'] / cp['M. domestica']:.1f}x"
    )


# ---------------------------------------------------------------------------
# 3. Antisickling activity (Tables 4.9-4.12, Figures 4.5-4.9)
# ---------------------------------------------------------------------------
def antisickling():
    section("3. IN-VITRO ANTISICKLING ACTIVITY (Tables 4.9-4.12)")
    df = pd.read_csv(DATA / "antisickling.csv")
    ctrl = pd.read_csv(DATA / "controls.csv")

    # Table 4.9 means across time for each concentration
    log("Mean % unsickled across time, by concentration (Table 4.9):")
    conc_means = df.pivot_table(index="extract", columns="concentration_mg_ml",
                                values="percent_unsickled").loc[FRUITS]
    log(conc_means.round(2).to_string())

    # Table 4.10 summary statistics
    rows = []
    for f in FRUITS:
        v = df.loc[df.extract == f, "percent_unsickled"]
        by_t = df[df.extract == f].groupby("time_min").percent_unsickled.mean()
        best_c = conc_means.loc[f].idxmax()
        rows.append({
            "extract": f,
            "n": len(v),
            "mean": v.mean(),
            "sd": v.std(ddof=1),
            "min": v.min(),
            "max": v.max(),
            "cv_percent": v.std(ddof=1) / v.mean() * 100,
            "n_conditions_ge_95": int((v >= 95).sum()),
            **{f"mean_T{t}": by_t[t] for t in by_t.index},
            "best_conc_mg_ml": best_c,
            "best_conc_mean": conc_means.loc[f, best_c],
        })
    summary = pd.DataFrame(rows).set_index("extract")
    log("\nSummary statistics (Table 4.10):")
    log(summary.round(2).T.to_string())
    summary.round(2).to_csv(OUT / "table_4_10_summary.csv")
    log(f"\nNormal saline mean T0-T90 = {ctrl.normal_saline.mean():.2f}%; PHBA = {ctrl.phba_5mg_ml.mean():.0f}%")

    # Table 4.11 one-way ANOVA + Tukey HSD
    groups = [df.loc[df.extract == f, "percent_unsickled"] for f in FRUITS]
    F, p = stats.f_oneway(*groups)
    dfb, dfw = len(FRUITS) - 1, len(df) - len(FRUITS)
    log(f"\nOne-way ANOVA: F({dfb}, {dfw}) = {F:.2f}, p = {p:.4f}")
    H, pk = stats.kruskal(*groups)
    log(f"Kruskal-Wallis: H = {H:.2f}, p = {pk:.4f}")

    tukey = pairwise_tukeyhsd(df.percent_unsickled, df.extract, alpha=ALPHA)
    tk = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
    log("\nTukey HSD (meandiff = group2 - group1):")
    log(tk.to_string(index=False))
    tk.to_csv(OUT / "table_4_11_tukey.csv", index=False)

    # Normality / variance checks (assumptions behind the ANOVA)
    log("\nAssumption checks:")
    for f, g in zip(FRUITS, groups):
        w, pw = stats.shapiro(g)
        log(f"  Shapiro-Wilk {f:<13} W = {w:.3f}, p = {pw:.3f}")
    lev, pl = stats.levene(*groups)
    log(f"  Levene's test            W = {lev:.3f}, p = {pl:.3f}")

    # Table 4.12 two-way ANOVA without replication within each extract.
    # With one value per cell, the interaction is the residual (error) term.
    log("\nTwo-way ANOVA without replication (Table 4.12):")
    tw_rows = []
    for f in FRUITS:
        sub = df[df.extract == f]
        model = smf.ols("percent_unsickled ~ C(concentration_mg_ml) + C(time_min)", data=sub).fit()
        a = anova_lm(model, typ=2)
        rc, rt = a.loc["C(concentration_mg_ml)"], a.loc["C(time_min)"]
        dfr = int(a.loc["Residual", "df"])
        tw_rows.append({
            "extract": f,
            "conc_F": rc.F, "conc_p": rc["PR(>F)"],
            "time_F": rt.F, "time_p": rt["PR(>F)"],
            "df": f"(3, {dfr})",
        })
        log(f"  {f:<13} concentration F(3,{dfr}) = {rc.F:.2f}, p = {rc['PR(>F)']:.3f};  "
            f"time F(3,{dfr}) = {rt.F:.2f}, p = {rt['PR(>F)']:.3f}")
    pd.DataFrame(tw_rows).round(3).to_csv(OUT / "table_4_12_two_way_anova.csv", index=False)

    # Comparison with negative control at T90
    t90 = summary["mean_T90"]
    diff = t90 - ctrl.set_index("time_min").loc[90, "normal_saline"]
    log("\nT90 advantage over normal saline (percentage points): "
        + ", ".join(f"{f} {d:.2f}" for f, d in diff.items()))

    # ---- Figures 4.5-4.7: per-extract time course vs controls
    concs = sorted(df.concentration_mg_ml.unique())
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=True)
    for ax, f in zip(axes, FRUITS):
        for c in concs:
            s = df[(df.extract == f) & (df.concentration_mg_ml == c)].sort_values("time_min")
            ax.plot(s.time_min, s.percent_unsickled, marker="o", label=f"{c} mg/ml")
        ax.plot(ctrl.time_min, ctrl.normal_saline, marker="x", ls=":", color="grey", label="Normal saline")
        ax.plot(ctrl.time_min, ctrl.phba_5mg_ml, marker="D", ls="--", color="black", label="PHBA 5 mg/ml")
        ax.set_title(italic(f))
        ax.set_xlabel("Time after deoxygenation (min)")
        ax.set_xticks([0, 30, 60, 90])
        ax.set_ylim(0, 105)
    axes[0].set_ylabel("Unsickled erythrocytes (%)")
    axes[-1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Sickling inhibition by each extract vs controls (Figures 4.5-4.7)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_4_5_to_4_7_time_courses.png", dpi=200)
    plt.close(fig)

    # ---- Figure 4.8: mean time course with concentration range bands
    fig, ax = plt.subplots(figsize=(8, 5.5))
    markers = {"M. acuminata": "o", "C. paradisi": "s", "M. domestica": "^"}
    for f in FRUITS:
        g = df[df.extract == f].groupby("time_min").percent_unsickled
        m, lo, hi = g.mean(), g.min(), g.max()
        ax.fill_between(m.index, lo, hi, color=COLOURS[f], alpha=0.15)
        ax.plot(m.index, m.values, marker=markers[f], color=COLOURS[f],
                label=f"{italic(f)} (mean of 1-50 mg/ml)")
    ax.plot(ctrl.time_min, ctrl.phba_5mg_ml, "k--D", label="PHBA 5 mg/ml (positive control)")
    ax.plot(ctrl.time_min, ctrl.normal_saline, ":x", color="dimgrey", label="Normal saline (negative control)")
    ax.set_xticks([0, 30, 60, 90])
    ax.set_ylim(0, 105)
    ax.set_xlabel("Incubation time after deoxygenation (minutes)")
    ax.set_ylabel("Unsickled erythrocytes (%)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.set_title("Mean antisickling activity over time (Figure 4.8)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_4_8_mean_time_course.png", dpi=200)
    plt.close(fig)

    # ---- Figure 4.9: overall mean +/- SD with Tukey letters
    letters = compact_letters(tukey, FRUITS, summary["mean"])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    x = np.arange(len(FRUITS))
    ax.bar(x, summary.loc[FRUITS, "mean"], yerr=summary.loc[FRUITS, "sd"], capsize=6,
           color=[COLOURS[f] for f in FRUITS], edgecolor="black", width=0.55)
    for i, f in enumerate(FRUITS):
        ax.text(i, summary.loc[f, "mean"] / 2, f"{summary.loc[f, 'mean']:.2f}%",
                ha="center", color="white", fontweight="bold")
        ax.text(i, 103, letters[f], ha="center", fontweight="bold", fontsize=12)
    ax.axhline(ctrl.normal_saline.mean(), ls=":", color="black",
               label=f"Normal saline, mean T0-T90 ({ctrl.normal_saline.mean():.2f}%)")
    ax.axhline(100, ls="--", color="black", label="PHBA 5 mg/ml (100%)")
    ax.set_xticks(x, [italic(f) for f in FRUITS])
    ax.set_ylim(0, 118)
    ax.set_ylabel("Mean unsickled erythrocytes (%) ± SD")
    ax.legend(loc="upper center", ncol=2, fontsize=8, frameon=False)
    ax.set_title("Overall antisickling activity (Figure 4.9)", pad=12)
    fig.tight_layout()
    fig.savefig(FIG / "fig_4_9_overall_means.png", dpi=200)
    plt.close(fig)


def compact_letters(tukey, groups, means):
    """Simple compact-letter display for a small number of groups."""
    res = pd.DataFrame(tukey.summary().data[1:], columns=tukey.summary().data[0])
    sig = {(r.group1, r.group2): bool(r.reject) for r in res.itertuples()}
    sig.update({(b, a): v for (a, b), v in list(sig.items())})
    ordered = sorted(groups, key=lambda g: -means[g])
    letter_sets: list[set] = []
    for g in ordered:
        placed = False
        for s in letter_sets:
            if all(not sig[(g, h)] for h in s):
                s.add(g)
                placed = True
        if not placed:
            letter_sets.append({g})
    out = {g: "" for g in groups}
    for i, s in enumerate(letter_sets):
        for g in s:
            out[g] += "abcdefghij"[i]
    return out


if __name__ == "__main__":
    phytochemicals()
    ascorbic_acid()
    antisickling()
    (OUT / "report.txt").write_text("\n".join(report_lines) + "\n")
    print(f"\nResults written to {OUT}")
