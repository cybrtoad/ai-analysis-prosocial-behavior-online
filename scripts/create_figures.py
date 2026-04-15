"""
Thesis Chapter 4 — Results Figures
Generates Figures 4–10 for the prosocial corrective communication study.
Output: figures/ subdirectory (PNG + PDF for each figure)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), "..")
FIG_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

MAIN_PARQUET = os.path.join(ROOT, "output/phase2/phase2_analysis_dataset.parquet")
WIKI_PARQUET = os.path.join(ROOT, "output/phase2/phase2_analysis_dataset_wikipedia.parquet")
CLAUDE_SCORES = os.path.join(ROOT, "human-validation/claude_scores_EMBARGOED.csv")
HUMAN_TRAIN = os.path.join(ROOT, "output/human_labels/human_train.parquet")
HUMAN_VAL = os.path.join(ROOT, "output/human_labels/human_val.parquet")

# ── Style ─────────────────────────────────────────────────────────────────────
PLATFORM_COLORS = {
    "Reddit": "#FF6B6B",
    "Stack Exchange": "#4ECDC4",
    "Wikipedia": "#95A5A6",
}
DIM_COLORS = {
    "empathy": "#E74C3C",
    "constructiveness": "#3498DB",
    "respect": "#2ECC71",
    "social_cohesion": "#9B59B6",
}
ITYPE_COLORS = {
    "positive_reinforcement": "#27AE60",
    "reframe": "#2980B9",
    "restriction": "#E67E22",
    "warning": "#C0392B",
}
ITYPE_LABELS = {
    "positive_reinforcement": "Positive\nReinforcement",
    "reframe": "Reframe",
    "restriction": "Restriction",
    "warning": "Warning",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 150,
})


def save(fig, name):
    for ext in ("png", "pdf"):
        path = os.path.join(FIG_DIR, f"{name}.{ext}")
        fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"  Saved {name}.png / .pdf")
    plt.close(fig)


# ── Load data ─────────────────────────────────────────────────────────────────
print("Loading data…")
df_main = pd.read_parquet(MAIN_PARQUET)
df_wiki = pd.read_parquet(WIKI_PARQUET)

claude_raw = pd.read_csv(CLAUDE_SCORES)
human_raw = pd.concat([
    pd.read_parquet(HUMAN_TRAIN),
    pd.read_parquet(HUMAN_VAL),
], ignore_index=True)

# Merge human + claude on record_id; drop invalid_triple
merged = (
    claude_raw[["record_id", "empathy", "constructiveness", "respect",
                "social_cohesion", "invalid_triple"]]
    .merge(
        human_raw[["record_id", "empathy", "constructiveness",
                   "respect", "social_cohesion"]],
        on="record_id", suffixes=("_claude", "_human"),
    )
)
merged = merged[merged["invalid_triple"] != True].copy()
print(f"  Matched records for Bland-Altman: {len(merged)}")

DIMS = ["empathy", "constructiveness", "respect", "social_cohesion"]
DIM_LABELS = ["Empathy", "Constructiveness", "Respect", "Social\nCohesion"]


# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Prosociality Composite Distribution by Platform (violin)
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 4…")

reddit = df_main[df_main["platform"] == "reddit"]["prosociality_composite"].dropna()
se = df_main[df_main["platform"] == "stackexchange"]["prosociality_composite"].dropna()
wiki = df_wiki["prosociality_composite"].dropna()

platform_data = [
    ("Reddit", reddit),
    ("Stack Exchange", se),
    ("Wikipedia", wiki),
]

fig, ax = plt.subplots(figsize=(8, 5.5))

positions = [1, 2, 3]
violins = ax.violinplot(
    [d.values for _, d in platform_data],
    positions=positions,
    showmedians=False,
    showextrema=False,
    widths=0.7,
)

for i, ((label, _), body) in enumerate(zip(platform_data, violins["bodies"])):
    body.set_facecolor(PLATFORM_COLORS[label])
    body.set_edgecolor("white")
    body.set_alpha(0.85)
    body.set_linewidth(1.2)

# Overlay median + IQR box
for i, (label, d) in enumerate(platform_data):
    pos = positions[i]
    med = d.median()
    q1, q3 = d.quantile(0.25), d.quantile(0.75)
    ax.hlines(med, pos - 0.06, pos + 0.06, color="white", linewidth=3.5, zorder=5)
    ax.hlines(med, pos - 0.055, pos + 0.055, color="#2C3E50", linewidth=2, zorder=6)
    ax.vlines(pos, q1, q3, color="#2C3E50", linewidth=5, alpha=0.35, zorder=4)
    n = len(d)
    ax.text(pos, -0.22, f"n = {n:,}", ha="center", va="top",
            fontsize=9, color="#555555")
    ax.text(pos, med + 0.08, f"Mdn = {med:.2f}", ha="center", va="bottom",
            fontsize=9.5, fontweight="bold", color="#2C3E50", zorder=7)

ax.set_xticks(positions)
ax.set_xticklabels([label for label, _ in platform_data], fontsize=12)
ax.set_ylabel("Prosociality Composite Score", fontsize=12)
ax.set_ylim(-0.35, 5.6)
ax.set_xlim(0.4, 3.6)
ax.set_yticks([1, 2, 3, 4, 5])
ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax.set_axisbelow(True)

# Annotation: Δmedian
ax.annotate(
    "", xy=(2, 3.5 + 0.12), xytext=(1, 2.5 + 0.12),
    arrowprops=dict(arrowstyle="<->", color="#2C3E50", lw=1.5),
)
ax.text(1.5, 3.25, "Δ Mdn = 1.0", ha="center", fontsize=9.5,
        color="#2C3E50", style="italic")

# Wikipedia floor-effect note
ax.text(3, 0.95, "Floor\neffect", ha="center", fontsize=8.5,
        color="#7F8C8D", style="italic")

fig.suptitle(
    "Figure 4. Prosociality Composite Distribution by Platform",
    fontsize=13, fontweight="bold", y=1.01,
)
ax.set_title(
    "Violin plot with median (line) and IQR (shaded bar). "
    "Reddit n = 4,660; Stack Exchange n = 90,076; Wikipedia n = 4,770.",
    fontsize=9, color="#555555", pad=6,
)
save(fig, "figure4_platform_violin")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Within-Reddit Intervention Type Comparison
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 5…")

reddit_df = df_main[df_main["platform"] == "reddit"].copy()

bar_data = []
for itype in ["reframe", "positive_reinforcement"]:
    g = reddit_df[reddit_df["intervention_type"] == itype]["prosociality_composite"].dropna()
    m = g.mean()
    ci_lo, ci_hi = stats.t.interval(0.95, len(g) - 1, loc=m, scale=stats.sem(g))
    bar_data.append({
        "label": "Reframe" if itype == "reframe" else "Positive\nReinforcement",
        "mean": m,
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "n": len(g),
        "color": ITYPE_COLORS[itype],
    })

fig, ax = plt.subplots(figsize=(5.5, 5))

for i, row in enumerate(bar_data):
    bar = ax.bar(i, row["mean"], color=row["color"], alpha=0.85,
                 width=0.5, edgecolor="white", linewidth=1.2)
    err_lo = row["mean"] - row["ci_lo"]
    err_hi = row["ci_hi"] - row["mean"]
    ax.errorbar(i, row["mean"], yerr=[[err_lo], [err_hi]],
                fmt="none", color="#2C3E50", capsize=6, capthick=2, linewidth=2)
    ax.text(i, row["mean"] + err_hi + 0.06, f'{row["mean"]:.3f}',
            ha="center", fontsize=10.5, fontweight="bold")
    ax.text(i, -0.16, f'n = {row["n"]:,}', ha="center",
            fontsize=9, color="#555555")

ax.set_xticks(range(len(bar_data)))
ax.set_xticklabels([row["label"] for row in bar_data], fontsize=12)
ax.set_ylabel("Mean Prosociality Composite Score", fontsize=11)
ax.set_ylim(-0.25, 3.5)
ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax.set_axisbelow(True)

# Null result annotation
ax.annotate("", xy=(1, 2.9), xytext=(0, 2.9),
            arrowprops=dict(arrowstyle="<->", color="#7F8C8D", lw=1.5))
ax.text(0.5, 2.97, "n.s.", ha="center", fontsize=11, color="#7F8C8D")

fig.suptitle(
    "Figure 5. Within-Reddit Intervention Type Comparison",
    fontsize=13, fontweight="bold", y=1.01,
)
ax.set_title(
    "Bars = mean prosociality composite. Error bars = 95% CI.\n"
    "Reddit only (reframe n = 2,414; positive reinforcement n = 2,246).",
    fontsize=9, color="#555555", pad=6,
)
save(fig, "figure5_reddit_intervention_bar")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 6 — Tier D Departure Rates by Intervention Type
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 6…")

tier_d = [
    ("positive_reinforcement", 3.8,  13293,   503),
    ("reframe",                4.1, 388570, 15819),
    ("restriction",           11.0,   7420,   819),
    ("warning",               36.8,  14389,  5290),
]

fig, ax = plt.subplots(figsize=(7, 5))

for i, (itype, rate, total, nd) in enumerate(tier_d):
    color = ITYPE_COLORS[itype]
    ax.bar(i, rate, color=color, alpha=0.85, width=0.6,
           edgecolor="white", linewidth=1.2)
    ax.text(i, rate + 0.6, f"{rate}%", ha="center", fontsize=11,
            fontweight="bold", color="#2C3E50")
    ax.text(i, -2.5, f"n = {total:,}", ha="center", fontsize=8.5,
            color="#555555")
    ax.text(i, -3.8, f"({nd:,} Tier D)", ha="center", fontsize=8,
            color="#7F8C8D")

ax.set_xticks(range(len(tier_d)))
ax.set_xticklabels(
    [ITYPE_LABELS[itype] for itype, *_ in tier_d], fontsize=11
)
ax.set_ylabel("Departure Rate (%)", fontsize=11)
ax.set_ylim(-5.5, 44)
ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax.set_axisbelow(True)

# Highlight warning bar annotation
ax.annotate(
    "Survivorship\nbias risk",
    xy=(3, 36.8), xytext=(2.35, 40.5),
    fontsize=9, color="#C0392B",
    arrowprops=dict(arrowstyle="->", color="#C0392B", lw=1.2),
    ha="center",
)

fig.suptitle(
    "Figure 6. Tier D Departure Rates by Intervention Type",
    fontsize=13, fontweight="bold", y=1.01,
)
ax.set_title(
    "Percentage of records with no meaningful post-intervention response\n"
    "(n over each bar = total records per type).",
    fontsize=9, color="#555555", pad=6,
)
save(fig, "figure6_tier_d_departure_rates")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 7 — Inter-Dimension Correlation Heatmap
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 7…")

# Main dataset correlation matrix (Table 35)
corr_vals = np.array([
    [1.000, 0.816, 0.794, 0.937],
    [0.816, 1.000, 0.697, 0.842],
    [0.794, 0.697, 1.000, 0.890],
    [0.937, 0.842, 0.890, 1.000],
])
corr_df = pd.DataFrame(
    corr_vals,
    index=["Empathy", "Constructiveness", "Respect", "Social\nCohesion"],
    columns=["Empathy", "Constructiveness", "Respect", "Social\nCohesion"],
)

fig, ax = plt.subplots(figsize=(6, 5.5))

cmap = LinearSegmentedColormap.from_list(
    "corr_cmap", ["#EBF5FB", "#2980B9", "#1A5276"], N=256
)
im = ax.imshow(corr_vals, cmap=cmap, vmin=0.6, vmax=1.0, aspect="auto")

ax.set_xticks(range(4))
ax.set_yticks(range(4))
ax.set_xticklabels(corr_df.columns, fontsize=11)
ax.set_yticklabels(corr_df.index, fontsize=11)

# Annotate cells
for i in range(4):
    for j in range(4):
        val = corr_vals[i, j]
        text_color = "white" if val > 0.82 else "#1A3A5C"
        weight = "bold" if i != j else "normal"
        ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                fontsize=11.5, color=text_color, fontweight=weight)

cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Pearson r", fontsize=10)
cbar.ax.tick_params(labelsize=9)

fig.suptitle(
    "Figure 7. Inter-Dimension Correlation Heatmap",
    fontsize=13, fontweight="bold", y=1.02,
)
ax.set_title(
    "Pearson r among Tier 1 prosociality dimensions (main dataset, n = 94,736).\n"
    "All correlations p < .001.",
    fontsize=9, color="#555555", pad=6,
)
fig.tight_layout()
save(fig, "figure7_correlation_heatmap")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 8 — PCA Scree Plot
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 8…")

pca_var = [87.3, 7.7, 4.0, 0.9]
pca_labels = ["PC1", "PC2", "PC3", "PC4"]

fig, ax = plt.subplots(figsize=(5.5, 4.5))

bar_colors = ["#2980B9" if i == 0 else "#AED6F1" for i in range(4)]
bars = ax.bar(range(4), pca_var, color=bar_colors, alpha=0.9,
              edgecolor="white", linewidth=1.2, width=0.55)

for i, v in enumerate(pca_var):
    ax.text(i, v + 0.8, f"{v}%", ha="center", va="bottom",
            fontsize=11, fontweight="bold" if i == 0 else "normal")

ax.set_xticks(range(4))
ax.set_xticklabels(pca_labels, fontsize=12)
ax.set_ylabel("Variance Explained (%)", fontsize=11)
ax.set_ylim(0, 100)
ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax.set_axisbelow(True)

# Elbow annotation
ax.annotate(
    "Visual cliff:\nsingle-factor\nstructure",
    xy=(0.5, (pca_var[0] + pca_var[1]) / 2),
    xytext=(1.6, 70),
    fontsize=9, color="#1A5276",
    arrowprops=dict(arrowstyle="->", color="#1A5276", lw=1.2),
    ha="center",
)

fig.suptitle(
    "Figure 8. PCA Scree Plot",
    fontsize=13, fontweight="bold", y=1.01,
)
ax.set_title(
    "Variance explained by each principal component\n"
    "(main dataset, n = 94,736; 4 Tier 1 dimensions).",
    fontsize=9, color="#555555", pad=6,
)
save(fig, "figure8_pca_scree")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 9 — Bland-Altman Plots (2×2 panel)
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 9…")

dim_display = ["Empathy", "Constructiveness", "Respect", "Social Cohesion"]
dim_cols = ["empathy", "constructiveness", "respect", "social_cohesion"]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes = axes.flatten()

for idx, (dim, label) in enumerate(zip(dim_cols, dim_display)):
    ax = axes[idx]
    c = merged[f"{dim}_claude"].values.astype(float)
    h = merged[f"{dim}_human"].values.astype(float)

    mean_scores = (c + h) / 2.0
    diff_scores = c - h

    bias = np.mean(diff_scores)
    sd_diff = np.std(diff_scores, ddof=1)
    loa_lo = bias - 1.96 * sd_diff
    loa_hi = bias + 1.96 * sd_diff

    ax.scatter(mean_scores, diff_scores, alpha=0.35, s=14,
               color=list(DIM_COLORS.values())[idx], edgecolors="none")

    ax.axhline(bias, color="#E74C3C", linewidth=2, linestyle="-",
               label=f"Bias = {bias:+.3f}")
    ax.axhline(loa_lo, color="#95A5A6", linewidth=1.5, linestyle="--",
               label=f"LoA = [{loa_lo:.2f}, {loa_hi:.2f}]")
    ax.axhline(loa_hi, color="#95A5A6", linewidth=1.5, linestyle="--")
    ax.axhline(0, color="#BDC3C7", linewidth=0.8, linestyle=":")

    # Right-side annotations
    xmax = ax.get_xlim()[1] if ax.get_xlim()[1] != 0 else 5.5
    ax.text(5.65, bias, f"{bias:+.3f}", va="center", ha="left",
            fontsize=8.5, color="#E74C3C", fontweight="bold",
            clip_on=False)
    ax.text(5.65, loa_hi, f"{loa_hi:.2f}", va="center", ha="left",
            fontsize=8, color="#7F8C8D", clip_on=False)
    ax.text(5.65, loa_lo, f"{loa_lo:.2f}", va="center", ha="left",
            fontsize=8, color="#7F8C8D", clip_on=False)

    ax.set_title(label, fontsize=12, fontweight="bold")
    ax.set_xlabel("Mean (Claude + Human) / 2", fontsize=9.5)
    ax.set_ylabel("Claude − Human", fontsize=9.5)
    ax.set_xlim(0.5, 5.5)
    ax.yaxis.grid(True, linestyle="--", alpha=0.3)
    ax.set_axisbelow(True)

    bias_dir = "overscores" if bias > 0.05 else ("underscores" if bias < -0.05 else "≈ 0")
    ax.text(0.03, 0.96, f"Claude {bias_dir}", transform=ax.transAxes,
            fontsize=8.5, va="top", color="#555555", style="italic")

fig.suptitle(
    "Figure 9. Bland-Altman Plots for Human–Claude Agreement",
    fontsize=13, fontweight="bold",
)
fig.text(
    0.5, 0.00,
    "Each panel: mean of Claude and human consensus score (x-axis) vs. "
    "Claude − human difference (y-axis).\n"
    "Red line = mean bias; dashed lines = 95% limits of agreement. "
    f"n = {len(merged)} matched records.",
    ha="center", fontsize=9, color="#555555",
)
fig.tight_layout(rect=[0, 0.04, 1, 0.96])
save(fig, "figure9_bland_altman")


# ══════════════════════════════════════════════════════════════════════════════
# Figure 10 — Wikipedia Intervention Type Gradient
# ══════════════════════════════════════════════════════════════════════════════
print("Figure 10…")

wiki_itypes = [
    ("positive_reinforcement", 3.500, 3.257, 756),
    ("reframe",                2.125, 2.459, 473),
    ("restriction",            2.000, 2.441, 1439),
    ("warning",                1.375, 1.958, 2102),
]

fig, ax = plt.subplots(figsize=(7, 5.2))

x = range(len(wiki_itypes))
for i, (itype, median, mean, n) in enumerate(wiki_itypes):
    color = ITYPE_COLORS[itype]
    ax.bar(i, median, color=color, alpha=0.88, width=0.55,
           edgecolor="white", linewidth=1.2, label=ITYPE_LABELS[itype])
    ax.text(i, median + 0.07, f"Mdn = {median:.3f}", ha="center",
            fontsize=10, fontweight="bold", color="#2C3E50")
    ax.text(i, -0.22, f"n = {n:,}", ha="center",
            fontsize=8.5, color="#555555")

ax.set_xticks(range(len(wiki_itypes)))
ax.set_xticklabels(
    [ITYPE_LABELS[itype] for itype, *_ in wiki_itypes], fontsize=11
)
ax.set_ylabel("Median Prosociality Composite Score", fontsize=11)
ax.set_ylim(-0.4, 4.5)
ax.yaxis.grid(True, linestyle="--", alpha=0.4, zorder=0)
ax.set_axisbelow(True)

# Gradient arrow
ax.annotate(
    "", xy=(3, 0.3), xytext=(0, 3.2),
    arrowprops=dict(arrowstyle="->", color="#7F8C8D", lw=1.5,
                    connectionstyle="arc3,rad=0"),
)
ax.text(1.95, 2.5, "Decreasing\nprosociality", ha="center",
        fontsize=9, color="#7F8C8D", style="italic", rotation=-30)

# Effect size annotation
ax.text(0.98, 0.97, r"$\eta^2 = 0.119$ (medium)",
        transform=ax.transAxes, ha="right", va="top",
        fontsize=9.5, color="#2C3E50",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#ECF0F1",
                  edgecolor="#BDC3C7", alpha=0.9))

fig.suptitle(
    "Figure 10. Wikipedia Intervention Type Gradient",
    fontsize=13, fontweight="bold", y=1.01,
)
ax.set_title(
    "Median prosociality composite by intervention type, Wikipedia only (n = 4,770).\n"
    r"Kruskal-Wallis: H(3) = 571.84, p < .001, $\eta^2$ = 0.119.",
    fontsize=9, color="#555555", pad=6,
)
save(fig, "figure10_wikipedia_gradient")


print("\nAll figures saved to figures/")
print(os.listdir(FIG_DIR))
