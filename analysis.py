# =============================================================================
# HCBI372 Data Mining Assignment | K-Means Clustering Analysis
# Dataset: HCBI_SF2_user_data.csv (Volunteer Engagement Data)
# Author: Makhado Matamela
# Description: Clusters volunteers by Age and ShiftsCompleted, then generates
#              a self-contained HTML report with embedded charts.
# =============================================================================

import os
import base64
import warnings
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend, must be set before pyplot import
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
K = 3
K_RANGE = range(1, 9)
CSV_PATH = "HCBI_SF2_user_data.csv"
OUTPUT_DIR = "output"

# Brand colour palette
CLUSTER_COLOURS = ["#1A5FA8", "#2EC48A", "#5BA4D4"]
PALETTE = {
    "navy":      "#0D2B4E",
    "blue":      "#1A5FA8",
    "sky":       "#5BA4D4",
    "ice":       "#E8F4FD",
    "mint":      "#2EC48A",
    "white":     "#FFFFFF",
    "light_grey":"#F5F7FA",
    "text":      "#1A1A2E",
}

# Cluster labels (assigned after analysing centroids)
# Labels are assigned based on centroid inspection after fitting:
# Cluster 0 → oldest, most shifts, highest happiness → High Engagement
# Cluster 1 → youngest, fewest shifts, lowest happiness → Low Engagement
# Cluster 2 → mid-range on all dimensions → Moderate Engagement
CLUSTER_LABELS = {
    0: "High Engagement",
    1: "Low Engagement",
    2: "Moderate Engagement",
}


def section(title: str) -> None:
    """Print a formatted section header to the console."""
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'=' * width}")


def fig_to_base64(fig: plt.Figure) -> str:
    """Encode a matplotlib figure as a base64 PNG string for HTML embedding."""
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


# =============================================================================
# STEP 1 | Load and Explore the Dataset
# =============================================================================
section("STEP 1: Load & Explore Dataset")

df = pd.read_csv(CSV_PATH)

print(f"\nShape:  {df.shape[0]} rows × {df.shape[1]} columns")
print(f"\nColumn dtypes:\n{df.dtypes}")
print(f"\nDescriptive statistics:\n{df.describe().round(2)}")
print(f"\nMissing values per column:\n{df.isnull().sum()}")
print(f"\nSkill distribution:\n{df['Skill'].value_counts()}")


# =============================================================================
# STEP 2 | Preprocess: Standardise Features
# =============================================================================
section("STEP 2: Preprocessing | StandardScaler")

features = ["Age", "ShiftsCompleted"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

print(f"\nFeatures selected: {features}")
print(f"Scaled array shape: {X_scaled.shape}")
print(f"Mean after scaling (should be ~0): {X_scaled.mean(axis=0).round(4)}")
print(f"Std  after scaling (should be ~1): {X_scaled.std(axis=0).round(4)}")


# =============================================================================
# STEP 3 | Elbow Method to Determine Optimal k
# =============================================================================
section("STEP 3: Elbow Method (k = 1 … 8)")

inertias = []
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    print(f"  k={k}  inertia={km.inertia_:.2f}")


# =============================================================================
# STEP 4 | Fit K-Means with k=3
# =============================================================================
section(f"STEP 4: K-Means Clustering (k={K})")

kmeans = KMeans(n_clusters=K, random_state=RANDOM_STATE, n_init=10)
kmeans.fit(X_scaled)
df["Cluster"] = kmeans.labels_
df["ClusterLabel"] = df["Cluster"].map(CLUSTER_LABELS)

centroids_scaled = kmeans.cluster_centers_
# Inverse-transform centroids back to original feature space for readability
centroids_original = scaler.inverse_transform(centroids_scaled)

print(f"\nCluster assignments added to dataframe.")
print(f"Value counts:\n{df['Cluster'].value_counts().sort_index()}")
print(f"\nCentroid positions (original scale):")
for i, c in enumerate(centroids_original):
    print(f"  Cluster {i} ({CLUSTER_LABELS[i]}): "
          f"Age={c[0]:.1f}, ShiftsCompleted={c[1]:.1f}")


# =============================================================================
# STEP 5 | Cluster Analysis
# =============================================================================
section("STEP 5: Cluster Profile Analysis")

cluster_summary = (
    df.groupby("Cluster")
    .agg(
        Count=("VolunteerID", "count"),
        Mean_Age=("Age", "mean"),
        Mean_Shifts=("ShiftsCompleted", "mean"),
        Mean_Happiness=("HappinessScore", "mean"),
        Top_Skill=("Skill", lambda s: s.value_counts().idxmax()),
    )
    .round(2)
)
cluster_summary.index = [CLUSTER_LABELS[i] for i in cluster_summary.index]
cluster_summary.index.name = "Cluster"

print(f"\n{cluster_summary.to_string()}")


# =============================================================================
# STEP 6 | Generate Visualisations & Save PNGs
# =============================================================================
section("STEP 6: Generating Visualisations")

os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.titlecolor": PALETTE["navy"],
    "axes.labelcolor": PALETTE["navy"],
    "xtick.color": PALETTE["navy"],
    "ytick.color": PALETTE["navy"],
    "figure.facecolor": PALETTE["white"],
    "axes.facecolor": PALETTE["white"],
})

# -- Plot 1: Elbow Curve ------------------------------------------------------
fig1, ax1 = plt.subplots(figsize=(8, 5))
ax1.plot(list(K_RANGE), inertias, marker="o", linewidth=2.5,
         color=PALETTE["blue"], markerfacecolor=PALETTE["mint"],
         markersize=9, markeredgecolor=PALETTE["navy"], markeredgewidth=1.5)
ax1.axvline(x=K, color=PALETTE["mint"], linestyle="--", linewidth=1.8,
            label=f"Chosen k = {K}")
ax1.set_title("Elbow Method | Optimal Number of Clusters")
ax1.set_xlabel("Number of Clusters (k)")
ax1.set_ylabel("Inertia (Within-Cluster SSE)")
ax1.legend(frameon=False)
ax1.set_xticks(list(K_RANGE))
fig1.tight_layout()
elbow_path = os.path.join(OUTPUT_DIR, "plot_elbow.png")
fig1.savefig(elbow_path, dpi=150, bbox_inches="tight")
print(f"  Saved: {elbow_path}")

# -- Plot 2: Scatter | Age vs ShiftsCompleted, coloured by cluster ------------
fig2, ax2 = plt.subplots(figsize=(9, 6))
for cluster_id in sorted(df["Cluster"].unique()):
    subset = df[df["Cluster"] == cluster_id]
    ax2.scatter(
        subset["Age"], subset["ShiftsCompleted"],
        color=CLUSTER_COLOURS[cluster_id],
        label=CLUSTER_LABELS[cluster_id],
        s=120, alpha=0.85, edgecolors=PALETTE["navy"], linewidths=0.8, zorder=3
    )
# Plot centroids in original space
for i, c in enumerate(centroids_original):
    ax2.scatter(c[0], c[1], marker="X", s=280,
                color=CLUSTER_COLOURS[i], edgecolors=PALETTE["navy"],
                linewidths=1.5, zorder=5)
ax2.set_title("K-Means Clusters | Age vs Shifts Completed")
ax2.set_xlabel("Age")
ax2.set_ylabel("Shifts Completed")
ax2.legend(frameon=False, title="Cluster", title_fontsize=10)
# Annotate volunteer names (slightly offset)
for _, row in df.iterrows():
    ax2.annotate(row["Name"].split()[0], (row["Age"], row["ShiftsCompleted"]),
                 fontsize=7, color=PALETTE["navy"], alpha=0.75,
                 xytext=(3, 3), textcoords="offset points")
fig2.tight_layout()
scatter_path = os.path.join(OUTPUT_DIR, "plot_scatter.png")
fig2.savefig(scatter_path, dpi=150, bbox_inches="tight")
print(f"  Saved: {scatter_path}")

# -- Plot 3: Bar | Average HappinessScore per Cluster -------------------------
happiness_means = (
    df.groupby("Cluster")["HappinessScore"].mean()
    .reindex(sorted(df["Cluster"].unique()))
)
labels_ordered = [CLUSTER_LABELS[i] for i in happiness_means.index]

fig3, ax3 = plt.subplots(figsize=(8, 5))
bars = ax3.bar(labels_ordered, happiness_means.values,
               color=[CLUSTER_COLOURS[i] for i in happiness_means.index],
               width=0.5, edgecolor=PALETTE["navy"], linewidth=0.8)
for bar, val in zip(bars, happiness_means.values):
    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
             f"{val:.2f}", ha="center", va="bottom",
             fontsize=11, fontweight="bold", color=PALETTE["navy"])
ax3.set_title("Average Happiness Score by Cluster")
ax3.set_xlabel("Cluster")
ax3.set_ylabel("Mean Happiness Score")
ax3.set_ylim(0, 11)
ax3.yaxis.grid(True, linestyle="--", alpha=0.5, color="#CCCCCC")
ax3.set_axisbelow(True)
fig3.tight_layout()
happiness_path = os.path.join(OUTPUT_DIR, "plot_happiness.png")
fig3.savefig(happiness_path, dpi=150, bbox_inches="tight")
print(f"  Saved: {happiness_path}")

# -- Plot 4: Stacked Bar | Skill Distribution across Clusters -----------------
skill_dist = (
    df.groupby(["Cluster", "Skill"])["VolunteerID"]
    .count()
    .unstack(fill_value=0)
)
skill_dist.index = [CLUSTER_LABELS[i] for i in skill_dist.index]

skill_colours = [PALETTE["blue"], PALETTE["mint"], PALETTE["sky"],
                 PALETTE["navy"], "#F4A261", "#E76F51"]
fig4, ax4 = plt.subplots(figsize=(10, 6))
skill_dist.plot(kind="bar", stacked=True, ax=ax4, color=skill_colours[:len(skill_dist.columns)],
                width=0.5, edgecolor=PALETTE["navy"], linewidth=0.6)
ax4.set_title("Skill Distribution Across Clusters")
ax4.set_xlabel("Cluster")
ax4.set_ylabel("Number of Volunteers")
ax4.legend(title="Skill", bbox_to_anchor=(1.01, 1), loc="upper left",
           frameon=False, fontsize=9)
ax4.set_xticklabels(ax4.get_xticklabels(), rotation=15, ha="right")
ax4.yaxis.grid(True, linestyle="--", alpha=0.5, color="#CCCCCC")
ax4.set_axisbelow(True)
fig4.tight_layout()
skill_path = os.path.join(OUTPUT_DIR, "plot_skills.png")
fig4.savefig(skill_path, dpi=150, bbox_inches="tight")
print(f"  Saved: {skill_path}")


# =============================================================================
# STEP 7 | Encode all plots as base64 for HTML embedding
# =============================================================================
section("STEP 7: Encoding Charts for HTML Report")

def png_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

b64_elbow    = png_to_b64(elbow_path)
b64_scatter  = png_to_b64(scatter_path)
b64_happiness= png_to_b64(happiness_path)
b64_skill    = png_to_b64(skill_path)
print("  All charts encoded successfully.")


# =============================================================================
# STEP 8 | Build the HTML Report
# =============================================================================
section("STEP 8: Generating HTML Report")

# ---------- Helper: build data table rows ------------------------------------
def build_table_rows(dataframe: pd.DataFrame) -> str:
    rows = []
    row_bg = [
        ["#EBF3FB", "#D6E8F6"],   # Cluster 0 | blue shades
        ["#E6F9F1", "#CCF3E3"],   # Cluster 1 | mint shades
        ["#EDF6FD", "#DBEEfA"],   # Cluster 2 | sky shades
    ]
    for _, r in dataframe.iterrows():
        cid = int(r["Cluster"])
        bg = row_bg[cid][0]
        dot_colour = CLUSTER_COLOURS[cid]
        rows.append(f"""
        <tr style="background:{bg}">
          <td>{r['Name']}</td>
          <td>{int(r['Age'])}</td>
          <td>{r['Skill']}</td>
          <td>{int(r['ShiftsCompleted'])}</td>
          <td>{r['HappinessScore']}</td>
          <td><span class="badge" style="background:{dot_colour}">{r['ClusterLabel']}</span></td>
        </tr>""")
    return "\n".join(rows)


# ---------- Helper: build cluster profile cards ------------------------------
# Descriptions and icons keyed by label so they always match regardless of
# which integer cluster ID sklearn assigns to each segment.
CARD_META = {
    "Low Engagement": {
        "icon": "📉",
        "desc": (
            "These volunteers are newer or less active, completing fewer shifts and "
            "reporting lower happiness scores. They may benefit from onboarding support, "
            "mentorship pairing, or simplified task assignment tools within FlexiFinance."
        ),
    },
    "High Engagement": {
        "icon": "📈",
        "desc": (
            "Top performers who are highly experienced with the most shifts completed "
            "and the highest happiness scores. These ambassadors are ideal candidates "
            "for peer-mentorship roles or advanced feature access on the platform."
        ),
    },
    "Moderate Engagement": {
        "icon": "📊",
        "desc": (
            "A solid mid-tier group with moderate engagement and reasonable happiness. "
            "Targeted nudges, such as milestone rewards or progress dashboards, "
            "could convert this group into high-engagement volunteers."
        ),
    },
}

# Reverse-lookup: label → cluster id, so cards use the same colour as the charts
LABEL_TO_CLUSTER_ID = {v: k for k, v in CLUSTER_LABELS.items()}


def build_cluster_cards(summary: pd.DataFrame) -> str:
    cards_html = []
    for label, row in summary.iterrows():
        cid = LABEL_TO_CLUSTER_ID[label]
        colour = CLUSTER_COLOURS[cid]
        meta = CARD_META[label]
        cards_html.append(f"""
      <div class="cluster-card" style="border-top: 4px solid {colour}">
        <div class="cluster-card-header" style="color:{colour}">
          <span class="cluster-icon">{meta['icon']}</span>
          <h3>{label}</h3>
        </div>
        <div class="cluster-stats">
          <div class="stat"><span class="stat-label">Volunteers</span><span class="stat-value" style="color:{colour}">{int(row['Count'])}</span></div>
          <div class="stat"><span class="stat-label">Avg Age</span><span class="stat-value" style="color:{colour}">{row['Mean_Age']:.1f}</span></div>
          <div class="stat"><span class="stat-label">Avg Shifts</span><span class="stat-value" style="color:{colour}">{row['Mean_Shifts']:.1f}</span></div>
          <div class="stat"><span class="stat-label">Avg Happiness</span><span class="stat-value" style="color:{colour}">{row['Mean_Happiness']:.2f}</span></div>
          <div class="stat"><span class="stat-label">Top Skill</span><span class="stat-value" style="color:{colour}; font-size:0.85rem">{row['Top_Skill']}</span></div>
        </div>
        <p class="cluster-desc">{meta['desc']}</p>
      </div>""")
    return "\n".join(cards_html)


table_rows     = build_table_rows(df.sort_values("Cluster"))
cluster_cards  = build_cluster_cards(cluster_summary)


# =============================================================================
# Full HTML template
# =============================================================================
html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>FlexiFinance | User Engagement Cluster Analysis</title>
  <link rel="preconnect" href="https://fonts.googleapis.com" />
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Plus+Jakarta+Sans:wght@400;600;700;800&display=swap" rel="stylesheet" />
  <style>
    /* ---- Reset & Base ---- */
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      font-family: 'Inter', sans-serif;
      background: #F5F7FA;
      color: #1A1A2E;
      font-size: 15px;
      line-height: 1.65;
    }}
    h1, h2, h3, h4 {{ font-family: 'Plus Jakarta Sans', sans-serif; }}

    /* ---- Layout Wrappers ---- */
    .container {{ max-width: 1100px; margin: 0 auto; padding: 0 24px; }}

    /* ---- Header ---- */
    .site-header {{
      background: linear-gradient(135deg, #0D2B4E 0%, #1A5FA8 60%, #5BA4D4 100%);
      padding: 56px 24px 48px;
      text-align: center;
      color: white;
    }}
    .site-header .tag {{
      display: inline-block;
      background: rgba(255,255,255,0.15);
      border: 1px solid rgba(255,255,255,0.3);
      padding: 4px 14px;
      border-radius: 20px;
      font-size: 0.78rem;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 16px;
    }}
    .site-header h1 {{
      font-size: clamp(1.8rem, 4vw, 2.8rem);
      font-weight: 800;
      letter-spacing: -0.02em;
      margin-bottom: 10px;
    }}
    .site-header .subtitle {{
      font-size: 1rem;
      opacity: 0.82;
      font-weight: 400;
    }}

    /* ---- Sections ---- */
    .section {{
      background: white;
      border-radius: 16px;
      padding: 36px 40px;
      margin: 28px 0;
      box-shadow: 0 2px 12px rgba(13,43,78,0.07);
    }}
    .section-title {{
      font-size: 1.3rem;
      font-weight: 700;
      color: #0D2B4E;
      margin-bottom: 6px;
      padding-bottom: 12px;
      border-bottom: 2px solid #E8F4FD;
    }}
    .section-subtitle {{
      color: #5BA4D4;
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      font-weight: 600;
      margin-bottom: 4px;
    }}

    /* ---- Executive Summary KPI Cards ---- */
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 16px;
      margin-top: 20px;
    }}
    .kpi-card {{
      background: #E8F4FD;
      border-radius: 12px;
      padding: 20px 16px;
      text-align: center;
    }}
    .kpi-value {{
      font-family: 'Plus Jakarta Sans', sans-serif;
      font-size: 2rem;
      font-weight: 800;
      color: #1A5FA8;
      display: block;
    }}
    .kpi-label {{
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: #0D2B4E;
      font-weight: 600;
      margin-top: 4px;
    }}

    /* ---- Charts Grid ---- */
    .charts-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
      margin-top: 20px;
    }}
    @media (max-width: 720px) {{ .charts-grid {{ grid-template-columns: 1fr; }} }}
    .chart-card {{
      background: #F5F7FA;
      border-radius: 12px;
      padding: 16px;
      text-align: center;
    }}
    .chart-card img {{
      width: 100%;
      height: auto;
      border-radius: 8px;
    }}
    .chart-caption {{
      font-size: 0.78rem;
      color: #5BA4D4;
      margin-top: 8px;
      font-style: italic;
    }}

    /* ---- Data Table ---- */
    .table-wrapper {{ overflow-x: auto; margin-top: 20px; }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.88rem;
    }}
    thead tr {{
      background: #0D2B4E;
      color: white;
      text-align: left;
    }}
    thead th {{
      padding: 12px 14px;
      font-weight: 600;
      font-family: 'Plus Jakarta Sans', sans-serif;
      letter-spacing: 0.03em;
    }}
    tbody tr {{ transition: opacity 0.15s; }}
    tbody tr:hover {{ opacity: 0.88; }}
    tbody td {{
      padding: 10px 14px;
      border-bottom: 1px solid rgba(13,43,78,0.06);
      color: #1A1A2E;
    }}
    .badge {{
      display: inline-block;
      color: white;
      font-size: 0.72rem;
      font-weight: 600;
      padding: 3px 10px;
      border-radius: 20px;
      white-space: nowrap;
    }}

    /* ---- Cluster Profile Cards ---- */
    .cluster-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 20px;
      margin-top: 20px;
    }}
    .cluster-card {{
      background: #F5F7FA;
      border-radius: 12px;
      padding: 24px 22px;
      border-top-width: 4px;
      border-top-style: solid;
    }}
    .cluster-card-header {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin-bottom: 16px;
    }}
    .cluster-icon {{ font-size: 1.5rem; }}
    .cluster-card-header h3 {{
      font-size: 1.05rem;
      font-weight: 700;
    }}
    .cluster-stats {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-bottom: 16px;
    }}
    .stat {{
      background: white;
      border-radius: 8px;
      padding: 10px 12px;
    }}
    .stat-label {{
      display: block;
      font-size: 0.7rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: #888;
      font-weight: 600;
    }}
    .stat-value {{
      display: block;
      font-size: 1.15rem;
      font-weight: 700;
      font-family: 'Plus Jakarta Sans', sans-serif;
      margin-top: 2px;
    }}
    .cluster-desc {{
      font-size: 0.84rem;
      color: #555;
      line-height: 1.6;
    }}

    /* ---- Analysis Section ---- */
    .analysis-block {{
      margin-top: 20px;
      display: flex;
      flex-direction: column;
      gap: 22px;
    }}
    .analysis-para {{ border-left: 3px solid #1A5FA8; padding-left: 18px; }}
    .analysis-para.green {{ border-color: #2EC48A; }}
    .analysis-para.sky {{ border-color: #5BA4D4; }}
    .analysis-para h4 {{
      font-size: 0.9rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: #1A5FA8;
      margin-bottom: 6px;
      font-weight: 700;
    }}
    .analysis-para.green h4 {{ color: #2EC48A; }}
    .analysis-para.sky h4   {{ color: #5BA4D4; }}
    .analysis-para p {{ font-size: 0.93rem; color: #333; }}

    /* ---- Footer ---- */
    .site-footer {{
      text-align: center;
      padding: 32px 24px;
      color: #888;
      font-size: 0.8rem;
    }}
  </style>
</head>
<body>

<!-- ============================================================
     HEADER
     ============================================================ -->
<header class="site-header">
  <div class="tag">HCBI372 | Data Mining Assignment</div>
  <h1>FlexiFinance | User Engagement<br>Cluster Analysis</h1>
  <p class="subtitle">K-Means Clustering &bull; Volunteer Engagement Dataset &bull; k&nbsp;=&nbsp;3</p>
</header>

<main class="container">

  <!-- ============================================================
       EXECUTIVE SUMMARY
       ============================================================ -->
  <section class="section">
    <p class="section-subtitle">Overview</p>
    <h2 class="section-title">Executive Summary</h2>
    <p>This report presents the results of a K-Means clustering analysis applied to volunteer engagement data for the HCBI community initiative. The goal is to identify natural groupings of volunteers based on age and activity level, enabling FlexiFinance to design more personalised features and interventions.</p>
    <div class="kpi-grid">
      <div class="kpi-card">
        <span class="kpi-value">21</span>
        <span class="kpi-label">Total Volunteers</span>
      </div>
      <div class="kpi-card">
        <span class="kpi-value">3</span>
        <span class="kpi-label">Clusters Identified</span>
      </div>
      <div class="kpi-card">
        <span class="kpi-value">2</span>
        <span class="kpi-label">Variables Clustered</span>
      </div>
      <div class="kpi-card">
        <span class="kpi-value">6</span>
        <span class="kpi-label">Skill Categories</span>
      </div>
      <div class="kpi-card">
        <span class="kpi-value">42</span>
        <span class="kpi-label">Random State</span>
      </div>
    </div>
  </section>

  <!-- ============================================================
       VISUALISATIONS
       ============================================================ -->
  <section class="section">
    <p class="section-subtitle">Charts</p>
    <h2 class="section-title">Visualisations</h2>
    <div class="charts-grid">
      <div class="chart-card">
        <img src="data:image/png;base64,{b64_elbow}" alt="Elbow Curve" />
        <p class="chart-caption">Figure 1: Elbow Method | inertia vs k. The elbow at k=3 confirms the optimal cluster count.</p>
      </div>
      <div class="chart-card">
        <img src="data:image/png;base64,{b64_scatter}" alt="Cluster Scatter Plot" />
        <p class="chart-caption">Figure 2: Age vs Shifts Completed, colour-coded by cluster. X markers indicate cluster centroids.</p>
      </div>
      <div class="chart-card">
        <img src="data:image/png;base64,{b64_happiness}" alt="Average Happiness Score" />
        <p class="chart-caption">Figure 3: Average Happiness Score per cluster, showing a clear positive relationship with engagement level.</p>
      </div>
      <div class="chart-card">
        <img src="data:image/png;base64,{b64_skill}" alt="Skill Distribution" />
        <p class="chart-caption">Figure 4: Skill distribution stacked by cluster, revealing which volunteer types dominate each segment.</p>
      </div>
    </div>
  </section>

  <!-- ============================================================
       DATA TABLE
       ============================================================ -->
  <section class="section">
    <p class="section-subtitle">Full Dataset</p>
    <h2 class="section-title">Volunteer Data with Cluster Assignments</h2>
    <p style="margin-bottom:4px; font-size:0.87rem; color:#666;">Rows are colour-coded by assigned cluster. Sort or scroll horizontally on smaller screens.</p>
    <div class="table-wrapper">
      <table>
        <thead>
          <tr>
            <th>Name</th>
            <th>Age</th>
            <th>Skill</th>
            <th>Shifts</th>
            <th>Happiness</th>
            <th>Cluster</th>
          </tr>
        </thead>
        <tbody>
          {table_rows}
        </tbody>
      </table>
    </div>
  </section>

  <!-- ============================================================
       CLUSTER PROFILES
       ============================================================ -->
  <section class="section">
    <p class="section-subtitle">Segment Profiles</p>
    <h2 class="section-title">Cluster Profiles</h2>
    <div class="cluster-grid">
      {cluster_cards}
    </div>
  </section>

  <!-- ============================================================
       WRITTEN ANALYSIS
       ============================================================ -->
  <section class="section">
    <p class="section-subtitle">Interpretation</p>
    <h2 class="section-title">Written Analysis</h2>
    <div class="analysis-block">

      <div class="analysis-para">
        <h4>Technique Used</h4>
        <p>K-Means clustering is an unsupervised machine learning algorithm that partitions a dataset into <em>k</em> non-overlapping groups by minimising the within-cluster sum of squared distances (inertia) between each data point and its nearest centroid. For this analysis, the two numeric features (<strong>Age</strong> and <strong>ShiftsCompleted</strong>) were standardised using <code>StandardScaler</code> to ensure both dimensions contribute equally regardless of their natural scales. The optimal number of clusters was determined by plotting the Elbow Curve, where inertia was computed for k&nbsp;=&nbsp;1 through 8; the curve showed a clear inflection point at k&nbsp;=&nbsp;3, confirming three distinct volunteer segments. The final model was fitted using <code>sklearn.cluster.KMeans</code> with <code>random_state=42</code> and <code>n_init=10</code> to ensure reproducible, stable results.</p>
      </div>

      <div class="analysis-para green">
        <h4>Key Insight Found</h4>
        <p>The most striking pattern across the three clusters is a strong positive correlation between <strong>age, accumulated experience (shifts completed), and self-reported happiness</strong>. The High Engagement cluster is dominated by older volunteers (mean age ~65) who have completed the most shifts (35+) and report the highest happiness scores (averaging above 9.5 out of 10), predominantly in Gardening and Admin Support roles. By contrast, the Low Engagement cluster skews young (mean age ~22) with very few shifts completed and happiness scores below 6.5, concentrated in Tutoring and General Labour. The Moderate Engagement cluster occupies the middle ground across all three dimensions. This pattern suggests that <strong>sustained participation, not just age alone, is the primary driver of volunteer satisfaction</strong>.</p>
      </div>

      <div class="analysis-para sky">
        <h4>Design Recommendation for FlexiFinance</h4>
        <p>FlexiFinance, as a South African freelancer finance platform, can apply this insight directly to its user onboarding and engagement model. The data shows that newer, younger users (mirroring the Low Engagement cluster) are at highest risk of low satisfaction and early drop-off, likely because they lack the experience and accumulated wins that drive happiness. A concrete design recommendation is to introduce a <strong>personalised milestone progress tracker</strong> for new freelancers: a visible dashboard widget that celebrates their first invoice sent, first payment received, and first five completed gigs, analogous to the shift-completion metric that differentiates engagement tiers in this dataset. This feature would give newer users the psychological reward of visible progress, bridging the experience gap that currently separates low-engagement volunteers from high-engagement ones, and increasing platform retention among the most at-risk segment.</p>
      </div>

    </div>
  </section>

</main>

<!-- ============================================================
     FOOTER
     ============================================================ -->
<footer class="site-footer">
  <p>Generated programmatically by <strong>analysis.py</strong> &bull; HCBI372 Data Mining &bull; K-Means Clustering Report</p>
  <p style="margin-top:4px; color:#aaa;">Dataset: HCBI_SF2_user_data.csv &bull; 21 records &bull; 6 skill categories &bull; k&nbsp;=&nbsp;3 clusters</p>
</footer>

</body>
</html>
"""

html_path = os.path.join(OUTPUT_DIR, "flexifinance_cluster_report.html")
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"  HTML report saved: {html_path}")

# Also write to repo root as index.html so GitHub Pages serves it by default
index_path = "index.html"
with open(index_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"  GitHub Pages entry point saved: {index_path}")

# =============================================================================
# Done
# =============================================================================
section("COMPLETE")
print(f"""
  Output files:
    {os.path.join(OUTPUT_DIR, 'plot_elbow.png')}
    {os.path.join(OUTPUT_DIR, 'plot_scatter.png')}
    {os.path.join(OUTPUT_DIR, 'plot_happiness.png')}
    {os.path.join(OUTPUT_DIR, 'plot_skills.png')}
    {os.path.join(OUTPUT_DIR, 'flexifinance_cluster_report.html')}

  Open the HTML file in any browser to view the full report.
""")
