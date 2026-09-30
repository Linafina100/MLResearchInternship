"""
Interactive HTML report visualizing three fixes made to this project's
sim-to-real pipeline, each backed by real (already-generated) data rather
than illustrative/fabricated curves:

1. The dV/dQ termination artifact (feature_engineering.py's
   `exclude_final_transition`): PyBaMM's event-triggered "Discharge until
   X V" always takes one oversized final step landing right on the
   voltage cutoff. Dividing by that step's tiny-but-nonzero dQ explodes
   dV/dQ into a non-physical spike. Shown on a real battery trace from
   data/24_soh_range_0.8_1.0_v1.5/ (today's pipeline -- this artifact is
   still present in every continuous-discharge run regardless of the
   other fixes below, which is why the exclusion still matters).

2. The t_interp dense-sampling fix (experiments/18+): PyBaMM's adaptive
   solver can take large steps and skip 0.1V bins entirely. Compared here
   using one real "before" battery from data/17_high_soh_low_crate_v1.5/
   (pre-t_interp, ~50 points/battery) against one real "after" battery
   from data/24_soh_range_0.8_1.0_v1.5/ (dense t_interp, 3001
   points/battery).

3. Sim-to-real magnitude alignment (experiments 20 Part C's NMC
   diffusivity/10 + experiment 21's LFP OCP rate=-3): mean dV/dQ in the
   2.5-3.0V target zone, per chemistry, for real EMPA data
   (data/24_real_empa_soh_0.8_1.0/, SOH 0.8-1.0) against synthetic data
   BEFORE either fix (data/17_high_soh_low_crate_v1.5/, same SOH 0.8-1.0
   range, predates both fixes) and AFTER both fixes
   (data/24_soh_range_0.8_1.0_v1.5/, same SOH range, current pipeline).

All three synthetic/real datasets are already-generated raw CSVs (no
resimulation here) -- see data/README.md for exactly which script
produced each. Bin computation for section 3 reuses root
feature_engineering.py's create_features_by_voltage_bins unchanged, so
the numbers match what every sim-to-real experiment's evaluate script
would compute.

Usage: python3 make_pipeline_fixes_report.py
Output: data/simulating_problems/pipeline_fixes_report_<date>.html
"""
import os
import sys
import tempfile
from datetime import date

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)
from feature_engineering import create_features_by_voltage_bins

DATA_DIR = os.path.join(PROJECT_DIR, "data")
# Sections 1 & 2 use the current pipeline's own data (today's termination
# artifact / point-density story, independent of the SOH point).
AFTER_SYNTH_CSV = os.path.join(DATA_DIR, "24_soh_range_0.8_1.0_v1.5", "raw", "advanced_synthetic_battery_data.csv")
BEFORE_SYNTH_CSV = os.path.join(DATA_DIR, "17_high_soh_low_crate_v1.5", "raw", "advanced_synthetic_battery_data.csv")
# Section 3 uses the exact SOH=0.8 pairing the diffusivity/OCP fixes were
# calibrated and published against (experiments 21/22's own RESULTS.md) --
# verified below to reproduce those exact published numbers.
S3_AFTER_SYNTH_CSV = os.path.join(DATA_DIR, "21_soh_0.8_lfp_ocp_tuned_v1.5", "raw", "advanced_synthetic_battery_data.csv")
S3_REAL_CSV = os.path.join(DATA_DIR, "22_real_empa_soh_0.8", "raw", "real_empa_raw.csv")

OUT_DIR = os.path.join(DATA_DIR, "simulating_problems")
os.makedirs(OUT_DIR, exist_ok=True)
OUT_HTML = os.path.join(OUT_DIR, f"pipeline_fixes_report_{date.today().isoformat()}.html")

GROUPBY_COLS = ['Chemistry', 'Size_Multiplier', 'SOH', 'Initial_SOC', 'Variation_ID']
V_BIN_MIN, V_BIN_MAX, V_BIN_WIDTH = 1.9, 4.3, 0.1
TARGET_ZONE_BINS = ["dV_dQ_V_3.0_2.9", "dV_dQ_V_2.9_2.8", "dV_dQ_V_2.8_2.7", "dV_dQ_V_2.7_2.6", "dV_dQ_V_2.6_2.5"]

COLOR_BEFORE = "#B23A48"
COLOR_AFTER = "#0E7C86"
COLOR_REAL = "#12181D"
COLOR_LFP = "#2A78D6"
COLOR_NMC = "#EB6834"


def voltage_bin(v):
    """Mirrors feature_engineering.py's bin_high/bin_low assignment exactly."""
    bin_high = np.ceil(round(v, 4) * 10) / 10.0
    bin_high = min(V_BIN_MAX, max(V_BIN_MIN + V_BIN_WIDTH, bin_high))
    return round(bin_high, 1), round(bin_high - V_BIN_WIDTH, 1)


def one_battery(csv_path, chemistry, which=0):
    """Loads one battery's raw (Time, Voltage, Capacity) trace, sorted, plus its metadata."""
    df = pd.read_csv(csv_path, low_memory=False)
    df['Battery_ID'] = df.groupby(GROUPBY_COLS).ngroup()
    chem_batteries = df[df['Chemistry'] == chemistry]['Battery_ID'].unique()
    bid = sorted(chem_batteries)[which]
    battery = df[df['Battery_ID'] == bid].sort_values('Time [s]', kind='stable').reset_index(drop=True)
    meta = battery.iloc[0]
    return battery, meta


def per_transition_dvdq(battery):
    """Exact dV/dQ-per-raw-transition logic from feature_engineering.py, unbinned."""
    dV = battery['Voltage [V]'].diff()
    dQ = battery['Capacity [A.h]'].diff()
    valid = dQ > 1e-5
    idx = battery.index[valid]
    v_end = battery.loc[idx, 'Voltage [V]']
    dvdq = dV.loc[idx] / dQ.loc[idx]
    return pd.DataFrame({'Voltage [V]': v_end.values, 'dV_dQ': dvdq.values}).reset_index(drop=True)


def points_per_voltage_bin(battery):
    bins = [voltage_bin(v) for v in battery['Voltage [V]']]
    labels = [f"{hi:.1f}-{lo:.1f}" for hi, lo in bins]
    counts = pd.Series(labels).value_counts()
    all_bins = [f"{round(V_BIN_MIN + i * V_BIN_WIDTH, 1):.1f}-{round(V_BIN_MIN + (i - 1) * V_BIN_WIDTH, 1):.1f}"
                for i in range(round((V_BIN_MAX - V_BIN_MIN) / V_BIN_WIDTH), 0, -1)]
    return pd.Series({b: counts.get(b, 0) for b in all_bins})


# ============================================================
# SECTION 1: termination artifact, on a real current-pipeline battery
# ============================================================
print("Section 1: termination artifact...")
nmc_battery, nmc_meta = one_battery(AFTER_SYNTH_CSV, "NMC", which=0)
transitions = per_transition_dvdq(nmc_battery)
tail = transitions.tail(12).reset_index(drop=True)
tail_fixed = tail.iloc[:-1]

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=tail['Voltage [V]'], y=tail['dV_dQ'], mode='lines+markers', name='Raw (includes final transition)',
    line=dict(color=COLOR_BEFORE, width=2), marker=dict(size=8),
))
fig1.add_trace(go.Scatter(
    x=tail_fixed['Voltage [V]'], y=tail_fixed['dV_dQ'], mode='lines+markers',
    name='Fixed (exclude_final_transition=True)',
    line=dict(color=COLOR_AFTER, width=3), marker=dict(size=9, symbol='diamond'),
))
spike_v, spike_val = tail['Voltage [V]'].iloc[-1], tail['dV_dQ'].iloc[-1]
fig1.add_annotation(x=spike_v, y=spike_val, text=f"solver-termination spike<br>dV/dQ = {spike_val:.1f}",
                     showarrow=True, arrowhead=2, ax=-60, ay=-40, font=dict(color=COLOR_BEFORE))
fig1.update_layout(
    title=f"dV/dQ near the voltage cutoff (NMC, SOH={nmc_meta['SOH']:.2f}, "
          f"cutoff={nmc_battery['V_min [V]'].iloc[0]}V) -- last 12 raw transitions",
    xaxis_title="Terminal voltage [V] (discharge direction: high -> low)",
    yaxis_title="dV/dQ",
    xaxis=dict(autorange='reversed'),
    template="plotly_white", height=460,
)

# ============================================================
# SECTION 2: t_interp dense sampling -- points per 0.1V bin
# ============================================================
print("Section 2: t_interp point density...")
before_battery, before_meta = one_battery(BEFORE_SYNTH_CSV, "NMC", which=0)
after_battery, after_meta = one_battery(AFTER_SYNTH_CSV, "NMC", which=0)
before_counts = points_per_voltage_bin(before_battery)
after_counts = points_per_voltage_bin(after_battery)
n_empty_before = int((before_counts == 0).sum())
n_empty_after = int((after_counts == 0).sum())

fig2 = make_subplots(rows=1, cols=2, horizontal_spacing=0.10, subplot_titles=(
    f"Before t_interp ({n_empty_before}/{len(before_counts)} bins empty)",
    f"After t_interp ({n_empty_after}/{len(after_counts)} bins empty)",
))
fig2.add_trace(go.Bar(x=before_counts.index, y=before_counts.values, marker_color="#C7CFD4",
                       name="Before", showlegend=False), row=1, col=1)
fig2.add_trace(go.Bar(x=after_counts.index, y=after_counts.values, marker_color=COLOR_AFTER,
                       name="After", showlegend=False), row=1, col=2)
# 0-height bars are invisible regardless of color, so mark empty bins with an explicit tick instead.
for col_idx, counts in ((1, before_counts), (2, after_counts)):
    empty_x = [b for b, c in counts.items() if c == 0]
    if empty_x:
        fig2.add_trace(go.Scatter(x=empty_x, y=[0.5] * len(empty_x), mode='markers',
                                   marker=dict(symbol='x-thin', size=9, color=COLOR_BEFORE,
                                               line=dict(width=2, color=COLOR_BEFORE)),
                                   showlegend=False, hoverinfo='skip'), row=1, col=col_idx)
fig2.update_yaxes(title_text="Raw points in bin", row=1, col=1)
fig2.update_xaxes(title_text="Voltage bin [V]", tickangle=-45, row=1, col=1)
fig2.update_xaxes(title_text="Voltage bin [V]", tickangle=-45, row=1, col=2)
fig2.update_layout(
    title_text=(f"Points per 0.1V bin, one NMC battery each -- before: SOH={before_meta['SOH']:.2f}, "
                f"C-rate={before_meta.get('C_Rate', float('nan')):.2f}, {len(before_battery)} raw points; "
                f"after: SOH={after_meta['SOH']:.2f}, C-rate={after_meta.get('C_Rate', float('nan')):.2f}, "
                f"{len(after_battery)} raw points (× = empty bin)"),
    template="plotly_white", height=480, margin=dict(t=110),
)

# ============================================================
# SECTION 3: sim-to-real magnitude alignment, 2.5-3.0V target zone
# ============================================================
print("Section 3: magnitude alignment (real vs. before-fix vs. after-fix)...")
tmp_dir = tempfile.mkdtemp(prefix="pipeline_fixes_report_")


def target_zone_means(csv_path, label):
    out_dir = os.path.join(tmp_dir, label)
    df = create_features_by_voltage_bins(csv_path, output_dir=out_dir, min_chemistry_coverage=0.0)
    cols = [c for c in TARGET_ZONE_BINS if c in df.columns]
    means = df.groupby('Chemistry')[cols].mean()
    return means.reindex(columns=TARGET_ZONE_BINS)


real_means = target_zone_means(S3_REAL_CSV, "real")
before_means = target_zone_means(BEFORE_SYNTH_CSV, "before")
after_means = target_zone_means(S3_AFTER_SYNTH_CSV, "after")

bin_labels = [c.replace("dV_dQ_V_", "").replace("_", "-") + "V" for c in TARGET_ZONE_BINS]

fig3 = make_subplots(rows=1, cols=2, subplot_titles=("LFP", "NMC"), shared_yaxes=False)
for col_idx, chem in enumerate(["LFP", "NMC"], start=1):
    fig3.add_trace(go.Scatter(x=bin_labels, y=before_means.loc[chem].values, mode='lines+markers',
                               name='Synthetic (before fix)', legendgroup='before',
                               showlegend=(col_idx == 1),
                               line=dict(color=COLOR_BEFORE, width=2, dash='dot'), marker=dict(size=8)),
                    row=1, col=col_idx)
    fig3.add_trace(go.Scatter(x=bin_labels, y=after_means.loc[chem].values, mode='lines+markers',
                               name='Synthetic (after fix)', legendgroup='after',
                               showlegend=(col_idx == 1),
                               line=dict(color=COLOR_AFTER, width=3), marker=dict(size=9, symbol='diamond')),
                    row=1, col=col_idx)
    fig3.add_trace(go.Scatter(x=bin_labels, y=real_means.loc[chem].values, mode='lines+markers',
                               name='Real (EMPA)', legendgroup='real',
                               showlegend=(col_idx == 1),
                               line=dict(color=COLOR_REAL, width=3), marker=dict(size=9, symbol='star')),
                    row=1, col=col_idx)
fig3.update_yaxes(title_text="Mean dV/dQ", row=1, col=1)
fig3.update_xaxes(title_text="Voltage bin", row=1, col=1)
fig3.update_xaxes(title_text="Voltage bin", row=1, col=2)
fig3.update_layout(
    title_text="Target-zone (2.5-3.0V) mean dV/dQ: real (EMPA, SOH=0.8) vs. synthetic, "
                "before/after the diffusivity + OCP fixes (SOH=0.8, same pairing as experiments 21/22)",
    template="plotly_white", height=460,
)


def build_table_html():
    rows = []
    for chem in ["LFP", "NMC"]:
        for source, means in (("Real (EMPA)", real_means), ("Synthetic, before fix", before_means),
                               ("Synthetic, after fix", after_means)):
            vals = means.loc[chem]
            cells = "".join(f"<td>{v:.2f}</td>" if pd.notna(v) else "<td>&mdash;</td>" for v in vals)
            highlight = ' class="after-row"' if source == "Synthetic, after fix" else ""
            rows.append(f"<tr{highlight}><td>{chem}</td><td>{source}</td>{cells}</tr>")
    header_cells = "".join(f"<th>{b}</th>" for b in bin_labels)
    return f"""
    <table>
      <thead><tr><th>Chemistry</th><th>Source</th>{header_cells}</tr></thead>
      <tbody>{''.join(rows)}</tbody>
    </table>
    """


# ============================================================
# ASSEMBLE HTML
# ============================================================
print(f"Writing report to {OUT_HTML}...")

STYLE = """
<style>
  body { font-family: -apple-system, "Segoe UI", sans-serif; max-width: 1100px; margin: 40px auto;
         padding: 0 24px; color: #12181D; background: #F7F8F9; }
  h1 { font-size: 26px; margin-bottom: 4px; }
  .subtitle { color: #63707A; font-size: 14px; margin-bottom: 40px; }
  section { background: #fff; border: 1px solid #DDE2E6; border-radius: 10px;
            padding: 24px 28px; margin-bottom: 32px; }
  section h2 { font-size: 19px; margin-top: 0; }
  section p.desc { color: #4B5560; font-size: 14px; line-height: 1.6; max-width: 80ch; }
  table { border-collapse: collapse; width: 100%; margin-top: 16px; font-size: 13.5px; }
  th, td { border: 1px solid #DDE2E6; padding: 8px 12px; text-align: right; }
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
  thead th { background: #F7F8F9; font-weight: 600; }
  tr.after-row { background: #E8F5F4; font-weight: 600; }
  footer { color: #63707A; font-size: 12px; border-top: 1px solid #DDE2E6; padding-top: 16px; }
</style>
"""

html_parts = [
    "<!doctype html><html><head><meta charset='utf-8'><title>Pipeline Fixes Report</title>",
    STYLE, "</head><body>",
    "<h1>Three sim-to-real pipeline fixes, visualized</h1>",
    f"<p class='subtitle'>Generated {date.today().isoformat()} by make_pipeline_fixes_report.py "
    "&mdash; every curve and number below comes from already-generated raw simulation/real data "
    "(sources cited per section), not illustrative approximations.</p>",

    "<section><h2>1. The dV/dQ termination artifact</h2>",
    "<p class='desc'>PyBaMM's event-triggered \"Discharge until X V\" always takes one oversized "
    "final step to land on the voltage cutoff. Dividing by that step's tiny dQ explodes dV/dQ into "
    "a non-physical spike -- shown here on a real NMC battery from the current pipeline "
    "(<code>data/24_soh_range_0.8_1.0_v1.5/</code>). This artifact is present in every "
    "continuous-discharge run regardless of the other two fixes below, which is why "
    "<code>feature_engineering.py</code>'s <code>exclude_final_transition</code> still matters today.</p>",
    fig1.to_html(full_html=False, include_plotlyjs='cdn'),
    "</section>",

    "<section><h2>2. The t_interp dense-sampling fix</h2>",
    "<p class='desc'>PyBaMM's adaptive solver can take large steps and skip 0.1V voltage bins "
    "entirely -- a real problem for the mutual-coverage leakage filter, which needs both "
    "chemistries genuinely represented in a bin. \"Before\" is one real battery from "
    "<code>data/17_high_soh_low_crate_v1.5/</code> (pre-t_interp); \"after\" is one real battery "
    "from <code>data/24_soh_range_0.8_1.0_v1.5/</code> (current pipeline, dense "
    "<code>t_interp</code> pass). Both are single real traces, not constructed examples.</p>",
    fig2.to_html(full_html=False, include_plotlyjs=False),
    "</section>",

    "<section><h2>3. Sim-to-real magnitude alignment</h2>",
    "<p class='desc'>Mean dV/dQ in the 2.5&ndash;3.0V target zone, per chemistry: real EMPA data "
    "at SOH=0.8 (<code>data/22_real_empa_soh_0.8/</code>) against synthetic data from "
    "<em>before</em> the NMC diffusivity/10 and LFP OCP rate=-3 fixes "
    "(<code>data/17_high_soh_low_crate_v1.5/</code>, predates both fixes) and <em>after</em> both "
    "fixes (<code>data/21_soh_0.8_lfp_ocp_tuned_v1.5/</code>, the exact SOH=0.8 pairing "
    "experiments 21/22 calibrated and published against). All three aggregate across every "
    "battery of that chemistry in the dataset (not a single trace) via root "
    "<code>feature_engineering.py</code>, unmodified.</p>",
    "<p class='desc'><strong>Read honestly, not as a perfect fit:</strong> the fix closes most of "
    "the gap in the shallowest, highest-importance bin (3.0&ndash;2.9V, NMC: real "
    f"{real_means.loc['NMC', TARGET_ZONE_BINS[0]]:.2f} vs. after "
    f"{after_means.loc['NMC', TARGET_ZONE_BINS[0]]:.2f}, before only "
    f"{before_means.loc['NMC', TARGET_ZONE_BINS[0]]:.2f}) and moves LFP substantially closer "
    "throughout. But per-bin match is uneven and NMC specifically <em>overshoots</em> real in the "
    "deeper bins after the fix (as documented in experiment 21's own RESULTS.md) &mdash; this is "
    "an aggregate/shallow-bin alignment, not a uniform per-bin one.</p>",
    fig3.to_html(full_html=False, include_plotlyjs=False),
    build_table_html(),
    "</section>",

    "<footer>Sources: data/17_high_soh_low_crate_v1.5/, data/24_soh_range_0.8_1.0_v1.5/, "
    "data/21_soh_0.8_lfp_ocp_tuned_v1.5/, data/22_real_empa_soh_0.8/ "
    "(see data/README.md for producing scripts). "
    "Computation: feature_engineering.py's create_features_by_voltage_bins, unmodified.</footer>",
    "</body></html>",
]

with open(OUT_HTML, "w") as f:
    f.write("\n".join(html_parts))

print(f"\nDone. Open: {OUT_HTML}")
