# ==============================================================
# BatteryLab Prototype — Full app with in-tab Copilot + data-aware replies
# ==============================================================

import io
import json
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import random, time  # for Copilot typing effect

# PDF export
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

# Matplotlib for plots we embed in the PDF
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Optional (MAT/ICA features)
try:
    from scipy.signal import savgol_filter, find_peaks, peak_widths
    from scipy.io import loadmat, savemat
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

# Your temperature-aware engine module
from batterylab_recipe_engine import ElectrodeSpec, CellDesignInput, design_pouch

# ---------------------------
# Streamlit page config
# ---------------------------
st.set_page_config(page_title="BatteryLab Prototype", page_icon=None, layout="wide")
st.title("BatteryLab — Prototype")
st.caption(
    "Recipe -> Performance (with temperature advisories) and Data Analytics: "
    "Upload -> analysis first (richness and next steps) -> button to visualize -> plots, features, interpretations, PDF export."
)

# ---- Copilot chat memory ----
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role": "user"/"assistant", "text": str}

def _copilot_add(role: str, text: str):
    st.session_state.chat_history.append({"role": role, "text": text})

# ---- Copilot data caches (latest outputs) ----
if "latest_design" not in st.session_state:
    st.session_state.latest_design = None   # {"spec_summary": {...}, "result": {...}}
if "latest_analytics" not in st.session_state:
    st.session_state.latest_analytics = None  # {"features_by_group": {...}, "vc_all": DataFrame, "ica_all": DataFrame}

# =========================
# Helpers (shared)
# =========================
def _standardize_columns(df: pd.DataFrame):
    cols_l = [c.lower() for c in df.columns]
    vcol = None
    for i, c in enumerate(cols_l):
        if c in ["v", "volt", "voltage", "voltage_v"]:
            vcol = df.columns[i]; break
    qcol = None
    for i, c in enumerate(cols_l):
        if "capacity" in c or c in ["q", "ah", "mah", "capacity_ah", "capacity_mah"]:
            qcol = df.columns[i]; break
    cyc = None
    for i, c in enumerate(cols_l):
        if c in ["cycle", "label", "group"]:
            cyc = df.columns[i]; break
    return vcol, qcol, cyc

def _prep_series(voltage, capacity):
    V = pd.to_numeric(voltage, errors="coerce").to_numpy()
    Qraw = pd.to_numeric(capacity, errors="coerce").to_numpy()
    Q = Qraw / 1000.0 if np.nanmax(Qraw) > 100 else Qraw
    mask = np.isfinite(V) & np.isfinite(Q)
    V = V[mask]; Q = Q[mask]
    order = np.argsort(V)
    return V[order], Q[order]

def _extract_features(V, Q):
    Qs = Q.copy()
    if SCIPY_OK and len(Q) >= 11:
        try:
            Qs = savgol_filter(Q, 11, 3)
        except Exception:
            pass
    dQdV = np.gradient(Qs, V, edge_order=2)
    peak_info = {"n_peaks": 0, "voltages": [], "widths_V": []}
    if SCIPY_OK and np.isfinite(dQdV).any():
        try:
            prom = np.nanmax(np.abs(dQdV))*0.05 if np.nanmax(np.abs(dQdV))>0 else 0.0
            peaks, _ = find_peaks(dQdV, prominence=prom)
            peak_info["n_peaks"] = int(len(peaks))
            peak_info["voltages"] = [float(V[p]) for p in peaks]
            if len(peaks) > 0:
                widths, _, _, _ = peak_widths(dQdV, peaks, rel_height=0.5)
                if len(V) > 1:
                    dv = np.mean(np.diff(V))
                    peak_info["widths_V"] = [float(w*dv) for w in widths]
        except Exception:
            pass
    try:
        dVdQ = np.gradient(V, Qs, edge_order=2)
        dVdQ_med = float(np.nanmedian(np.abs(dVdQ)))
    except Exception:
        dVdQ_med = float("nan")
    return dQdV, peak_info, dVdQ_med

def _compare_two_sets(name_a, feat_a, name_b, feat_b):
    interp = []
    cap_a = feat_a.get("cap_range_Ah", [np.nan, np.nan])[1]
    cap_b = feat_b.get("cap_range_Ah", [np.nan, np.nan])[1]
    if np.isfinite(cap_a) and np.isfinite(cap_b) and cap_a > 0:
        fade_pct = 100.0 * (cap_a - cap_b) / cap_a
        if abs(fade_pct) >= 3:
            interp.append(f"Capacity change from {name_a} to {name_b}: {fade_pct:.1f}% (negative = fade).")
    Va = feat_a.get("ica_peak_voltages_V", [])
    Vb = feat_b.get("ica_peak_voltages_V", [])
    if Va and Vb:
        n = min(len(Va), len(Vb))
        if n >= 1:
            mean_shift_mV = 1000.0 * float(np.nanmean(np.array(Vb[:n]) - np.array(Va[:n])))
            if abs(mean_shift_mV) >= 5:
                direction = "up" if mean_shift_mV > 0 else "down"
                interp.append(f"Average ICA peak shift {direction} ~{abs(mean_shift_mV):.0f} mV ({name_b} vs {name_a}).")
    Wa = feat_a.get("ica_peak_widths_V", [])
    Wb = feat_b.get("ica_peak_widths_V", [])
    if Wa and Wb:
        n = min(len(Wa), len(Wb))
        if n >= 1:
            mean_broad_mV = 1000.0 * float(np.nanmean(np.array(Wb[:n]) - np.array(Wa[:n])))
            if mean_broad_mV > 2:
               interp.append(f"ICA peak broadening ~{mean_broad_mV:.0f} mV ({name_b} vs {name_a}).")

    iva = feat_a.get("dVdQ_median_abs", np.nan)
    ivb = feat_b.get("dVdQ_median_abs", np.nan)
    if np.isfinite(iva) and np.isfinite(ivb) and ivb > iva * 1.05:
        interp.append(f"Median |dV/dQ| increased ({name_b} vs {name_a}).")
    if not interp:
        interp.append(f"No strong differences detected between {name_a} and {name_b} within prototype sensitivity.")
    return interp

# Plot -> PNG bytes for embedding in PDF
def _plot_to_bytes(df_all, ycol, title):
    fig, ax = plt.subplots(figsize=(6.2, 3.8), dpi=150)
    for c in df_all["Cycle"].unique():
        sub = df_all[df_all["Cycle"] == c]
        ax.plot(sub["Voltage"], sub[ycol], label=str(c))
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel(ycol)
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def generate_pdf_report(features_by_group, richness_notes, suggestions, interps, vc_all, ica_all):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=24, leftMargin=24, topMargin=24, bottomMargin=24)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("BatteryLab Analytics Report", styles["Title"]))
    elements.append(Spacer(1, 8))

    elements.append(Paragraph("Dataset Quality & Richness", styles["Heading2"]))
    if richness_notes:
        for r in richness_notes:
            elements.append(Paragraph("- " + r, styles["Normal"]))
    else:
        elements.append(Paragraph("No specific richness notes.", styles["Normal"]))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("Next-Step Suggestions", styles["Heading2"]))
    if suggestions:
        for s in suggestions:
            elements.append(Paragraph("- " + s, styles["Normal"]))
    else:
        elements.append(Paragraph("No suggestions generated.", styles["Normal"]))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("Key Features by Curve", styles["Heading2"]))
    table_data = [["Curve", "n_samples", "V Range (V)", "Capacity Range (Ah)",
                   "ICA Peaks", "ICA Voltages (V)", "ICA Widths (V)", "Median |dV/dQ|"]]
    for g, f in features_by_group.items():
        table_data.append([
            str(g),
            f['n_samples'],
            f"{f['voltage_range_V'][0]:.2f}–{f['voltage_range_V'][1]:.2f}",
            f"{f['cap_range_Ah'][0]:.2f}–{f['cap_range_Ah'][1]:.2f}",
            f['ica_peaks_count'],
            ", ".join([f"{v:.3f}" for v in f['ica_peak_voltages_V']]) if f['ica_peak_voltages_V'] else "—",
            ", ".join([f"{w:.3f}" for w in f['ica_peak_widths_V']]) if f['ica_peak_widths_V'] else "—",
            f"{f['dVdQ_median_abs']:.4f}" if np.isfinite(f['dVdQ_median_abs']) else "—"
        ])
    tbl = Table(table_data, repeatRows=1)
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.25, colors.black),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('ALIGN', (1,1), (-1,-1), 'CENTER')
    ]))
    elements.append(tbl)
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("AI-style Interpretations", styles["Heading2"]))
    if interps:
        for i in interps:
            elements.append(Paragraph("- " + i, styles["Normal"]))
    else:
        elements.append(Paragraph("No interpretations generated.", styles["Normal"]))
    elements.append(Spacer(1, 6))

    elements.append(Paragraph("Visualizations", styles["Heading2"]))
    if vc_all is not None and len(vc_all) > 0:
        elements.append(Paragraph("Voltage vs Capacity", styles["Heading3"]))
        img_bytes = _plot_to_bytes(vc_all, "Capacity_Ah", "Voltage vs Capacity")
        elements.append(Image(io.BytesIO(img_bytes), width=420, height=270))
    if ica_all is not None and len(ica_all) > 0:
        elements.append(Paragraph("ICA: dQ/dV vs Voltage", styles["Heading3"]))
        img_bytes = _plot_to_bytes(ica_all, "dQdV", "ICA: dQ/dV vs Voltage")
        elements.append(Image(io.BytesIO(img_bytes), width=420, height=270))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

def generate_recipe_pdf(spec_summary, result):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, rightMargin=24, leftMargin=24, topMargin=24, bottomMargin=24)
    styles = getSampleStyleSheet()
    elements = []

    # Title
    elements.append(Paragraph("BatteryLab Recipe Report", styles["Title"]))
    elements.append(Spacer(1, 8))

    # Inputs
    elements.append(Paragraph("Design Inputs", styles["Heading2"]))
    geom_txt = ""
    if spec_summary.get("area_cm2") is not None:
        geom_txt = f"Direct area: {spec_summary.get('area_cm2')} cm2"
    else:
        geom_txt = f"W x H: {spec_summary.get('width_mm')} x {spec_summary.get('height_mm')} mm"

    rows = [
        ["Geometry", geom_txt],
        ["Layers", f"{spec_summary.get('n_layers')}"],
        ["N/P ratio", f"{spec_summary.get('n_p_ratio'):.2f}"],
        ["Cathode", f"{spec_summary.get('cathode_material')} — thickness {spec_summary.get('cathode_thk_um')} um, porosity {spec_summary.get('cathode_por'):.2f}"],
        ["Anode", f"{spec_summary.get('anode_material')} (Si {spec_summary.get('anode_si_frac'):.2f}) — thickness {spec_summary.get('anode_thk_um')} um, porosity {spec_summary.get('anode_por'):.2f}"],
        ["Separator", f"{spec_summary.get('sep_thk_um')} um, porosity {spec_summary.get('sep_por'):.2f}"],
        ["Foils", f"Al {spec_summary.get('foil_al_um')} um, Cu {spec_summary.get('foil_cu_um')} um"],
        ["Electrolyte", f"{spec_summary.get('electrolyte')}"],
        ["Ambient", f"{spec_summary.get('ambient_C')} C"],
    ]

    table = Table([["Parameter", "Value"]] + rows, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.25, colors.black),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE')
    ]))
    elements.append(table)
    elements.append(Spacer(1, 8))

    # Results
    elements.append(Paragraph("Computed Performance", styles["Heading2"]))
    perf_rows = [
        ["Capacity (Ah)", f"{result['electrochem']['capacity_Ah']:.2f}"],
        ["Nominal Voltage (V)", f"{result['electrochem']['V_nom']:.2f}"],
        ["Wh/kg", f"{result['electrochem']['Wh_per_kg']:.0f}"],
        ["Wh/L", f"{result['electrochem']['Wh_per_L']:.0f}"],
        ["DeltaT @1C (C)", f"{result['thermal']['deltaT_1C_C']:.2f}"],
        ["DeltaT @3C (C)", f"{result['thermal']['deltaT_3C_C']:.2f}"],
    ]
    t2 = Table([["Metric", "Value"]] + perf_rows, repeatRows=1)
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('GRID', (0,0), (-1,-1), 0.25, colors.black),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
    ]))
    elements.append(t2)
    elements.append(Spacer(1, 8))

    # Feasibility
    elements.append(Paragraph("Mechanical & Feasibility", styles["Heading2"]))
    fz = result.get("feasibility", {})
    mech = result.get("mechanical", {})
    feas_points = [
        f"Swelling flag: {fz.get('swelling_flag')}",
        f"Thermal @3C: {fz.get('thermal_flag_3C')}",
        f"Swelling % @ 100% SOC: {round(mech.get('swelling_pct_100SOC', float('nan')), 2)}%",
    ]
    for p in feas_points:
        elements.append(Paragraph("- " + str(p), styles["Normal"]))
    elements.append(Spacer(1, 6))

    # Temperature advisories
    elements.append(Paragraph("Temperature Advisories", styles["Heading2"]))
    tg = result.get("temperature_guidance", {})
    ea = result.get("electrochem_temp_adjusted", {})
    t_lines = [
        f"Ambient: {tg.get('ambient_C', '—')} C",
        f"Ideal window: {tg.get('ideal_low_C','—')}–{tg.get('ideal_high_C','—')} C",
        f"Effective Capacity @ ambient: {ea.get('effective_capacity_Ah_at_ambient', '—')}",
        f"Relative Power vs 25 C: {ea.get('relative_power_vs_25C', '—')}x",
    ]
    if tg.get("cold_temp_risk", False):
        t_lines.append("Cold-condition risk (<= 0 C): expect higher impedance/lower power; pre-heat or derate C-rate.")
    if tg.get("high_temp_risk", False):
        t_lines.append("High ambient (>= 45 C): accelerated side reactions; consider high-temp electrolyte, charge derating, better cooling.")
    for p in t_lines:
        elements.append(Paragraph("- " + str(p), styles["Normal"]))
    elements.append(Spacer(1, 6))

    # AI Suggestions
    elements.append(Paragraph("AI Suggestions", styles["Heading2"]))
    for s in result.get("ai_suggestions", []):
        elements.append(Paragraph("- " + str(s), styles["Normal"]))

    doc.build(elements)
    buffer.seek(0)
    return buffer.getvalue()

# ---------- Data-aware utilities for Copilot ----------
def _compute_fade_pct(features_by_group: dict) -> str:
    if not features_by_group or len(features_by_group) < 2:
        return "I need at least two curves (e.g., Fresh vs Aged) to compute fade."
    keys = list(features_by_group.keys())
    def _k(s):
        import re
        nums = re.findall(r"\d+", str(s))
        return int(nums[0]) if nums else 0
    keys_sorted = sorted(keys, key=_k)
    a, b = keys_sorted[0], keys_sorted[-1]
    cap_a = features_by_group[a]["cap_range_Ah"][1]
    cap_b = features_by_group[b]["cap_range_Ah"][1]
    if not np.isfinite(cap_a) or not np.isfinite(cap_b) or cap_a <= 0:
        return "Couldn’t compute fade from the current features."
    fade = 100.0 * (cap_a - cap_b) / cap_a
    sign = "fade" if fade >= 0 else "gain"
    return f"Capacity {sign}: {abs(fade):.1f}% (comparing '{a}' → '{b}')."

def _compute_peak_shift(features_by_group: dict) -> str:
    if not features_by_group or len(features_by_group) < 2:
        return "I need at least two curves to estimate ICA peak shifts."
    keys = list(features_by_group.keys())
    def _k(s):
        import re
        nums = re.findall(r"\d+", str(s))
        return int(nums[0]) if nums else 0
    keys_sorted = sorted(keys, key=_k)
    a, b = keys_sorted[0], keys_sorted[-1]
    Va = np.array(features_by_group[a]["ica_peak_voltages_V"] or [], dtype=float)
    Vb = np.array(features_by_group[b]["ica_peak_voltages_V"] or [], dtype=float)
    if len(Va) == 0 or len(Vb) == 0:
        return "No ICA peaks detected (or insufficient resolution) to compute peak shifts."
    n = min(len(Va), len(Vb))
    shift_mV = 1000.0 * float(np.nanmean(Vb[:n] - Va[:n]))
    direction = "up" if shift_mV > 0 else "down"
    return f"Mean ICA peak shift: {abs(shift_mV):.0f} mV {direction} ('{b}' vs '{a}')."

def _compute_broadening(features_by_group: dict) -> str:
    if not features_by_group or len(features_by_group) < 2:
        return "I need at least two curves to estimate peak broadening."
    keys = list(features_by_group.keys())
    def _k(s):
        import re
        nums = re.findall(r"\d+", str(s))
        return int(nums[0]) if nums else 0
    keys_sorted = sorted(keys, key=_k)
    a, b = keys_sorted[0], keys_sorted[-1]
    Wa = np.array(features_by_group[a]["ica_peak_widths_V"] or [], dtype=float)
    Wb = np.array(features_by_group[b]["ica_peak_widths_V"] or [], dtype=float)
    if len(Wa) == 0 or len(Wb) == 0:
        return "Not enough peak width info to estimate broadening."
    n = min(len(Wa), len(Wb))
    broad_mV = 1000.0 * float(np.nanmean(Wb[:n] - Wa[:n]))
    tag = "broadened" if broad_mV >= 0 else "narrowed"
    return f"ICA peaks {tag} by ~{abs(broad_mV):.0f} mV ('{b}' vs '{a}')."

def _summarize_design_temperature(result: dict) -> str:
    if not result:
        return "Run a design first — I don’t have a computed pouch result yet."
    tg = result.get("temperature_guidance", {}) or {}
    ea = result.get("electrochem_temp_adjusted", {}) or {}
    parts = []
    parts.append(f"Ambient {tg.get('ambient_C','?')}°C; ideal window {tg.get('ideal_low_C','?')}–{tg.get('ideal_high_C','?')}°C.")
    if "effective_capacity_Ah_at_ambient" in ea:
        parts.append(f"Effective capacity @ ambient ~ {ea['effective_capacity_Ah_at_ambient']:.2f} Ah.")
    if "relative_power_vs_25C" in ea:
        parts.append(f"Relative power vs 25°C ~ {ea['relative_power_vs_25C']:.2f}×.")
    if tg.get("cold_temp_risk"): parts.append("Cold-risk flagged → pre-heat or derate C-rate.")
    if tg.get("high_temp_risk"): parts.append("High-temp risk → consider cooling, high-temp electrolyte, charge derating.")
    return " ".join(parts) or "No temperature guidance available."

# =========================
# Data-aware Copilot
# =========================
def _copilot_reply(user_text: str, context: str = "general") -> str:
    """
    Data-aware Copilot:
      - understands a few intents
      - pulls from st.session_state.latest_design / latest_analytics when available
    """
    text = (user_text or "").strip().lower()

    # quick intent detection
    wants_help   = any(k in text for k in ["/help", "help", "what can you do", "commands"])
    wants_fade   = any(k in text for k in ["fade", "capacity drop", "soh"])
    wants_shift  = any(k in text for k in ["peak shift", "ica shift", "peak position"])
    wants_broad  = any(k in text for k in ["broadening", "width", "fwhm"])
    wants_temp   = any(k in text for k in ["temperature", "thermal", "ambient", "cooling", "heating"])
    wants_reco   = any(k in text for k in ["recommend", "suggest", "next step", "what next"])
    wants_best   = any(k in text for k in ["best curve", "which curve", "highest capacity", "lowest fade"])

    # HELP
    if wants_help:
        return (
            "Here’s what I can do now:\n"
            "• Analytics: fade %, ICA peak shifts, broadening, quick insights (/help, 'fade', 'peak shift', 'broadening').\n"
            "• Design: summarize temperature risks & adjustments ('temperature summary').\n"
            "• Recommendations: next steps for analysis or design ('recommend next step').\n"
            "Tip: upload or analyze data first (Analytics tab), or run Compute Performance (Design tab)."
        )

    # pull caches
    la = st.session_state.latest_analytics
    ld = st.session_state.latest_design

    # ANALYTICS INTENTS
    if wants_fade and la and la.get("features_by_group"):
        return _compute_fade_pct(la["features_by_group"])

    if wants_shift and la and la.get("features_by_group"):
        return _compute_peak_shift(la["features_by_group"])

    if wants_broad and la and la.get("features_by_group"):
        return _compute_broadening(la["features_by_group"])

    if wants_best and la and la.get("features_by_group"):
        fbg = la["features_by_group"]
        best = max(fbg.items(), key=lambda kv: kv[1]["cap_range_Ah"][1])
        return f"Best curve by capacity: '{best[0]}' (max ~{best[1]['cap_range_Ah'][1]:.2f} Ah)."

    # DESIGN INTENTS
    if wants_temp and ld and ld.get("result"):
        return _summarize_design_temperature(ld["result"])

    if wants_reco:
        if context == "design" and ld and ld.get("result"):
            tg = ld["result"].get("temperature_guidance", {}) or {}
            recos = []
            if tg.get("high_temp_risk"): recos.append("Improve cooling or reduce charge C-rate; consider high-temp electrolyte/additives.")
            if tg.get("cold_temp_risk"): recos.append("Pre-heat or lower discharge C-rate; increase porosity for low-T power.")
            recos.append("Try ±5–10 µm cathode thickness and ±0.02 porosity sweep, compare Wh/L vs ΔT@3C.")
            return " • ".join(recos) if recos else "Looks stable—try small thickness/porosity sweeps to optimize Wh/L vs ΔT."
        if context == "analytics" and la and la.get("features_by_group"):
            return (
                "1) Quantify fade and peak shifts (‘fade’, ‘peak shift’). "
                "2) If IR/temperature available, correlate with |dV/dQ| median. "
                "3) Export MAT + repro script, then fit a simple regressor for EOL prediction."
            )

    # fallbacks by context
    if context == "design":
        return "Design Copilot here. Try: 'temperature summary', 'recommend next step'."
    if context == "analytics":
        return "Analytics Copilot here. Try: 'fade', 'peak shift', 'broadening', or 'recommend next step'."

    return "Got it! Use /help to see data-aware commands."

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2 = st.tabs(["Recipe -> Performance", "Data Analytics (CSV/MAT)"])

# ==========
# In-tab Copilot (sticky right column) – shared memory via st.session_state.chat_history
# ==========
def render_copilot(context_key: str, default_context: str = "Design"):
    radio_key = f"ctx_{context_key}"
    input_key = f"msg_{context_key}"
    send_key  = f"send_{context_key}"

    st.markdown(
        """
        <style>
          .blab-box { position: sticky; top: 70px; max-height: 78vh; overflow-y: auto; 
                      background: #f7f7f9; border: 1px solid #ddd; padding: 12px 14px; border-radius: 8px; }
          .blab-bubble { padding: 8px 10px; border-radius: 10px; margin-bottom: 8px; line-height: 1.45; }
          .blab-user { background:#DCF8C6; text-align:right; }
          .blab-bot  { background:#fff; border:1px solid #e6e6e6; }
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown("### BatteryLAB Copilot 💬")
    st.caption("Always-on chat for this tab. Brainstorm designs, request plots, or ask next steps.")

    ctx = st.radio(
        "Context",
        ["Design", "Analytics"],
        index=(0 if default_context.lower() == "design" else 1),
        horizontal=True,
        key=radio_key
    )

    with st.container():
        st.markdown('<div class="blab-box">', unsafe_allow_html=True)
        for msg in st.session_state.chat_history[-12:]:
            cls = "blab-user" if msg["role"] == "user" else "blab-bot"
            who = "You" if msg["role"] == "user" else "Copilot"
            st.markdown(f'<div class="blab-bubble {cls}"><b>{who}:</b> {msg["text"]}</div>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    user_text = st.text_input(
        "Type a message…",
        key=input_key,
        placeholder="e.g. Try Si=0.05 and lower anode porosity by 0.02"
    )
    if st.button("Send", key=send_key, use_container_width=True):
        if user_text.strip():
            _copilot_add("user", user_text.strip())
            reply = _copilot_reply(user_text, context=("design" if ctx == "Design" else "analytics"))
            _copilot_add("assistant", reply)

# =========================
# TAB 1: Recipe -> Performance (temperature-aware) WITH in-tab Copilot
# =========================
with tab1:
    col_main, col_chat = st.columns([0.68, 0.32], gap="large")

    with col_main:
        PRESETS = {
            "LFP 2.5Ah (demo)": {
                "cathode": {"material": "LFP", "thk": 70, "por": 0.35},
                "anode":   {"material": "Graphite", "thk": 85, "por": 0.40, "si": 0.00},
                "geom":    {"mode": "area", "area_cm2": 100.0, "n_layers": 36},
                "sep":     {"thk": 20, "por": 0.45}, "foil": {"al": 15, "cu": 8},
                "np": 1.10, "elyte": "1M LiPF6 in EC:EMC 3:7", "amb": 25
            },
            "NMC811 3Ah (demo)": {
                "cathode": {"material": "NMC811", "thk": 75, "por": 0.33},
                "anode":   {"material": "Graphite", "thk": 90, "por": 0.40, "si": 0.05},
                "geom":    {"mode": "area", "area_cm2": 110.0, "n_layers": 32},
                "sep":     {"thk": 20, "por": 0.45}, "foil": {"al": 15, "cu": 8},
                "np": 1.08, "elyte": "1M LiPF6 + 2% VC in EC:DEC", "amb": 25
            }
        }

        with st.sidebar:
            st.header("Quick Start")
            preset = st.selectbox("Preset", ["— none —"] + list(PRESETS.keys()))

            defaults = {
                "geom_mode": "area",
                "area_cm2": 100.0, "width_mm": 70.0, "height_mm": 100.0,
                "n_layers": 36, "n_p_ratio": 1.10,
                "electrolyte": "1M LiPF6 in EC:EMC 3:7", "ambient_C": 25,
                "cath_mat": "LFP", "cath_thk": 70, "cath_por": 0.35,
                "anode_mat": "Graphite", "anode_thk": 85, "anode_por": 0.40, "anode_si": 0.00,
                "sep_thk": 20, "sep_por": 0.45, "foil_al": 15, "foil_cu": 8,
            }

            if preset != "— none —":
                p = PRESETS[preset]
                defaults.update({
                    "geom_mode": p["geom"]["mode"],
                    "area_cm2": p["geom"]["area_cm2"], "n_layers": p["geom"]["n_layers"],
                    "n_p_ratio": p["np"], "electrolyte": p["elyte"], "ambient_C": p["amb"],
                    "cath_mat": p["cathode"]["material"], "cath_thk": p["cathode"]["thk"], "cath_por": p["cathode"]["por"],
                    "anode_mat": p["anode"]["material"], "anode_thk": p["anode"]["thk"],
                    "anode_por": p["anode"]["por"], "anode_si": p["anode"]["si"],
                    "sep_thk": p["sep"]["thk"], "sep_por": p["sep"]["por"],
                    "foil_al": p["foil"]["al"], "foil_cu": p["foil"]["cu"],
                })

            st.header("Cell Geometry")
            area_mode = st.radio("Area Input Mode", ["Direct area (cm2)", "Width x Height (mm)"],
                                 index=0 if defaults["geom_mode"] == "area" else 1)

            if area_mode == "Direct area (cm2)":
                area_cm2 = st.number_input("Layer area (cm2)", min_value=10.0, value=float(defaults["area_cm2"]), step=5.0)
                dims = {"area_cm2": area_cm2}
            else:
                width_mm = st.number_input("Width (mm)", min_value=10.0, value=float(defaults["width_mm"]), step=1.0)
                height_mm = st.number_input("Height (mm)", min_value=10.0, value=float(defaults["height_mm"]), step=1.0)
                dims = {"width_mm": width_mm, "height_mm": height_mm}

            n_layers = st.number_input("# Layers", min_value=2, value=int(defaults["n_layers"]), step=2)
            n_p_ratio = st.slider("N/P ratio", 1.00, 1.30, float(defaults["n_p_ratio"]), 0.01)
            electrolyte = st.text_input("Electrolyte (free text)", defaults["electrolyte"])
            ambient_C = st.slider("Ambient Temp (C)", -20, 60, int(defaults["ambient_C"]), 1)

        st.subheader("Cathode")
        cathode_material = st.selectbox("Material (Cathode)", ["LFP", "NMC811"],
                                        index=(0 if defaults["cath_mat"] == "LFP" else 1))
        cathode_thk = st.slider("Cathode thickness (um)", 20, 140, int(defaults["cath_thk"]), 1)
        cathode_por = st.slider("Cathode porosity", 0.20, 0.60, float(defaults["cath_por"]), 0.01)

        st.subheader("Anode")
        anode_material = st.selectbox("Material (Anode)", ["Graphite"], index=0)
        anode_thk = st.slider("Anode thickness (um)", 20, 140, int(defaults["anode_thk"]), 1)
        anode_por = st.slider("Anode porosity", 0.20, 0.60, float(defaults["anode_por"]), 0.01)
        anode_si = st.slider("Anode silicon fraction (0..1)", 0.0, 0.20, float(defaults["anode_si"]), 0.01)

        st.subheader("Separator & Foils")
        sep_thk = st.slider("Separator thickness (um)", 10, 40, int(defaults["sep_thk"]), 1)
        sep_por = st.slider("Separator porosity", 0.20, 0.70, float(defaults["sep_por"]), 0.01)
        foil_al = st.slider("Cathode Al foil (um)", 8, 20, int(defaults["foil_al"]), 1)
        foil_cu = st.slider("Anode Cu foil (um)", 4, 15, int(defaults["foil_cu"]), 1)

        if cathode_thk < 20 or anode_thk < 20:
            st.warning("Very thin coatings may make predictions less reliable.")
        if not (0.2 <= cathode_por <= 0.6 and 0.2 <= anode_por <= 0.6):
            st.warning("Porosity outside typical ranges may reduce accuracy.")

        run = st.button("Compute Performance", type="primary")

        if run:
            with st.spinner("Computing physics + AI suggestions..."):
                cathode = ElectrodeSpec(material=cathode_material, thickness_um=cathode_thk, porosity=cathode_por, active_frac=0.96)
                anode   = ElectrodeSpec(material=anode_material, thickness_um=anode_thk, porosity=anode_por, active_frac=0.96, silicon_frac=anode_si)
                spec = CellDesignInput(
                    cathode=cathode, anode=anode, n_layers=int(n_layers),
                    separator_thickness_um=sep_thk, separator_porosity=sep_por, n_p_ratio=float(n_p_ratio),
                    cathode_foil_um=foil_al, anode_foil_um=foil_cu, electrolyte=electrolyte, ambient_C=float(ambient_C),
                    **dims
                )
                result = design_pouch(spec)

            st.success("Computed successfully!")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Capacity (Ah)", f"{result['electrochem']['capacity_Ah']:.2f}")
                st.metric("Nominal Voltage (V)", f"{result['electrochem']['V_nom']:.2f}")
            with col2:
                st.metric("Wh/kg", f"{result['electrochem']['Wh_per_kg']:.0f}")
                st.metric("Wh/L", f"{result['electrochem']['Wh_per_L']:.0f}")
            with col3:
                st.metric("DeltaT @1C (C)", f"{result['thermal']['deltaT_1C_C']:.2f}")
                st.metric("DeltaT @3C (C)", f"{result['thermal']['deltaT_3C_C']:.2f}")

            st.markdown("### Mechanical & Feasibility")
            fz = result["feasibility"]
            cols = st.columns(3)
            cols[0].markdown(f"Swelling flag: {fz['swelling_flag']}")
            cols[1].markdown(f"Thermal @3C: {fz['thermal_flag_3C']}")
            cols[2].markdown(f"Swelling % @100% SOC: {round(result['mechanical']['swelling_pct_100SOC'],2)}%")

            # Temperature advisories
            st.markdown("### Temperature Advisories")
            tg = result.get("temperature_guidance", {})
            ea = result.get("electrochem_temp_adjusted", {})

            cols_t = st.columns(4)
            cols_t[0].markdown(f"Ambient: {tg.get('ambient_C', '—')} C")
            cols_t[1].markdown(f"Ideal window: {tg.get('ideal_low_C','—')}–{tg.get('ideal_high_C','—')} C")
            try:
                eff_cap = float(ea.get('effective_capacity_Ah_at_ambient', float('nan')))
                cols_t[2].markdown(f"Effective Capacity @ ambient: {eff_cap:.2f} Ah")
            except Exception:
                cols_t[2].markdown("Effective Capacity @ ambient: —")
            try:
                rel_pow = float(ea.get('relative_power_vs_25C', float('nan')))
                cols_t[3].markdown(f"Relative Power vs 25 C: {rel_pow:.2f}x")
            except Exception:
                cols_t[3].markdown("Relative Power vs 25 C: —")

            risk_msgs = []
            if tg.get("cold_temp_risk", False):
                risk_msgs.append("Cold-condition risk (<= 0 C): expect higher impedance/lower power; pre-heat or derate C-rate.")
            if tg.get("high_temp_risk", False):
                risk_msgs.append("High ambient (>= 45 C): accelerated side reactions; consider high-temp electrolyte, charge derating, better cooling.")
            if not risk_msgs:
                risk_msgs.append("Ambient in acceptable range for typical operation.")
            for m in risk_msgs:
                st.write("- " + m)

            st.markdown("### AI Suggestions")
            for s in result["ai_suggestions"]:
                st.write("- " + s)

            # Build a compact design-specs summary for the PDF
            spec_summary = {
                "area_cm2": (dims.get("area_cm2") if "area_cm2" in dims else None),
                "width_mm": (dims.get("width_mm") if "width_mm" in dims else None),
                "height_mm": (dims.get("height_mm") if "height_mm" in dims else None),
                "n_layers": int(n_layers),
                "n_p_ratio": float(n_p_ratio),
                "cathode_material": cathode_material,
                "cathode_thk_um": int(cathode_thk),
                "cathode_por": float(cathode_por),
                "anode_material": anode_material,
                "anode_thk_um": int(anode_thk),
                "anode_por": float(anode_por),
                "anode_si_frac": float(anode_si),
                "sep_thk_um": int(sep_thk),
                "sep_por": float(sep_por),
                "foil_al_um": int(foil_al),
                "foil_cu_um": int(foil_cu),
                "electrolyte": electrolyte,
                "ambient_C": float(ambient_C),
            }

            pdf_recipe = generate_recipe_pdf(spec_summary, result)
            st.download_button(
                "Download Recipe Report (PDF)",
                data=pdf_recipe,
                file_name="BatteryLab_recipe_report.pdf",
                mime="application/pdf"
            )

            # Store for Copilot (design cache)
            st.session_state.latest_design = {"spec_summary": spec_summary, "result": result}

        else:
            st.info("Pick a preset for a 1-click demo, or set your recipe parameters, then press Compute Performance.")

    with col_chat:
        render_copilot(context_key="tab1", default_context="Design")

# =========================
# TAB 2: Data Analytics WITH in-tab Copilot + exports (.mat, .py)
# =========================
with tab2:
    col_main, col_chat = st.columns([0.68, 0.32], gap="large")

    with col_main:
        st.subheader("Upload a dataset")
        st.write(
            "Accepted: .csv (recommended) or .mat (if SciPy is available). "
            "Columns: Voltage and Capacity_Ah (or Capacity_mAh). Optional: Cycle (e.g., Fresh/Aged)."
        )

        up = st.file_uploader("Upload CSV or MAT file", type=["csv", "mat"])
        if up is not None:
            with st.spinner("Parsing and extracting features..."):
                df = None
                name = up.name.lower()
                if name.endswith(".csv"):
                    df = pd.read_csv(up)
                elif name.endswith(".mat"):
                    if not SCIPY_OK:
                        st.error("SciPy not available. Please upload a CSV for now.")
                    else:
                        mat = loadmat(io.BytesIO(up.getvalue()))
                        candidates = {k: np.squeeze(v) for k, v in mat.items()
                                      if isinstance(v, np.ndarray) and v.size > 3}
                        V = None; Q = None
                        for key, arr in candidates.items():
                            lk = key.lower()
                            if V is None and ("volt" in lk or lk == "v"):
                                V = arr
                            if Q is None and ("cap" in lk or lk in ["q","capacity","capacity_ah","capacity_mah"]):
                                Q = arr
                        if V is None or Q is None:
                            st.error("Could not find voltage/capacity arrays in .mat. Use CSV with Voltage, Capacity_Ah.")
                        else:
                            df = pd.DataFrame({"Voltage": V, "Capacity": Q})

                if df is not None:
                    vcol, qcol, cyc = _standardize_columns(df)
                    if vcol is None or qcol is None:
                        st.error("Please include Voltage and Capacity_Ah (or Capacity_mAh).")
                        st.stop()
                    if cyc is None:
                        df["_Cycle"] = "Curve1"
                        cyc = "_Cycle"

                    groups = list(df[cyc].astype(str).unique())
                    features_by_group = {}
                    vc_all_rows, ica_all_rows = [], []

                    for g in groups:
                        sub = df[df[cyc].astype(str) == g]
                        V, Q = _prep_series(sub[vcol], sub[qcol])
                        if len(V) < 5:
                            continue
                        dQdV, peak_info, dVdQ_med = _extract_features(V, Q)
                        features_by_group[g] = {
                            "n_samples": int(len(V)),
                            "voltage_range_V": [float(np.nanmin(V)), float(np.nanmax(V))],
                            "cap_range_Ah": [float(np.nanmin(Q)), float(np.nanmax(Q))],
                            "ica_peaks_count": int(peak_info.get("n_peaks", 0)),
                            "ica_peak_voltages_V": peak_info.get("voltages", []),
                            "ica_peak_widths_V": peak_info.get("widths_V", []),
                            "dVdQ_median_abs": dVdQ_med,
                        }
                        vc_all_rows.append(pd.DataFrame({"Voltage": V, "Capacity_Ah": Q, "Cycle": g}))
                        ica_all_rows.append(pd.DataFrame({"Voltage": V, "dQdV": dQdV, "Cycle": g}))

                    if not features_by_group:
                        st.error("Not enough valid data points to analyze.")
                        st.stop()

                    # (1) DATASET QUALITY & RICHNESS
                    st.markdown("### Dataset Quality & Richness")
                    for g, f in features_by_group.items():
                        st.write(
                            f"{g} — {f['n_samples']} points | "
                            f"V range: {f['voltage_range_V'][0]:.2f}–{f['voltage_range_V'][1]:.2f} V | "
                            f"Capacity: {f['cap_range_Ah'][0]:.2f}–{f['cap_range_Ah'][1]:.2f} Ah | "
                            f"ICA peaks: {f['ica_peaks_count']}"
                        )

                    richness_notes = []
                    if len(features_by_group) >= 2:
                        richness_notes.append("Multiple curves detected -> enables trend comparisons (fade, peak shifts, impedance).")
                    else:
                        richness_notes.append("Single curve detected -> add an aged or baseline curve for richer insights.")
                    if any(f["n_samples"] < 30 for f in features_by_group.values()):
                        richness_notes.append("Some curves have <30 points -> derivatives may be noisy; consider higher-resolution sampling.")
                    if any(f["ica_peaks_count"] == 0 for f in features_by_group.values()):
                        richness_notes.append("ICA shows few/no peaks -> may indicate smooth kinetics or insufficient resolution.")
                    for rn in richness_notes:
                        st.write("- " + rn)

                    # (2) NEXT-STEP SUGGESTIONS
                    st.markdown("### Next-Step Suggestions")
                    suggestions = []
                    if len(features_by_group) == 1:
                        suggestions = [
                            "Add a comparison curve (e.g., Fresh vs Aged, Cycle 10 vs Cycle 500) to quantify capacity fade and ICA peak shifts.",
                            "Track ICA peak positions/widths across cycles to infer LLI vs LAM vs impedance growth.",
                            "Include IR/temperature columns (if available) to correlate electro-thermal behavior."
                        ]
                    else:
                        suggestions = [
                            "Quantify fade: compute % capacity change between earliest and latest curves.",
                            "Track ICA peak shifts (mV) and broadening (mV) -> LLI and impedance indicators.",
                            "Build a simple regression using extracted features to predict end-of-life or rate performance.",
                            "If you have cycle index/time, add it to the dataset for richer trend modeling."
                        ]
                    for s in suggestions:
                        st.write("- " + s)

                    st.divider()
                    do_plots = st.button("Visualize recommended plots", type="primary")

                    if do_plots:
                        st.markdown("### Visualizations")
                        vc_all = pd.concat(vc_all_rows, ignore_index=True)
                        ica_all = pd.concat(ica_all_rows, ignore_index=True)

                        vc_chart = (
                            alt.Chart(vc_all)
                            .mark_line()
                            .encode(
                                x=alt.X("Voltage:Q", title="Voltage (V)"),
                                y=alt.Y("Capacity_Ah:Q", title="Capacity (Ah)"),
                                color=alt.Color("Cycle:N", title="Curve")
                            )
                            .properties(title="Voltage vs Capacity")
                        )

                        ica_chart = (
                            alt.Chart(ica_all)
                            .mark_line()
                            .encode(
                                x=alt.X("Voltage:Q", title="Voltage (V)"),
                                y=alt.Y("dQdV:Q", title="dQ/dV (Ah/V)"),
                                color=alt.Color("Cycle:N", title="Curve")
                            )
                            .properties(title="ICA: dQ/dV vs Voltage")
                        )

                        c1, c2 = st.columns(2)
                        with c1:
                            st.altair_chart(vc_chart, use_container_width=True)
                        with c2:
                            st.altair_chart(ica_chart, use_container_width=True)

                        st.markdown("### Key Features by Curve")
                        st.write(features_by_group)

                        st.markdown("### AI-style Interpretations (dynamic)")
                        interps = []
                        if len(features_by_group) == 1:
                            interps = [
                                "Distinct ICA peaks often indicate well-defined phase transitions.",
                                "Broadening of ICA peaks over time is a common sign of rising impedance.",
                                "Compare this curve to an earlier/later cycle to quantify fade and peak shifts."
                            ]
                        else:
                            gnames = list(features_by_group.keys())[:2]
                            fA, fB = features_by_group[gnames[0]], features_by_group[gnames[1]]
                            featA = {
                                "cap_range_Ah": fA["cap_range_Ah"],
                                "ica_peak_voltages_V": fA["ica_peak_voltages_V"],
                                "ica_peak_widths_V": fA["ica_peak_widths_V"],
                                "dVdQ_median_abs": fA["dVdQ_median_abs"],
                            }
                            featB = {
                                "cap_range_Ah": fB["cap_range_Ah"],
                                "ica_peak_voltages_V": fB["ica_peak_voltages_V"],
                                "ica_peak_widths_V": fB["ica_peak_widths_V"],
                                "dVdQ_median_abs": fB["dVdQ_median_abs"],
                            }
                            interps = _compare_two_sets(gnames[0], featA, gnames[1], featB)
                        for b in interps:
                            st.write("- " + b)

                        # ---------------------------
                        # Export: .mat data + .py repro script
                        # ---------------------------
                        def _build_repro_script_py(default_mat_name="BatteryLab_analytics_data.mat"):
                            return f"""# Reproduce BatteryLAB analytics plots
import sys
import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.io import loadmat
except Exception as e:
    raise SystemExit("This script needs SciPy: pip install scipy") from e

fname = sys.argv[1] if len(sys.argv) > 1 else "{default_mat_name}"
m = loadmat(fname, squeeze_me=True)

# Voltage–Capacity table
V1   = m['vc_voltage']
Q1   = m['vc_capacity_ah']
c1   = m['vc_cycle_idx'].astype(int)
n1   = [str(x) for x in (m['vc_cycle_names'].tolist() if hasattr(m['vc_cycle_names'], 'tolist') else m['vc_cycle_names'])]

# ICA table
V2   = m['ica_voltage']
dqdv = m['ica_dqdv']
c2   = m['ica_cycle_idx'].astype(int)
n2   = [str(x) for x in (m['ica_cycle_names'].tolist() if hasattr(m['ica_cycle_names'], 'tolist') else m['ica_cycle_names'])]

# Plot Voltage–Capacity
plt.figure()
for idx in np.unique(c1):
    mask = (c1 == idx)
    plt.plot(V1[mask], Q1[mask], label=n1[int(idx)])
plt.xlabel('Voltage (V)')
plt.ylabel('Capacity (Ah)')
plt.title('Voltage vs Capacity')
plt.legend()
plt.tight_layout()

# Plot ICA
plt.figure()
for idx in np.unique(c2):
    mask = (c2 == idx)
    plt.plot(V2[mask], dqdv[mask], label=n2[int(idx)])
plt.xlabel('Voltage (V)')
plt.ylabel('dQ/dV (Ah/V)')
plt.title('ICA: dQ/dV vs Voltage')
plt.legend()
plt.tight_layout()
plt.show()
"""

                        # Build .mat payload (handles arbitrary # of curves)
                        vc_cat = vc_all["Cycle"].astype("category")
                        ica_cat = ica_all["Cycle"].astype("category")

                        vc_cycle_idx = vc_cat.cat.codes.to_numpy().astype(np.int32)
                        vc_cycle_names = np.array(vc_cat.cat.categories.tolist(), dtype=object)

                        ica_cycle_idx = ica_cat.cat.codes.to_numpy().astype(np.int32)
                        ica_cycle_names = np.array(ica_cat.cat.categories.tolist(), dtype=object)

                        # Prepare binary buffers for both files
                        py_script_text = _build_repro_script_py()
                        py_bytes = py_script_text.encode("utf-8")

                        if SCIPY_OK:
                            mat_buf = io.BytesIO()
                            savemat(mat_buf, {
                                # VC table
                                "vc_voltage":      vc_all["Voltage"].to_numpy().astype(float),
                                "vc_capacity_ah":  vc_all["Capacity_Ah"].to_numpy().astype(float),
                                "vc_cycle_idx":    vc_cycle_idx,
                                "vc_cycle_names":  vc_cycle_names,
                                # ICA table
                                "ica_voltage":     ica_all["Voltage"].to_numpy().astype(float),
                                "ica_dqdv":        ica_all["dQdV"].to_numpy().astype(float),
                                "ica_cycle_idx":   ica_cycle_idx,
                                "ica_cycle_names": ica_cycle_names,
                            })
                            mat_buf.seek(0)
                        else:
                            mat_buf = None  # SciPy not available

                        c_dl1, c_dl2 = st.columns(2)
                        with c_dl1:
                            if SCIPY_OK and mat_buf is not None:
                                st.download_button(
                                    "Download analytics data (.mat)",
                                    data=mat_buf.getvalue(),
                                    file_name="BatteryLab_analytics_data.mat",
                                    mime="application/octet-stream"
                                )
                            else:
                                st.info("Install SciPy on the server to enable `.mat` export (pip install scipy).")

                        with c_dl2:
                            st.download_button(
                                "Download repro plot script (.py)",
                                data=py_bytes,
                                file_name="repro_analytics_plots.py",
                                mime="text/x-python"
                            )

                        # PDF Download
                        pdf_bytes = generate_pdf_report(
                            features_by_group=features_by_group,
                            richness_notes=richness_notes,
                            suggestions=suggestions,
                            interps=interps,
                            vc_all=vc_all,
                            ica_all=ica_all
                        )
                        st.download_button(
                            "Download Full Report (PDF)",
                            data=pdf_bytes,
                            file_name="BatteryLab_analytics_report.pdf",
                            mime="application/pdf"
                        )

                        # Store for Copilot (analytics cache)
                        st.session_state.latest_analytics = {
                            "features_by_group": features_by_group,
                            "vc_all": vc_all,
                            "ica_all": ica_all,
                        }
                    else:
                        st.info("Click **Visualize recommended plots** to render charts, then see Key Features, interpretations, and download the PDF report.")
        else:
            st.info("Upload a CSV with Voltage and Capacity_Ah (or Capacity_mAh). Optional: Cycle column. .mat is supported if SciPy is available.")

    with col_chat:
        render_copilot(context_key="tab2", default_context="Analytics")
